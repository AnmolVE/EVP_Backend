import os
import re
import json
import requests
from dotenv import load_dotenv
load_dotenv()

subscription_key = os.environ.get('BING_SEARCH_V7_SUBSCRIPTION_KEY')
endpoint = os.environ.get('BING_SEARCH_V7_ENDPOINT') + "/v7.0/search"

from ..models import (
    Company,
    Perception,
    Loyalty,
    Advocacy,
    Attraction,
    Influence,
    Brand,
)

from ..serializers import (
    CompanySerializer,
    PerceptionSerializer,
    LoyaltySerializer,
    AdvocacySerializer,
    AttractionSerializer,
    InfluenceSerializer,
    BrandSerializer,
)

from ..utils.chatgpt import get_data_from_chatgpt_2

bing_query_data = [
        "Headquarters",
        "Established Date",
        "About the company",
        "Industry",
        "Company History",
        "Company Financials",
        "Top 3 Competitors",
        "Number of Employees",
        "Number of Geographies",
        "LinkedIn URL and followers",
        "Instagram URL and followers",
        "Tiktok URL and followers",
        "Facebook URL and followers",
        "Twitter(X) URL and followers",
        "Glassdoor Score",
        "Online Forums Summary",
        "Number of Media Mentions",
        "Number of Awards",
        "Employee Value Proposition",
        "Culture and Values",
        "Purpose",
        "Vision",
        "Mission",
]

def extract_snippet_data(relevant_info_from_query, number_of_iteration):
    all_snippet_data = []
    for snippet_data in relevant_info_from_query[:number_of_iteration]:
        all_snippet_data.append(snippet_data.get("snippet"))
    all_snippet_data = " ".join(all_snippet_data)
    return all_snippet_data

def get_data_from_bing(company_name, fields_to_query_with_bing):
    # mkt = 'en-US'
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    relevant_info_from_query = []
    all_data_from_chatgpt_2 = {}

    try:
        for query_field in fields_to_query_with_bing:
            print("bing : ", query_field)
            snippet_data = ""
            query_params = company_name + " " + query_field
            params = {
                    'q': query_params,
                    'count': 50,
                    # "offset": i,
                    # "mkt": mkt,
                    "freshness": "Month"
                }
            response = requests.get(endpoint, headers=headers, params=params)

            response.raise_for_status()

            crawl_data = response.json()

            if(query_field == "Company Financials"):
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 5)
                data_from_chatgpt_2 = get_data_from_chatgpt_2(company_name, snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[query_field] = cleaned_result
            elif(query_field == "Top 3 Competitors"):
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 2)
                data_from_chatgpt_2 = get_data_from_chatgpt_2(company_name, snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[query_field] = cleaned_result
            elif query_field in ['LinkedIn URL and followers', "Instagram UR and followers", "Tiktok URL and followers", "Facebook URL and followers", "Twitter(X) URL and followers"]:
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 3)
                url = crawl_data.get("webPages", {}).get("value", [])[0]["url"]
                snippet_data += f" The url is : {url}"
                data_from_chatgpt_2 = get_data_from_chatgpt_2(company_name, snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[query_field] = cleaned_result
            elif query_field in ["Number of Employees", "Number of Geographies", "Employee Feedback Summary", "Glassdoor Score", "Online Forums Summary", "Number of Media Mentions", "Number of Awards", "Employee Value Proposition", "Culture and Values", "Purpose", "Vision", "Mission"]:
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 5)
                data_from_chatgpt_2 = get_data_from_chatgpt_2(company_name, snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[query_field] = cleaned_result

        return all_data_from_chatgpt_2
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def cleaning_chatgpt_info(info_from_chatgpt):
    relevant_info = json.loads(info_from_chatgpt)
    cleaned_info = {}
    for key, value in relevant_info.items(""):
        cleaned_info[key.strip("")] = value.strip("").replace('\n', '')
    return cleaned_info

def save_data_to_database(final_data, company_name, user):
    company = Company.objects.create(
            user=user,
            name=company_name,
            headquarters=final_data.get('Headquarters', ''),
            established_date=final_data.get('Established Date', ''),
            about_the_company=final_data.get('About the company', ''),
            industry=final_data.get('Industry', ''),
            company_financials=final_data.get('Company Financials', ''),
            company_history=final_data.get('Company History', ''),
            top_3_competitors=final_data.get('Top 3 Competitors', ''),
            number_of_employees = final_data.get('Number of Employees', ''),
            number_of_geographies = final_data.get('Number of Geographies', ''),
            linked_info=final_data.get('LinkedIn URL and followers', ''),
            instagram_info=final_data.get('Instagram UR and followers', ''),
            tiktok_info=final_data.get('Tiktok URL and followers', ''),
            facebook_info=final_data.get('Facebook URL and followers', ''),
            twitter_info=final_data.get('Twitter(X) URL and followers', ''),
            internal_comms_channels=final_data.get('Internal Comms Channels', ''),
            glassdoor_score=final_data.get('Glassdorr Score', ''),
            what_retains_talent=final_data.get('What Retains Talent', ''),
            what_attracts_talent=final_data.get('What Attracts Talent', ''),
            employee_value_proposition=final_data.get('Employee Value Proposition', ''),
            culture_and_values=final_data.get('Culture and Values', ''),
            purpose=final_data.get('Purpose', ''),
            customer_value_proposition=final_data.get('Customer Value Proposition', ''),
            vision=final_data.get('Vision', ''),
            mission=final_data.get('Mission', ''),
            brand_guidelines=final_data.get('Brand Guidelines', ''),
        )

    company_vector = Company.objects.get(user=user, name=company_name)
    company_vector_serializer = CompanySerializer(company_vector)

    company_id = company_vector.id

    whole_data = {
        "company_vector": company_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)

    return formatted_string
