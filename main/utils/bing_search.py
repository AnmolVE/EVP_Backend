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

bing_query_data = {
        "headquarters":"{company} Headquarters",
        "founded_date":"{company} founded date",
        "about_the_company":"About {company}",
        "industry":"Industry of {company}",
        "financials":"Financials of {company}",
        "history":"History of {company}",
        "competitors":"{company} competitors",
        "number_of_employees":"Number of Employees at {company}",
        "number_of_geographies":"Number of Geographies {company} operates in",
        "linkedin_url_and_followers":"LinkedIn URL and followers for {company}",
        "instagram_url_and_followers":"Instagram URL and followers for {company}",
        "tiktok_url_and_followers":"Tiktok URL and followers for {company}",
        "facebook_url_and_followers":"Facebook URL and followers for {company}",
        "twitter_url_and_followers":"Twitter(X) URL and followers for {company}",
        "glassdoor_score":"Glassdoor Score of {company}",
        "online_forums_summary":"Online Forums Summary about {company}",
        "number_of_media_mentions":"Number of Media Mentions of {company}",
        "number_of_awards":"Number of Awards {company} has won",
        "employee_value_proposition":"{company} Employee Value Proposition",
        "culture_and_values":"{company} Culture and Values",
        "purpose":"Purpose of {company}",
        "vision":"Vision of {company}",
        "mission":"Mission of {company}",
}

def extract_snippet_data(relevant_info_from_query, number_of_iteration):
    all_snippet_data = []
    for snippet_data in relevant_info_from_query[:number_of_iteration]:
        all_snippet_data.append(snippet_data.get("snippet"))
    all_snippet_data = " ".join(all_snippet_data)
    print(len(all_snippet_data))
    return all_snippet_data

def get_data_from_bing(company_name, fields_to_query_with_bing):
    # mkt = 'en-US'
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    relevant_info_from_query = []
    all_data_from_chatgpt_2 = {}

    try:
        for field, query_field in fields_to_query_with_bing.items():
            print("bing : ", query_field)
            snippet_data = ""
            query_params = query_field.format(company=company_name)
            params = {
                    'q': query_params,
                    'count': 50,
                    # "offset": i,
                    # "mkt": mkt,
                    # "freshness": "Month"
                }
            response = requests.get(endpoint, headers=headers, params=params)

            response.raise_for_status()

            crawl_data = response.json()

            if query_field in ['LinkedIn URL and followers', "Instagram UR and followers", "Tiktok URL and followers", "Facebook URL and followers", "Twitter(X) URL and followers"]:
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 3)
                url = crawl_data.get("webPages", {}).get("value", [])[0]["url"]
                snippet_data += f" The url is : {url}"
                data_from_chatgpt_2 = get_data_from_chatgpt_2(snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[field] = cleaned_result
            else:
                relevant_info_from_query = crawl_data.get("webPages", {}).get("value", [])
                snippet_data = extract_snippet_data(relevant_info_from_query, 9)
                data_from_chatgpt_2 = get_data_from_chatgpt_2(snippet_data, query_params)
                cleaned_result = re.sub(r'\\', '', data_from_chatgpt_2)
                cleaned_result = re.sub(r'\n', '', cleaned_result)
                cleaned_result = cleaned_result.strip('"')
                all_data_from_chatgpt_2[field] = cleaned_result

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
            headquarters=final_data.get('headquarters', ''),
            established_date=final_data.get('founded_date', ''),
            about_the_company=final_data.get('about_the_company', ''),
            industry=final_data.get('industry', ''),
            company_financials=final_data.get('financials', ''),
            company_history=final_data.get('history', ''),
            top_3_competitors=final_data.get('competitors', ''),
            number_of_employees = final_data.get('number_of_employees', ''),
            number_of_geographies = final_data.get('number_of_geographies', ''),
            linked_info=final_data.get('linkedin_url_and_followers', ''),
            instagram_info=final_data.get('instagram_url_and_followers', ''),
            tiktok_info=final_data.get('tiktok_url_and_followers', ''),
            facebook_info=final_data.get('facebook_url_and_followers', ''),
            twitter_info=final_data.get('twitter_url_and_followers', ''),
            internal_comms_channels=final_data.get('Internal Comms Channels', ''),
            glassdoor_score=final_data.get('glassdoor_score', ''),
            what_retains_talent=final_data.get('What Retains Talent', ''),
            what_attracts_talent=final_data.get('What Attracts Talent', ''),
            employee_value_proposition=final_data.get('employee_value_proposition', ''),
            culture_and_values=final_data.get('culture_and_values', ''),
            purpose=final_data.get('purpose', ''),
            customer_value_proposition=final_data.get('Customer Value Proposition', ''),
            vision=final_data.get('vision', ''),
            mission=final_data.get('mission', ''),
            brand_guidelines=final_data.get('Brand Guidelines', ''),
        )

    company_vector = Company.objects.get(user=user, name=company_name)
    company_vector_serializer = CompanySerializer(company_vector)

    whole_data = {
        "company_vector": company_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)

    return formatted_string
