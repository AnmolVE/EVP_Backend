# **Important point** : If you don't find the information for any key from the text then just leave it blank string don't add any text just leave it blank.

import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI

AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

chat_client = AzureOpenAI(
    azure_endpoint = AZURE_ENDPOINT, 
    api_key=AZURE_OPENAI_KEY,  
    api_version=AZURE_OPENAI_API_VERSION
)

chatgpt_1_query_data = {
    "Headquarters": "",
    "Established Date": "",
    "About the company": "",
    "Industry": "",
    "Company History": "",
}

def get_data_from_chatgpt_1(company_name, fields_to_query_with_chatgpt_1):
    prompt = f"""
    I want to fetch the information about the company named {company_name}
    So, give me the complete information mentioned in the json_format below format about the company.

    RESPONSE_JSON : {fields_to_query_with_chatgpt_1}

    **Note**: Make sure to format your response exactly like RESPONSE_JSON and use it as a guide.
    """

    message_text = [
        {"role": "system", "content": f"You are an expert in gathering information from multiple sources."},
        {"role": "user", "content": prompt}
    ]

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    response_format={ "type": "json_object" },
    messages = message_text,
    temperature=0.3,
    max_tokens=4000,
    )
    response_from_chatgpt_1 = completion.choices[0].message.content
    try:
        json_response = json.loads(response_from_chatgpt_1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}
    print("data : ", json_response)
    return json_response

query_for_chatgpt_2 = {
"headquarters": """
    Search for the primary location of the company's headquarters.
    The response should appear as city name, country name.
""",
"established_date": """
    Find the date or year the company was found or established.
    The response should only be appear in yyyy format.
""",
"about_the_company": """
    Create a summary which gives the description about the company.
    Focus on how the company describes itself.
""",
"industry": """
    Identify the industry or sector the company operates in.
    Specify the primary market or sector the company is associated with.
""",
"company_financials": """
    Locate financial information about the company.
""",
"company_history": """
    Provide a detailed overview of company's history, focusing on its founding, major milestones, key product developments, and significant shifts in strategy or market presence.
""",
"top_3_competitors": """
    Find the top three competitors of the company. Use keywords like 'main competitors,' 'industry competitors,' or 'competitive landscape.'
    Identify and list the top three companies competing with the organization.
""",
"number_of_employees": """
    Search for the total number of employees in the company.
    Response should only contain the number of employees. No other words or statements. Numeric response only.
""",
"number_of_geographies": """
    Identify the number of geographical locations where the company has operations.
    Response should only list the geographies and not have any additional words.
""",
"linked_info": """
    Search for information on the company's LinkedIn profile.
    Use keywords like 'LinkedIn profile,' 'LinkedIn company page,' or 'LinkedIn followers.' Provide the URL, follower count, and a summary of the company's activity on LinkedIn. Include the frequency of posts, the average number of likes per post, and the average number of comments per post.
""",
"instagram_info": """
    Find information on the company's Instagram profile.
    Use keywords like 'Instagram profile,' 'Instagram company page,' or 'Instagram followers.' Provide the URL, follower count, and a summary of the company's activity on Instagram.
""",
"facebook_info": """
    Locate information on the company's Facebook page.
    Use keywords like 'Facebook profile,' 'Facebook page,' or 'Facebook followers.' Provide the URL, follower count, and a summary of the company's activity on Facebook.
""",
"twitter_info": """
    Find information on the company's (X)  Twitter profile.
    Use keywords like 'Twitter profile,' 'Twitter page,' or 'Twitter followers.' Provide the URL, follower count, and a summary of the company's activity on Twitter.
""",
"glassdoor_score": """
    Locate the company's Glassdoor score. Use terms like 'Glassdoor score,' 'Glassdoor rating,' or 'employee reviews score.' Provide the current rating and a summary of  how many reviews are being considered.
    Do not summarize the actual reviews.
""",
"employee_value_proposition": """
    Find if there is an existing Employee Value Proposition (EVP) and paste the actual statement here.
""",
"culture_and_values": """
    Locate the culture and or values of the company.
""",
"customer_value_proposition": """
    Locate the tagline / CVP of the company.
""",
"purpose": """
    Locate the purpose of the company.
""",
"vision": """
    Locate the vision statement of the company.
""",
"mission": """
    Locate the mission statement of the company.
""",
"brand_guidelines": """
    Locate the colors, imagery guidelines, logo guidelines.
""",
}

def get_data_from_chatgpt_2(snippet_data, field):

    query = query_for_chatgpt_2.get(field)

    prompt = f"""
        First analyze the given information below:

        Given Information: {snippet_data}

        After completely analyzing the given information, fetch the result for the below query from the given information only.
        
        Query: {query}.

        And don't create any heading just give the response as a paragraph.
        """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    messages = [
        {
            "role":"system",
            "content":"""You are a helpful expert research Analyst.
                        """
        },
        {
            "role":"user",
            "content":prompt
        }
    ],
    temperature=0.3,
    max_tokens=4000,
    )
    response_from_chatgpt_2 = completion.choices[0].message.content
    print("data : ", response_from_chatgpt_2)
    return response_from_chatgpt_2
