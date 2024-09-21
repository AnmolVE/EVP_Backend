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

def get_data_from_chatgpt_2(snippet_data, query_params):

    prompt = f"""
        First analyze the given information below:

        Given Information: {snippet_data}

        After completely analyzing the given information, fetch the result for the below query from the given information only.
        
        Query: {query_params}.

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
