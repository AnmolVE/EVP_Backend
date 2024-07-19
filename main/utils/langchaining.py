import os
import re
import json
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import (
    Company,
    AttributesOfGreatPlace,
    KeyThemes,
    AudienceWiseMessaging,
    SwotAnalysis,
    Alignment,
    MessagingHierarchyTabs,
    EVPPromise,
    EVPAudit,
)

from ..serializers import (
    AttributesOfGreatPlaceSerializer,
    KeyThemesSerializer,
    AudienceWiseMessagingSerializer,
    SwotAnalysisSerializer,
    AlignmentSerializer,
    MessagingHierarchyTabsSerializer,
    EVPPromiseSerializer,
    EVPAuditSerializer,
)

from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from openai import AzureOpenAI
import chromadb.utils.embedding_functions as embedding_functions

AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_TYPE = os.environ["AZURE_OPENAI_TYPE"]
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]
AZURE_EMBEDDING_MODEL = os.environ["AZURE_EMBEDDING_MODEL"]

chat_client = AzureOpenAI(
    azure_endpoint = AZURE_ENDPOINT, 
    api_key=AZURE_OPENAI_KEY,  
    api_version=AZURE_OPENAI_API_VERSION
)

langchain_query = {
"Headquarters": """
        Search for the primary location of the company's headquarters. Look for terms like 'head office,' 'corporate office,' or 'main office location.' Identify the city, state, and country where the company's headquarters is located.
""",
"Established Date": """
        Find the date or year the company was established. Look for phrases like 'founded in,' 'established in,' or 'incorporated on.' Identify the specific year or date of the company's founding
""",
"About the company": """
        Create a summary which gives the description about the company.
""",
"Industry": """
        Identify the industry or sector the company operates in. Use terms like 'industry sector,' 'business sector,' or 'industry classification.' Specify the primary market or sector the company is associated with.
""",
"Company Financials": """
        Locate financial information about the company. Search for terms like 'financial performance,' 'company revenue,' 'annual report,' or 'financial statement.' Provide data on the company's revenue, profits, and overall financial health.
""",
"Company History": """
                Provide a detailed overview of company's history, focusing on its founding, major milestones, key product developments, and significant shifts in strategy or market presence. Include information on the evolution of its leadership, notable acquisitions or mergers, and any major challenges or controversies it has faced. Summarize how these events have shaped the company's current status in its industry.
""",
"Top 3 Competitors": """
                Find the top three competitors of the company. Use keywords like 'main competitors,' 'industry competitors,' or 'competitive landscape.' Identify and list the top three companies competing with the organization.
""",
"Number of Employees": """
                Search for the total number of employees in the company. Use terms like 'employee count,' 'number of employees,' or 'company workforce.' Provide the latest available employee count.
""",
"Number of Geographies": """
                Identify the number of geographical locations where the company has operations. Search for phrases like 'number of locations,' 'geographical presence,' or 'global footprint.' List the distinct regions or countries where the company is active.
""",
"LinkedIn URL and followers": """
                Search for information on the company's LinkedIn profile. Use keywords like 'LinkedIn profile,' 'LinkedIn company page,' or 'LinkedIn followers.' Provide the URL, follower count, and a summary of the company's activity on LinkedIn.
""",
"Instagram UR and followers": """
                Find information on the company's Instagram profile. Use keywords like 'Instagram profile,' 'Instagram company page,' or 'Instagram followers.' Provide the URL, follower count, and a summary of the company's activity on Instagram.
""",
"Tiktok URL and followers": """
                Search for information on the company's TikTok profile. Use terms like 'TikTok profile,' 'TikTok company page,' or 'TikTok followers.' Provide the URL, follower count, and a summary of the company's activity on TikTok.
""",
"Facebook URL and followers": """
                Locate information on the company's Facebook page. Use keywords like 'Facebook profile,' 'Facebook page,' or 'Facebook followers.' Provide the URL, follower count, and a summary of the company's activity on Facebook.
""",
"Twitter(X) URL and followers": """
                Find information on the company's (X)  Twitter profile. Use keywords like 'Twitter profile,' 'Twitter page,' or 'Twitter followers.' Provide the URL, follower count, and a summary of the company's activity on Twitter.
""",
"Internal Comms Channels": """

""",
"Exit Interview Feedback Summary": """
                Find feedback from exited employees about the company. Use keywords like 'exit feedback,' 'leaving form,' or 'exit interview'. Provide a summary of common sentiments and key points.
""",
"Employee Feedback Summary": """
                Find feedback from current or former employees about the company. Use keywords like 'employee feedback,' 'staff opinions,' or 'employee reviews' or 'HR complaints'.  Provide a summary of common sentiments and key points from employee feedback.
""",
"Engagement Survey Result Summary": """
                Search for results from employee engagement surveys. Look for terms like 'engagement survey results,' 'employee satisfaction survey,' or 'engagement metrics.' or 'ESat survey'. Summarize the key findings and overall engagement levels of the employees.
""",
"Glassdorr Score": """
                Locate the company's Glassdoor score. Use terms like 'Glassdoor score,' 'Glassdoor rating,' or 'employee reviews score.' Provide the current rating and a summary of key feedback from reviews.
""",
"Online Forums Summary": """
                Find online mentions of the company name and summarize the feedback about the company as an employer. Use keywords like 'feedback,' 'opinions,' 'reviews,' and 'perception.' Provide a summary of common sentiments and key points from various sources. Focus on recurring themes, both positive and negative, and highlight any specific comments that illustrate the overall perception of the company as an employer.
""",
"Average Tenure of Employees": """
                Locate the average tenure of employees in the company. Use terms like 'average tenure,' 'employee tenure,' or 'tenure length.' Calculate the average tenure of the employees.
""",
"Net Promoter Score": """
                Locate phrases like 'NPS' or "Net Promoter Score" in documents.
""",
"Number of Early Exits": """
                Locate the number of early exits, specifically employees who left within 90 days of being hired. Use keywords like 'early exits,' 'exits within 90 days,' or 'new employee turnover.' Provide the count of early exits.
""",
"Number of Rehires": """
                Locate the number of REHIRES or the number of people who left the company and rejoined the company. Use keywords like 'Boomerangs' 'returning employees' 'second life' or 'comeback'
""",
"What Retains Talent": """
                Identify the top three reasons why talent tends to stay with the company. Search the document for key phrases such as 'What Retains Talent,' 'Why Talent Chooses Us,' or similar terms. Summarize the common themes and specific factors that contribute to employee retention, highlighting any notable policies, benefits, or cultural aspects that are frequently mentioned.
""",
"Number of Referrals": """
                Find the total number of employee referrals received by the company. Use keywords like 'employee referrals,' 'referral count,' or 'referrals received.' Provide the count of referrals.
""",
"Number of Referrals to Hires": """
                Locate the number of hires made through employee referrals. Use keywords like 'referral hires,' 'hires through referrals,' or 'referral-to-hire ratio.' Provide the count of referral-based hires.
""",
"ESAT Recommendability Score": """
                Find the Employee Satisfaction (ESAT) recommendability score for the company. Use terms like 'ESAT score,' 'employee satisfaction score,' or 'recommendability score.' Provide the ESAT score and a brief explanation
""",
"Number of Jobs Posted": """
                Find the total number of job postings listed by the company. Use terms like 'job postings,' 'job listings,' or 'open positions.' Provide the count of current job postings.
""",
"Average Number of Job Post Clicks": """
                Search for the average number of clicks job postings received for the Company. Look for metrics such as 'job post clicks,' 'average job clicks,' and 'click-through rate (CTR) for job postings.' Summarize the average number of clicks and what this indicates about candidate interest and engagement.
""",
"Number of Direct Hires": """
                Find the number of direct hires by the company. Use terms like 'direct hires,' 'internal hires,' or 'company hires.' Provide the count of employees hired directly by the company.
""",
"Average Time to Fill": """
                Locate the average time-to-fill for job positions in the company. Use keywords like 'time-to-fill,' 'hiring time,' or 'recruitment duration.' Provide the average duration from job posting to hiring.
""",
"Number of Offers Made": """
                Find the total number of job offers made by the company. Use terms like 'number of offers,' 'job offers made,' or 'offers extended.' Provide the count of job offers given by the company.
""",
"Number of Offers Accepted": """
                Locate the number of job offers accepted by candidates. Use terms like 'offers accepted,' 'job offers taken,' or 'accepted offers.' Provide the count of accepted job offers.
""",
"Number of Direct Applications": """
                Search for the total number of direct job applications received by the company. Use keywords like 'direct applications,' 'job applications received,' or 'applications count.' Provide the count of direct applications.
""",
"Number of Hires": """
                Search for the total number of new hires in the company. Use terms like 'number of hires,' 'new employees,' or 'hires count.' Provide the count of new hires.
""",
"What Attracts Talent": """
                Identify the key factors that attract talent to the company. Use keywords like 'talent attraction,' 'reasons to join,' or 'employment appeal.' Summarize the main attributes that make the company appealing to potential employees.
""",
"Number of Career Page Subscribers": """
                Locate information on the number of subscribers to the company's careers page. Search for terms like 'careers page subscribers,' 'job alert subscribers,' or 'careers newsletter.' Provide the latest subscriber count.
""",
"Number of Views": """
                Find information on the number of views or impressions for company's social media posts. Use terms like 'social media views,' 'social media impressions,' and 'post views.' Provide a summary of the total views and how it reflects the reach and visibility of their social media content.
""",
"Engagement": """
                Search for data on social media engagement for the Company. Use keywords such as 'social media engagement,' 'post engagement,' 'likes, comments, shares,' and 'audience interaction.' Summarize the main forms of engagement and their impact on the company's social media presence.
""",
"Number of Media Mentions": """
                Locate the number of media mentions for the company. Use keywords like 'media mentions,' 'press coverage,' or 'news mentions.' Provide the count of media mentions and highlight significant coverage.
""",
"Number of Awards": """
                Find the number of awards the company has won. Use terms like 'awards won,' 'number of awards,' or 'accolades received.' Provide the count of awards and notable awards won.
""",
"Summary of Awards / Recognition": """
                Extract information on the company's rewards and recognition programs. Include details on performance awards, incentive programs, and employee feedback on these programs.
""",
"Employee Value Proposition": """
                Find the Employee Value Proposition (EVP):
""",
"Culture and Values": """
                Search for information on the company's culture and values. Use terms like 'company culture,' 'organizational values,' or 'core values.' Summarize the key aspects of the workplace culture and the guiding principles of the company.
""",
"Purpose": """
                Locate the purpose of the company. Use keywords like 'company purpose,' 'organizational purpose,' or 'mission.' Provide a summary of why the company exists and its core guiding principles.
""",
"Customer Value Proposition": """
                Locate information on the Company's customer value proposition. Search for terms like 'customer value proposition,' 'value proposition statement,' 'unique selling proposition,' and 'competitive advantage.' Summarize the key benefits they offer to customers and how they differentiate from competitors.
""",
"Vision": """
                Find the company's vision statement. Search for terms like 'company vision,' 'future aspirations,' or 'long-term goals.' Summarize the company's desired future state and long-term aspirations.
""",
"Mission": """
                Locate the company's mission statement. Use keywords like 'company mission,' 'mission statement,' or 'core objectives.' Provide a summary of the company’s main goals and approach to achieving its purpose.
""",
"Internal Comms Samples": """
                Locate samples of internal communications for the Company. Use search terms like 'internal communications examples,' 'employee communications,' 'internal newsletters,' and 'company intranet communications.' Summarize the key methods and types of internal communications used.
""",
"External Comms Samples": """
                Find examples of external communications for the Company. Search using terms such as 'external communications examples,' 'public relations samples,' 'press releases,' and 'public announcements.' Summarize the main types and characteristics of their external communications.
""",
"Brand Guidelines": """
                Locate the company's brand guidelines. Search for terms like 'brand guidelines,' 'branding rules,' or 'brand standards.' Fetch Logo, Brand Color, Imagery/Photography, Brand Personality, Tone, Voice, Icons or Iconography, FONT, Typerface, Icons, Iconography.
"""
}

# def create_embeddings():
#     embeddings = AzureOpenAIEmbeddings(
#                     openai_api_key = AZURE_OPENAI_KEY,
#                     azure_endpoint = AZURE_ENDPOINT,
#                     openai_api_version = AZURE_OPENAI_API_VERSION,
#                     openai_api_type = AZURE_OPENAI_TYPE,
#                     azure_deployment = AZURE_EMBEDDING_DEPLOYMENT,
#                     model = AZURE_EMBEDDING_MODEL
#                 )
#     return embeddings

def create_embeddings():
    embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=AZURE_OPENAI_KEY,
            api_base=AZURE_ENDPOINT,
            api_type=AZURE_OPENAI_TYPE,
            api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_EMBEDDING_MODEL,
        )
    return embeddings

def query_with_langchain(company_name):
    loader = PyPDFLoader(r"media\final_pdf\merged_pdf.pdf")
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
    print("Hello")
    documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
    ids=[f"id{i}" for i in range(len(documents))]

    embeddings = create_embeddings()

    embedded_documents = embeddings([documents[i] for i in range(len(documents))])

    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    client = chromadb.PersistentClient(path=f"vector_databases/{sanitized_company_name}")
    collection = client.get_or_create_collection(
        name="test",
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        embeddings=embedded_documents,
        documents=documents,
        ids=ids,
    )

    json_data = {}
    for key, query in langchain_query.items():
        print(key)

        query_results = collection.query(
                query_texts=[query],
                n_results=20,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        Information: {fetched_documents} \n \n Question: {query}.
        """
        print(prompt)

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant. Your users are asking questions about information contained in the given data.
                                You will be shown the user's question, and the relevant information from the data.
                                After analyzing the complete information, your task is to answer the user's question using only this information.
                                IF YOU DON'T FIND THE ANSWER IN THE GIVEN INFORMATION PLEASE SAY -- "Not found".
                            """
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000,
        )
        chat_response = completion.choices[0].message.content
        json_data[key] = chat_response
    return json_data

    # def process_query(query):
    #     result = qa({"query": query})
    #     cleaned_result = re.sub(r'\\', '', result["result"])
    #     cleaned_result = re.sub(r'\n', '', cleaned_result)
    #     cleaned_result = cleaned_result.strip('"')
    #     return query, cleaned_result
    # json_data = {}
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     future_to_query = {executor.submit(process_query, query): query for query in langchain_query}

    #     for future in as_completed(future_to_query):
    #         query, cleaned_result = future.result()
    #         json_data[query] = cleaned_result
    
    # return json_data


def save_pgData_to_vector_database(file_path, company_name):
    loader = TextLoader(file_path)
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
    documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
    ids=[f"id{i}" for i in range(len(documents))]

    embeddings = create_embeddings()

    embedded_documents = embeddings([documents[i] for i in range(len(documents))])

    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    client = chromadb.PersistentClient(path=persistent_directory)
    if os.path.exists(os.path.join(persistent_directory)):
        collection = client.get_or_create_collection(
            name="test",
            embedding_function=embeddings,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            embeddings=embedded_documents,
            documents=documents,
            ids=ids,
        )

        return "Data added successfully in the vector database"
    return "Some error occured"

attributes_of_great_place_query = {
"Culture":"""
        Provide a detailed description of the company's culture. Focus on the intangible aspects such as the level of transparency, hierarchy, and formality. Describe the overall atmosphere and how employees feel about their workplace. How do employees interact with each other and with management? Are relationships more formal or casual? Do employees feel empowered and valued? Provide specific examples or anecdotes that illustrate these cultural traits, capturing the subtle, unspoken elements that define the company's environment.
""",
"Purpose and Values":"""
        Describe the company's core purpose and values. How are these values communicated to employees and integrated into daily operations? Provide examples of initiatives or programs that reflect the company's commitment to its purpose and values.
""",
"Benefits and Perks":"""
        Identify and describe the benefits and perks offered by the company to its employees. How do these benefits and perks compare to industry standards? Include details on health insurance, retirement plans, wellness programs, flexible work arrangements, and any unique perks that differentiate the company.
""",
"Career Development":"""
        Examine the opportunities for career development within the company. How does the organization support employee growth and professional development? Discuss available training programs, mentorship opportunities, promotion policies, and any other initiatives aimed at fostering career advancement.
""",
"Office and Facilities":"""
        Provide insights into the company's office environment and facilities. Describe the physical workspace, including the layout, amenities, and any special features that contribute to the work environment. How do the office and facilities support employee productivity and well-being?
""",
"Leadership and Management":"""
        Analyze the leadership and management style within the company. How do leaders interact with employees and make decisions? Discuss the level of transparency, approachability, and support provided by the management team.
""",
"Rewards and Recognition":"""
        Describe the company's approach to rewards and recognition. How are employees recognized for their contributions and achievements? Provide examples of formal and informal recognition programs and their impact on employee morale.
""",
"Teamwork and Collaboration":"""
         Evaluate the level of teamwork and collaboration within the company. How do employees work together across different departments and teams? Discuss any tools, processes, or cultural aspects that facilitate or hinder collaboration
""",
"Brand and Reputation":"""
        Assess the company's brand and reputation, both internally and externally. How do employees perceive the company's brand? What is the public and industry perception of the company? Include any relevant awards, recognitions, or public relations efforts.
""",
"Work life balance":"""
        Provide insights into how the company supports work-life balance for its employees. Discuss policies and practices such as flexible working hours, remote work options, and leave policies. How do employees feel about their ability to balance work and personal life?
"""
}

key_themes_query = {
"top_key_themes":"""
        What are some key themes related to building the company's employee value proposition? Focus on themes that help the company stand out as an attractive employer. What are unique aspects about the company that can help to attract and retain the best talent?
"""
}

audience_wise_messaging_query = {
"Existing Employees":"""
    What current employees working at the client company are saying about the company.
    Create a summary from the data and then give me the response.
""",
"Alumni":"""
    What ex-employees who used to work at the client company are saying about the company.
    Create a small summary.
""",
"Targeted Talent":"""
    What candidates and people who are not working at client company are saying about the client. Also include what these people are saying about what they look for in a desired employer.
    Create a small summary. 
""",
"Leadership":"""
    What vice president and above who work at client company are saying about the client company.
    Create a small summary.
""",
"Recruiters":"""
    What recruiters who currently hire talent for client are saying everybody thinks about the client.
    Create a small summary.
""",
"Clients":"""
    What clients of the client company are saying about the client company.
    Create a small summary.
""",
"Offer Drops":"""
    What people who did not accept client's offer are saying about client company.
    Create a small summary and don't give generalized results.
"""
}

def get_develop_data_from_vector_database(company_name, user):
    embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=AZURE_OPENAI_KEY,
            api_base=AZURE_ENDPOINT,
            api_type=AZURE_OPENAI_TYPE,
            api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_EMBEDDING_MODEL,
        )
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        develop_collection = chroma_client.get_collection(
            name="test",
            embedding_function=embeddings,
        )
    print("In develop")

    json_data = {}
    for key, query in attributes_of_great_place_query.items():
        print(key)

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        Information: {fetched_documents} \n \n Question: {query}.
        """

        print(prompt)
        print("*************************************************************************************************************")

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant. Your users are asking questions about information contained in the given data.
                                You will be shown the user's question, and the relevant information from the data.
                                After analyzing the complete information, your task is to answer the user's question using only this information.
                                IF YOU DON'T FIND THE ANSWER IN THE GIVEN INFORMATION PLEASE SAY -- "Not found".
                            """
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        json_data[key] = chat_response

    for key, query in key_themes_query.items():
        print(key)

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        Information: {fetched_documents} \n \n Question: {query}.
        """

        print(prompt)
        print("*************************************************************************************************************")

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant. Your users are asking questions about information contained in the given data.
                                You will be shown the user's question, and the relevant information from the data.
                                After analyzing the complete information, your task is to answer the user's question using only this information.
                                IF YOU DON'T FIND THE ANSWER IN THE GIVEN INFORMATION PLEASE SAY -- "Not found".
                            """
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        json_data[key] = chat_response

    for key, query in audience_wise_messaging_query.items():
        print(key)

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        Information: {fetched_documents} \n \n Question: {query}.
        """

        print(prompt)
        print("*************************************************************************************************************")

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant. Your users are asking questions about information contained in the given data.
                                You will be shown the user's question, and the relevant information from the data.
                                After analyzing the complete information, your task is to answer the user's question using only this information.
                                IF YOU DON'T FIND THE ANSWER IN THE GIVEN INFORMATION PLEASE SAY -- "Not found".
                            """
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        json_data[key] = chat_response

    # return json_data

    company = Company.objects.get(user=user, name=company_name)

    attributes_of_great_workplace = AttributesOfGreatPlace.objects.create(
        user=user,
        company=company,
        culture = json_data.get("Culture", ""),
        purpose_and_values = json_data.get("Purpose and Values", ""),
        benefits_perks = json_data.get("Benefits and Perks", ""),
        career_development = json_data.get("Career Development", ""),
        office_and_facilities = json_data.get("Office and Facilities", ""),
        leadership_and_management = json_data.get("Leadership and Management", ""),
        rewards_and_recognition = json_data.get("Rewards and Recognition", ""),
        teamwork_and_collaboration = json_data.get("Teamwork and Collaboration", ""),
        brand_and_reputation = json_data.get("Brand and Reputation", ""),
        work_life_balance = json_data.get("Work life balance", ""),
    )

    key_themes = KeyThemes.objects.create(
        user=user,
        company=company,
        top_key_themes = json_data.get("top_key_themes", ""),
    )

    audience_wise_messaging = AudienceWiseMessaging.objects.create(
        user=user,
        company=company,
        existing_employees = json_data.get("Existing Employees", ""),
        alumni = json_data.get("Alumni", ""),
        targeted_talent = json_data.get("Targeted Talent", ""),
        leadership = json_data.get("Leadership", ""),
        recruiters = json_data.get("Recruiters", ""),
        clients = json_data.get("Clients", ""),
        offer_drops = json_data.get("Offer Drops", ""),
    )


swot_analysis_query = {
"what_is_working_well_for_the_organization": """Identify the attributes that highlight what is working well for the organization. Focus on aspects that employees and external reviewers consistently praise or express satisfaction with. Provide detailed insights on these positive aspects and how they contribute to the overall success and positive reputation of the organization.
        Create multiple headings and give the description of those headings and summarize them.
""",
"what_is_not_working_well_for_the_organization": """Analyze the provided data to identify the top insights/themes that highlight what is not working well for the organization. Focus on aspects that employees and external reviewers consistently criticize or express concerns about. Provide detailed insights on these negative aspects and how they impact employee satisfaction and the overall performance of the organization.
        Create multiple headings and give the description after summarizing them as I only want summary.
"""
}

def get_dissect_data_from_vector_database(company_name, user):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    attributes_of_great_place_vector = AttributesOfGreatPlace.objects.get(user=user, company=company_id)
    attributes_of_great_place_vector_serializer = AttributesOfGreatPlaceSerializer(attributes_of_great_place_vector)

    key_themes_vector = KeyThemes.objects.get(user=user, company=company_id)
    key_themes_vector_serializer = KeyThemesSerializer(key_themes_vector)

    audience_wise_messaging_vector = AudienceWiseMessaging.objects.get(user=user, company=company_id)
    audience_wise_messaging_vector_serializer = AudienceWiseMessagingSerializer(audience_wise_messaging_vector)

    whole_data = {
        "attributes_of_great_place_vector": attributes_of_great_place_vector_serializer.data,
        "key_themes_vector": key_themes_vector_serializer.data,
        "audience_wise_messaging_vector": audience_wise_messaging_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)
    print(len(formatted_string))
    
    json_data = {}
    for key, query in swot_analysis_query.items():
        prompt = f"""
                I want to fetch the information from the given data.
                First analyze the complete data below
                data : {formatted_string}

                Now fetch the below information from the data and give me 5 points on this.
                {query} and returns the response. 

                **Note :** Fetch the complete information from the given data only and don't include anything extra.
                Don't add anything which you don't find in the given data just give the information which is available in the given data.
        """
        message_text = [
            {"role": "system", "content": f"You are an expert in fetching information from the given data and you are only allowed to fetch information from the given data."},
            {"role": "user", "content": prompt}
        ]

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = message_text,
        temperature=0.7,
        max_tokens=4000,
        )
        response = completion.choices[0].message.content
        json_data[key] = response
    # return json_data
    company = Company.objects.get(user=user, name=company_name)
    
    swot_analysis = SwotAnalysis.objects.create(
        user=user,
        company=company,
        what_is_working_well_for_the_organization = json_data.get("what_is_working_well_for_the_organization", ""),
        what_is_not_working_well_for_the_organization = json_data.get("what_is_not_working_well_for_the_organization", ""),
    )

    embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=AZURE_OPENAI_KEY,
            api_base=AZURE_ENDPOINT,
            api_type=AZURE_OPENAI_TYPE,
            api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_EMBEDDING_MODEL,
        )
    
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        design_collection = chroma_client.get_collection(
            name="test_design",
            embedding_function=embeddings,
        )

        query_results = design_collection.query(
                query_texts=["Create a summary of the whole data within 300 words"],
                n_results=5,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        Use and Analyze the following pieces of data to fetch the information from it

        Context: <{fetched_documents}>

        Create a summary of the whole context data within 300 words. 
        """

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
                {
                    "role":"system",
                    "content":"""You are an expert in fetching information from the given unstructured data.
                              Instructions:
                              - Only answer questions related to the user's query
                              - If you're unsure of an answer, you can say "I don't know".
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
        chat_response = completion.choices[0].message.content
        dataset_2 = chat_response
        print(dataset_2)
        
        prompt2 = f"""First analyze both datasets below:

            Review the primary research available in entire Dataset 1 (both sections 'whats working well' and whats not working well).
            **Dataset 1** : {formatted_string}

            **Dataset 2**: {dataset_2}

            Now using the information available in Dataset 1, create a summary for each point mentioned in 'what we want to be known for' section in the Dataset 2.
            Give me all positive and negative aspects about every point after analyzing whole data.

            The summary for each point should be divided into two sub-sections. 

            1) Positive Aspects - what is working well for the organization. Focus on aspects that employees and external reviewers consistently compliment or express happiness and positivity about. Provide detailed insights on these positive aspects and how they improve employee satisfaction and the overall performance of the organization.

            2) Negative Aspects - what is not working well for the organization. Focus on aspects that employees and external reviewers consistently criticize or express concerns about. Provide detailed insights on these negative aspects and how they impact employee satisfaction and the overall performance of the organization.

            YOU ARE ONLY ALLOWED TO EXTRACT INFORMATION FROM THE DATA AVAILABLE IN DATASET 1 FOR EVERY POINT.
            IF THE INFORMATION IS NOT AVAILABLE PLEASE SAY - "The information is not available".

            Don't only look for exact word or phrase match - you should be intelligent enough to co-relate different points.
            Focus on the essence of what is being said and not specific key words.

            Arrange all points in numbers.
        """
        message_text = [
            {"role": "system", "content": f"You are an expert in fetching information from the given context and and you cannot look outside of the given data."},
            {"role": "user", "content": prompt2}
        ]

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = message_text,
        temperature=0.1,
        max_tokens=4000,
        )
        response2 = completion.choices[0].message.content
        json_data["what we want to be known for"] = response2

        alignment = Alignment.objects.create(
            user=user,
            company=company,
            what_we_want_to_be_known_for = json_data.get("what we want to be known for", "")
        )
        return json_data
        

def get_design_data_from_database(company_name, user):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    analysis_vector = SwotAnalysis.objects.get(user=user, company=company_id)
    analysis_vector_serializer = SwotAnalysisSerializer(analysis_vector)

    alignment_vector = Alignment.objects.get(user=user, company=company_id)
    alignment_vector_serializer = AlignmentSerializer(alignment_vector)

    whole_data = {
        "analysis_vector": analysis_vector_serializer.data,
        "alignment_vector": alignment_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)

    query1 = """Identify 4 themes that are unique about the company and will help it stand out as an employer. Focus on themes that are different from standard good HR practices. These themes should be believable about the company but also have an element of aspiration, which means that these could be things the company aspires towards and may not have completely achieve yet.
                Rank these themes from most relevant to least relevant and don't include numbers or anything just headings and description.
            """

    prompt = f"""
        Information: {formatted_string}.

        Analyze the complete information above.
        After analyzing it, give the response in json format.

        Question: {query1}

        Make sure to format the response in the json format where headings as the keys and description as the value of the response.
        """

    completion = chat_client.chat.completions.create(
    model="STIMULAIGPT4O",
    response_format={ "type": "json_object" },
    messages = [
        {
            "role":"system",
            "content":"""You are a helpful expert research assistant designed to output JSON. You will be shown the given information.
                            After analyzing the complete information, your task is to answer the question using only given information.
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
    chat_response = completion.choices[0].message.content
    try:
        json_response = json.loads(chat_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    for key, value in json_response.items():
        MessagingHierarchyTabs.objects.create(
            user=user,
            company = company,
            tab_name = key,
            tabs_data = value
        )
    messaging_hierarchy_tabs = MessagingHierarchyTabs.objects.filter(user=user, company=company_id)
    serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs, many=True)
    return serializer.data

def get_tagline(company_name, user, main_theme, pillar_1, pillar_2, pillar_3, combined_tabs_data):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    if pillar_1 and pillar_2 and pillar_3:
        query = f"""
                    Act like an advertising expert. Now create a narrative. A narrative is a combination of a Tagline and advertising body copy. The logic for the narrative is that the {main_theme} will become the main theme of that narrative. The remaining {pillar_1} , {pillar_2} and {pillar_3} become secondary or supporting pillars. 
                    This is how the advertising copy of the narrative will flow. Don't write the subheads below, but follow the instructions.
                    Start with a hook or an engaging statement that captures the reader's attention.
                    Provide a clear and concise explanation of the main theme.
                    Now link the main theme to the supporting themes.
                    Use language that resonates emotionally with employees and potential employees, creating a connection.
                    Emphasize what sets the company apart from competitors.
                    End with a strong call to action, encouraging the audience to take the next step, such as joining the company.
                    Don't include the name of the company in the taglines. But consider the industry of the company.
                """
    else:
        query = f"""
                    Act like an advertising expert. Now create a narrative. A narrative is a combination of a Tagline and advertising body copy. The logic for the narrative is that the {main_theme} will become the main theme of that narrative. 
                    This is how the advertising copy of the narrative will flow. Don't write the subheads below, but follow the instructions.
                    Start with a hook or an engaging statement that captures the reader's attention.
                    Provide a clear and concise explanation of the main theme.
                    Now link the main theme to the supporting themes.
                    Use language that resonates emotionally with employees and potential employees, creating a connection.
                    Emphasize what sets the company apart from competitors.
                    End with a strong call to action, encouraging the audience to take the next step, such as joining the company.
                    Don't include the name of the company in the taglines. But consider the industry of the company.
                """
    
    prompt = f"""
        Information: {combined_tabs_data}

        Analyze the complete information above.
        After analyzing it, give the response regarding below query.
         
         Query: {query}.
        """

    completion = chat_client.chat.completions.create(
    model="STIMULAIGPT4O",
    messages = [
        {
            "role":"system",
            "content":"""You are a helpful expert research assistant. You will be shown the given information.
                            After analyzing the complete information, your task is to answer the question using only given information.
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
    chat_response = completion.choices[0].message.content
    return chat_response

def get_creative_direction_from_chatgpt(brand_guidelines, tagline):
    
    prompt = f"""
                First analyze the brand guidelines and tagline given below:

                Brand Guidelines : {brand_guidelines}

                Tagline : {tagline}

                Now suggest a single visual that captures the tagline and advertising body copy from the messaging hierarchy section while adhering to the company's brand guidelines in terms of color, style, tone etc as well as the industry of the company. Focus on creating one visual that focuses mainly on the 'overarching theme' and very subtly incorporates the secondary pillars.
             """

    completion = chat_client.chat.completions.create(
    model="STIMULAIGPT4O",
    messages = [
        {
            "role":"system",
            "content":"""You are an expert advertising creative art director.
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
    chat_response = completion.choices[0].message.content
    return chat_response

def get_evp_promise_from_chatgpt(company_name, user, themes_data):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    RESPONSE_JSON = {
        "Theme": {
            "What makes this theme distinctive": "A summary of 2 lines explaining what makes this theme stand out.",
            "What employees can expect": "Create 3 points describing what employees can expect in relation to this theme.",
            "What is expected of employees": "Create 3 points outlining what is expected of employees in relation to this theme."
        }
    }

    prompt = f"""
                First analyze the given Themes Data below and return the data in json format:

                Themes Data : {themes_data}

                For each of the four themes, provide a detailed messaging overview that includes response for the below query:

                RESPONSE_JSON : {RESPONSE_JSON}

                The json format will contain keys same as the RESPONSE_JSON and value as the response to the query and replace Theme with the actual theme name.
                Don't include list in the response and don't include numbers or anything, I just want keys and the description.

                Make sure to format your response like RESPONSE_JSON and use it as a guide.
             """

    completion = chat_client.chat.completions.create(
    model="STIMULAIGPT4O",
    response_format={ "type": "json_object" },
    messages = [
        {
            "role":"system",
            "content":"""You are an expert advertising creative art director.
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
    chat_response = completion.choices[0].message.content
    try:
        json_response = json.loads(chat_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    for key,value in json_response.items():
        theme = key
        what_makes_this_theme_distinctive = value["What makes this theme distinctive"]
        what_employees_can_expect = value["What employees can expect"]
        what_is_expected_of_employees = value["What is expected of employees"]

        EVPPromise.objects.create(
            user=user,
            company = company,
            theme = theme,
            what_makes_this_theme_distinctive = what_makes_this_theme_distinctive,
            what_employees_can_expect = what_employees_can_expect,
            what_is_expected_of_employees = what_is_expected_of_employees,
        )

    evp_promise = EVPPromise.objects.filter(user=user, company=company_id)
    serializer = EVPPromiseSerializer(evp_promise, many=True)
    return serializer.data

def get_evp_audit_from_chatgpt(company_name, user, analysis_data, alignment_data, four_themes):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    RESPONSE_JSON = {
        "Theme": {
            "What makes this credible": "Evaluate  what aspects of this theme are believable about the company. Look for elements that are true today and being experienced by employees. ",
            "Where do we need to stretch": "Evaluate  what aspects of this theme are not yet fully believable and can be considered 'aspirational' by the company. Look for elements that are not necessarily 100% true today or being fully experienced by employees but are elements that the company would like to aspire towards.",
        }
    }

    prompt = f"""
                First analyze the given datasets below and return the data in json format:
                Analyze the Analysis Data.

                Analysis Data : {analysis_data}

                Now analyze the Alignment Data.

                Alignment Data : {alignment_data}

                Four Themes Available: {four_themes}

                For each of the four themes, provide a detailed messaging overview that includes response for the below query:

                RESPONSE_JSON : {RESPONSE_JSON}

                The json format will contain keys same as the RESPONSE_JSON and value as the response to the query and replace Theme with the actual theme name.
                Don't include list in the response and don't include numbers or anything, I just want keys and the description.

                Make sure to format your response like RESPONSE_JSON and use it as a guide.
             """
    
    completion = chat_client.chat.completions.create(
    model="STIMULAIGPT4O",
    response_format={ "type": "json_object" },
    messages = [
        {
            "role":"system",
            "content":"""You are an expert advertising creative art director.
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
    chat_response = completion.choices[0].message.content
    try:
        json_response = json.loads(chat_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    for key,value in json_response.items():
        theme = key
        what_makes_this_credible = value["What makes this credible"]
        where_do_we_need_to_stretch = value["Where do we need to stretch"]

        EVPAudit.objects.create(
            user=user,
            company = company,
            theme = theme,
            what_makes_this_credible = what_makes_this_credible,
            where_do_we_need_to_stretch = where_do_we_need_to_stretch,
        )

    evp_audit = EVPAudit.objects.filter(user=user, company=company_id)
    serializer = EVPAuditSerializer(evp_audit, many=True)
    return serializer.data

import chromadb

def testing_data(collection):

    json_data = {}
    for key, query in langchain_query.items():
        print(key)

        query_results = collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""Analyze the unstructured Dataset below:
        Dataset = <{fetched_documents}>

        Now fetch the complete information regarding below query using the Dataset only and if you don't find the information please say -- "Not Found".

        Query: {query}

        YOU ARE ONLY ALLOWED TO EXTRACT INFORMATION FROM THE DATA AVAILABLE IN Dataset.
        """

        print(prompt)
        print("*************************************************************************************************************")

        chat_client = AzureOpenAI(
            azure_endpoint = AZURE_ENDPOINT, 
            api_key=AZURE_OPENAI_KEY,  
            api_version=AZURE_OPENAI_API_VERSION
        )

        completion = chat_client.chat.completions.create(
        model="STIMULAIGPT4O",
        messages = [
                {
                    "role":"system",
                    "content":"""You are an expert in fetching information from the given unstructured data.
                              Instructions:
                              - Only answer questions related to the user's query
                              - If you're unsure of an answer, you can say "I don't know".
                              """
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ],
        temperature=0,
        max_tokens=800,
        )
        chat_response = completion.choices[0].message.content
        json_data[key] = chat_response
    return json_data
