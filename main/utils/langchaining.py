import os
import re
import json
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import (
    Company,
    TalentDataset,
    AttributesOfGreatPlace,
    KeyThemes,
    AudienceWiseMessaging,
    SwotAnalysis,
    Alignment,
    MessagingHierarchyTabs,
    CreativeDirection,
    EVPDefinition,
    EVPPromise,
    EVPAudit,
    EVPEmbedmentStage,
    EVPEmbedmentTouchpoint,
    EVPEmbedmentMessage,
)

from ..serializers import (
    TalentDatasetSerializer,
    AttributesOfGreatPlaceSerializer,
    KeyThemesSerializer,
    AudienceWiseMessagingSerializer,
    SwotAnalysisSerializer,
    AlignmentSerializer,
    MessagingHierarchyTabsSerializer,
    CreativeDirectionSerializer,
    EVPDefinitionSerializer,
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

def save_documents_to_master_vector_database():
    loader = PyPDFLoader(r"media\admin_merged_pdf\merged_pdf.pdf")
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
    documents = [text_chunks[i].page_content for i in range(len(text_chunks))]

    embeddings = create_embeddings()

    client = chromadb.PersistentClient(path="vector_databases/MasterVectorDatabase")

    collection = client.get_or_create_collection(
        name="master_database",
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"},
    )

    current_count = collection.count()
    ids = [f"id{current_count + i}" for i in range(len(documents))]
    embedded_documents = embeddings([documents[i] for i in range(len(documents))])

    collection.add(
        embeddings=embedded_documents,
        documents=documents,
        ids=ids,
    )

    return "Documents stored successfully!!!"

langchain_query = {
"Headquarters": """
        Search for the primary location of the company's headquarters. Look for terms like 'head office,' 'corporate office,' or 'main office location.' Identify the city, state, and country where the company's headquarters is located.
        The response should appear as city name, country name.
""",
"Established Date": """
        Find the date or year the company was established. Look for phrases like 'founded in,' 'established in,' or 'incorporated on.' Identify the specific year or date of the company's founding.
        The response should only be appear in yyyy format.
""",
"About the company": """
        Create a summary which gives the description about the company. Focus on how the company describes itself. Examples include "professional services firm" or "automobile company" or "global consulting company". Include a line on their product or services. Include a line on their clientele. 
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
                Response should only contain the number of employees. No other words or statements. Numeric response only.
""",
"Number of Geographies": """
                Identify the number of geographical locations where the company has operations. Search for phrases like 'number of locations,' 'geographical presence,' or 'global footprint.' List the distinct regions or countries where the company is active.
                Response should only list the geographies and not have any additional words.
""",
"LinkedIn URL and followers": """
                Search for information on the company's LinkedIn profile. Use keywords like 'LinkedIn profile,' 'LinkedIn company page,' or 'LinkedIn followers.' Provide the URL, follower count, and a summary of the company's activity on LinkedIn. Include the frequency of posts, the average number of likes per post, and the average number of comments per post.
""",
"Instagram URL and followers": """
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
                Intranet, emails, newsletter, townhalls, collaboration platform like Teams / Slack
""",
"Glassdorr Score": """
                Locate the company's Glassdoor score. Use terms like 'Glassdoor score,' 'Glassdoor rating,' or 'employee reviews score.' Provide the current rating and a summary of  how many reviews are being considered. Do not summarize the actual reviews.
""",
"What Retains Talent": """
                Identify the top three reasons why talent tends to stay with the company. Search the document for key phrases such as 'What Retains Talent,' 'Why Talent Chooses Us,' or similar terms. Summarize the common themes and specific factors that contribute to employee retention, highlighting any notable policies, benefits, or cultural aspects that are frequently mentioned.
""",
"What Attracts Talent": """
                Identify the key factors that attract talent to the company. Use keywords like 'talent attraction,' 'reasons to join,' or 'employment appeal.' Summarize the main attributes that make the company appealing to potential employees.
""",
"Employee Value Proposition": """
                Find if there is an existing Employee Value Proposition (EVP)and paste the actual statement here
""",
"Culture and Values": """
                Locate the culture and or values of the company from the secondary research section and paste here for user to validate
""",
"Purpose": """
                Locate the purpose of the company from the secondary research section and paste here for user to validate
""",
"Customer Value Proposition": """
                Locate the tagline / CVP of the company from the secondary research section and paste it as is for user to validate
""",
"Vision": """
                Locate the vision statement of the company from the secondary research section and paste here for user to validate
""",
"Mission": """
                Locate the mission statement of the company from the secondary research section and paste here for user to validate
""",
"Brand Guidelines": """
                Locate the colors, imagery guidelines, logo guidelines
"""
}

def query_with_langchain(company_name):
    loader = PyPDFLoader(r"media\final_pdf\merged_pdf.pdf")
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
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
        model=AZURE_OPENAI_DEPLOYMENT,
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

def get_talent_dataset_from_chatgpt(company_name, user):
    company = Company.objects.get(user=user, name=company_name)
    embeddings = create_embeddings()

    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    client = chromadb.PersistentClient(path=f"vector_databases/{sanitized_company_name}")

    collection = client.get_or_create_collection(
        name="test",
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"},
    )

    query = """
            Look for documents with titles like Job Description or Job Openings as well as the company's careers website and any presence on job websites including indeed.com, seek.com, LinkedIn jobs.
    """
    query_results = collection.query(
                query_texts=[query],
                n_results=40,
            )
    fetched_documents = " ".join(query_results["documents"][0])

    RESPONSE_JSON = {
        "talent_dataset": [
            {
                "id": "1",
                "area": "value",
                "role": "value",
                "location": "value",
                "seniority": "value"
            },
            {
                "id": "2",
                "area": "value",
                "role": "value",
                "location": "value",
                "seniority": "value"
            }
        ]
    }

    prompt = f"""First analyze the given Dataset given below and return the response in json format.

        Dataset: {fetched_documents}.

        Now from the given Dataset, fetch the complete information about :

        - Search for the type of area that is being advertised. Examples: Technology, HR, Admin, Legal, Sales etc.
        - Search for the designation/role or the position  that is being advertised. Examples: Software Developer, Sales Manager, etc.
        - Search for the location where the role is based. Examples: India, Manila - Philippines, Europe, North America, etc.
        - Search for the seniority or level of the role. Examples: Entry, Mid, Senior, Executive, Director etc.

        I have added examples just for your reference don't take the examples for granted and fetch the actual information in the given data.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Replace the value with the actual information.

        **Important : ** Area, Location and Seniority can be repeated but role cannot repeat.
        """
    
    completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={ "type": "json_object" },
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant.
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
    try:
        json_response = json.loads(chat_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}
    talent_dataset = json_response["talent_dataset"]
    # return talent_dataset

    for dataset in talent_dataset:
        TalentDataset.objects.create(
            user=user,
            company=company,
            area=dataset["area"],
            role=dataset["role"],
            location=dataset["location"],
            seniority=dataset["seniority"],
            key_motivators = "",
        )
    talent_datasets = TalentDataset.objects.filter(user=user, company=company)
    serializer = TalentDatasetSerializer(talent_datasets, many=True)
    return serializer.data

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
""",
"Exit Interview Feedback Summary": """
                Find feedback from exited employees about the company. Use keywords like 'exit feedback,' 'leaving form,' or 'exit interview'. Provide a summary of how many employees are represented and the topics they have provided feedback on. Do not summarize the actual feedback. 
""",
"Employee Feedback Summary": """
                Find feedback from current or former employees about the company. Use keywords like 'employee feedback,' 'staff opinions,' or 'employee reviews' or 'HR complaints'.  Provide a summary of how many employees are represented and the topics they have provided feedback on. Do not summarise the actual feedback.
""",
"Engagement Survey Result Summary": """
                Search for results from employee engagement surveys in the documents. Look for terms like 'engagement survey results,' 'employee satisfaction survey,' or 'engagement metrics.' or 'ESat survey' and provide a summary of how many employees are represented and the topics they have provided feedback on. Do not summarize the actual feedback.
""",
"Online Forums Mentions": """
                Pull verbatim online mentions of the company name and  the feedback about the company as an employer. Use keywords like 'feedback,' 'opinions,' 'reviews,' and 'perception. Provide the names of forums and how many mentions considered.
"""
}

def get_develop_data_from_vector_database(company_name, user):
    embeddings = create_embeddings()
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
        model=AZURE_OPENAI_DEPLOYMENT,
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
        model=AZURE_OPENAI_DEPLOYMENT,
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
        model=AZURE_OPENAI_DEPLOYMENT,
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
        exit_interview_feedback = json_data.get("Exit Interview Feedback Summary", ""),
        employee_feedback_summary = json_data.get("Employee Feedback Summary", ""),
        engagement_survey_results = json_data.get("Engagement Survey Result Summary", ""),
        online_forums_mentions = json_data.get("Online Forums Mentions", ""),
    )

def get_talent_insights_from_chatgpt(company_name, all_talent_dataset):
    company = Company.objects.get(name=company_name)
    embeddings = create_embeddings()

    client = chromadb.PersistentClient(path="vector_databases/MasterVectorDatabase")

    collection = client.get_collection(
        name="master_database",
        embedding_function=embeddings,
    )

    query = """
            Identify the key motivators and drivers for individuals.
            What inspires them to stay in their roles and perform well?
            Look for phrases including and similar to "career drivers" "career motivators" "job motivators" "Professional Aspirations" "Professional Drivers".
    """
    query_results = collection.query(
                query_texts=[query],
                n_results=40,
            )
    fetched_documents = " ".join(query_results["documents"][0])
    print(len(fetched_documents))

    RESPONSE_JSON = {
        "talent_insights": all_talent_dataset
    }

    prompt = f"""First analyze the given Dataset given below and return the response in json format.

        Dataset: {fetched_documents}.

        After analyzing the complete Dataset,
        Search for phrases including and similar to "career drivers" "career motivators" "job motivators" "Professional Aspirations" "Professional Drivers" corresponding to the job title.
        After searching, create a short paragraph to summarize it 100 words.

        Your task is to fill the actual data as the value of key_motivators in each object using the given dataset.

        Make sure to format the response in json exactly like {RESPONSE_JSON} and use it as a guide.
        Fill the vale of key_motivators with the actual information.
        """
    
    completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={ "type": "json_object" },
        messages = [
            {
                "role":"system",
                "content":"""You are a helpful expert research assistant.
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
    talent_insights = json_response["talent_insights"]
    return talent_insights

swot_analysis_query = {
"what_is_working_well_for_the_organization": """Identify the attributes that highlight what is working well for the organization. Focus on aspects that employees and external reviewers consistently praise or express satisfaction with. Provide detailed insights on these positive aspects and how they contribute to the overall success and positive reputation of the organization.
        Create multiple headings and give the description of those headings and summarize them.
""",
"what_is_not_working_well_for_the_organization": """Analyze the provided data to identify the top insights/themes that highlight what is not working well for the organization. Focus on aspects that employees and external reviewers consistently criticize or express concerns about. Provide detailed insights on these negative aspects and how they impact employee satisfaction and the overall performance of the organization.
        Create multiple headings and give the description after summarizing them as I only want summary.
"""
}

def get_dissect_data_from_vector_database(company_name, user, design_principles):
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
        model=AZURE_OPENAI_DEPLOYMENT,
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
        
    prompt = f"""First analyze both datasets below:

        Review the primary research available in entire Dataset 1 (both sections 'whats working well' and whats not working well).
        **Dataset 1** : {formatted_string}

        **what we want to be known for**: {design_principles}

        Now using the information available in Dataset 1, create a summary for each point mentioned in 'what we want to be known for' section.
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
        {"role": "user", "content": prompt}
    ]

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
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

    query = """Identify 4 themes that are unique about the company and will help it stand out as an employer. Focus on themes that are different from standard good HR practices. These themes should be believable about the company but also have an element of aspiration, which means that these could be things the company aspires towards and may not have completely achieve yet.
                Rank these themes from most relevant to least relevant and don't include numbers or anything just headings and description.
            """
    
    RESPONSE_JSON = {
            "themes": [
            {
                "id": "1",
                "tab_name": "heading1",
                "tabs_data": "description1",
            },
            {
                "id": "2",
                "tab_name": "heading2",
                "tabs_data": "description2",
            },
        ]
    }

    prompt = f"""
        Information: {formatted_string}.

        Analyze the complete information above.
        After analyzing it, give the response in json format.

        Question: {query}

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Replace headings and description with the actual value.
        """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    response_format={ "type": "json_object" },
    messages = [
        {
            "role":"system",
            "content":"""You are a helpful expert research assistant.
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

    return json_response["themes"]

def get_tagline(
        main_theme,
        combined_tabs_data,
        pillars
):
    supporting_pillars = ", ".join(pillars)
    if len(pillars) > 0:
        query = f"""
                    Act like an advertising expert. Now create a narrative. A narrative is a combination of a Tagline and advertising body copy. The logic for the narrative is that the {main_theme} will become the main theme of that narrative. The remaining {supporting_pillars} become secondary or supporting pillars. 
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
    model=AZURE_OPENAI_DEPLOYMENT,
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
    model=AZURE_OPENAI_DEPLOYMENT,
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

def get_evp_definition_from_chatgpt(company_name, user, analysis_data, alignment_data, all_themes):
    company = Company.objects.get(name=company_name)
    company_id = company.id

    RESPONSE_JSON = {
        "Theme": {
            "What it means": "Provide a simplified explanation of what the description means in an office or employee context.",
            "What it does not mean": "Consider literal meanings of the pillar / description and list things that don't seem reasonable in an office or employee context."
        }
    }

    prompt = f"""
                First analyze the given datasets below and return the data in json format:
                Analyze the Analysis Data.

                Analysis Data : {analysis_data}

                Now analyze the Alignment Data.

                Alignment Data : {alignment_data}

                All Themes Available: {all_themes}

                For each of the available themes, provide a detailed messaging overview that includes response for the below query:

                RESPONSE_JSON : {RESPONSE_JSON}

                The json format will contain keys same as the RESPONSE_JSON and value as the response to the query and replace Theme with the actual theme name.
                Don't include list in the response and don't include numbers or anything, I just want keys and the description.

                Make sure to format your response like RESPONSE_JSON and use it as a guide.
             """
    
    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
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
        what_it_means = value["What it means"]
        what_it_does_not_mean = value["What it does not mean"]

        EVPDefinition.objects.create(
            user=user,
            company = company,
            theme = theme,
            what_it_means = what_it_means,
            what_it_does_not_mean = what_it_does_not_mean,
        )

    evp_definition = EVPDefinition.objects.filter(user=user, company=company_id)
    serializer = EVPDefinitionSerializer(evp_definition, many=True)
    return serializer.data

def get_evp_promise_from_chatgpt(company_name, user, all_themes):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    RESPONSE_JSON = {
        "Theme": {
            "What employees can expect": "Create 3 points describing what employees can expect in relation to this theme.",
            "What is expected of employees": "Create 3 points outlining what is expected of employees in relation to this theme."
        }
    }

    prompt = f"""
                First analyze the given Themes Data below and return the data in json format:

                Themes Data : {all_themes}

                For each of the given themes, provide a detailed messaging overview that includes response for the below query:

                RESPONSE_JSON : {RESPONSE_JSON}

                The json format will contain keys same as the RESPONSE_JSON and value as the response to the query and replace Theme with the actual theme name.
                Don't include list in the response and don't include numbers or anything, I just want keys and the description.

                Make sure to format your response like RESPONSE_JSON and use it as a guide.
             """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
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
        what_employees_can_expect = value["What employees can expect"]
        what_is_expected_of_employees = value["What is expected of employees"]

        EVPPromise.objects.create(
            user=user,
            company = company,
            theme = theme,
            what_employees_can_expect = what_employees_can_expect,
            what_is_expected_of_employees = what_is_expected_of_employees,
        )

    evp_promise = EVPPromise.objects.filter(user=user, company=company_id)
    serializer = EVPPromiseSerializer(evp_promise, many=True)
    return serializer.data

def get_evp_audit_from_chatgpt(company_name, user, analysis_data, alignment_data, all_themes):
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

                Themes Available: {all_themes}

                For each of the available themes, provide a detailed messaging overview that includes response for the below query:

                RESPONSE_JSON : {RESPONSE_JSON}

                The json format will contain keys same as the RESPONSE_JSON and value as the response to the query and replace Theme with the actual theme name.
                Don't include list in the response and don't include numbers or anything, I just want keys and the description.

                Make sure to format your response like RESPONSE_JSON and use it as a guide.
             """
    
    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
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

all_touchpoint_prompts = {
"Careers Website": """Create content for the Careers section of our website that highlights our company culture, benefits, and career growth opportunities. Include testimonials from current employees and visuals of our work environment.
""",
"LinkedIn": """Draft a LinkedIn post to attract potential candidates. Include a call-to-action for interested candidates to visit our careers page
""",
"Instagram": """Design an Instagram post showcasing a day in the life at our company.  a caption that highlights the fun and collaborative work environment.
""",
"Facebook": """Generate a Facebook post announcing our open positions and the benefits of working with us. Include a link to the job application page and encourage followers to share the post.
""",
"Tiktok": """Write a video script for TikTok post. The video should be engaging and fun, encouraging viewers to apply for open positions.
""",
"Twitter X": """Compose a tweet to announce job openings at our company. Highlight key benefits and provide a link to the application page. Use relevant hashtags to increase visibility.
""",
"Job Board": """Create a job board posting that details the responsibilities, requirements, and benefits of the open position. Make sure to include information about our company culture and growth opportunities
""",
"Job Description": """Write a detailed job description for the open position, including responsibilities, required skills, and qualifications. Highlight opportunities for career growth and development.
""",
"Job Ad": """Generate a job ad that will catch the attention of potential candidates. Focus on the benefits of working at our company and include a clear call-to-action for applying.
""",
"Referral Email": """Draft an email template for employees to refer candidates for open positions. Include information about the referral program and the benefits of working at our company.
""",
"Event Toolkit": """Create a toolkit for hiring events that includes promotional materials, banners, flyers, and information packets about our company and open positions.
""",
"Interview Talking Points": """Generate a list of talking points for interviewers to use during candidate interviews. Include key information about the company, role expectations, and career growth opportunities.
""",
"Application Form": """Design a user-friendly job application form that captures essential candidate information while providing a seamless application experience.
""",
"Offer Letter": """Draft a template for offer letters to be sent to successful candidates. Ensure it includes details about the role, compensation, benefits, and start date.
""",
"Welcome Email": """Create a welcome email template for new employees. The email should express enthusiasm, provide essential information about their first day, and include links to resources like the employee handbook.
""",
"Employee Handbook": """Develop an employee handbook that outlines company policies, procedures, and culture. Include sections on benefits, code of conduct, and employee resources.
""",
"Merchandise": """Generate ideas for onboarding merchandise, such as branded t-shirts, mugs, and notebooks, that new employees can receive on their first day.
""",
"Orientation Deck": """Design an orientation presentation deck that introduces new employees to the company's mission, values, and key personnel. Include information about company history and future goals.
""",
"Email Template": """Create an internal email template for company-wide announcements and updates. Ensure it is visually appealing and easy to read, with space for images and important links
""",
"PPT Template": """Design a PowerPoint template for internal presentations. The template should be branded with the company logo and colors, and include slide layouts for various types of content.
""",
"Living the EVP Module": """Develop a training module that helps employees understand and live the Employee Value Proposition (EVP). Include interactive elements, real-life examples, and exercises that reinforce the company’s values and culture
""",
"Goal Setting": """Create a goal-setting template for employees to outline their objectives and key results (OKRs). Ensure it includes sections for personal development and alignment with company goals.
""",
"Feedback Mechanism": """Develop a feedback mechanism that allows employees to provide and receive constructive feedback. Include templates for 360-degree feedback, performance reviews, and peer feedback.
""",
"Posters": """Design posters that reflect the company's values and culture to be displayed around the office. Ensure they are visually appealing and motivational.
""",
"Wall Branding": """Create wall branding ideas that incorporate the company's logo, colors, and mission statement. Focus on areas like the lobby, meeting rooms, and common areas
""",
"Breakout Areas": """Provide design concepts for breakout areas that encourage relaxation and collaboration. Include furniture suggestions and layout ideas.
""",
"Overall Layout": """Develop a floor plan layout that maximizes space efficiency and fosters a productive work environment. Consider incorporating open-plan areas, private workstations, and collaboration zones.
""",
"Increment Letter": """Draft a template for increment letters that inform employees about their salary increases. Include details about the new compensation, effective date, and reasons for the increase.
""",
"Promotion Letter": """Create a promotion letter template to congratulate employees on their new role. Include information about the new position, responsibilities, and any changes in compensation.
""",
"Correction Letter": """Generate a template for correction letters to address any discrepancies or changes in employee compensation. Ensure it is clear and professional.
""",
"Reward and Recognition Program": """Develop a reward and recognition program that outlines how employees can be acknowledged for their achievements. Include guidelines for nominations, selection criteria, and types of rewards.
""",
"Employee Survey": """Design an employee engagement survey to gather feedback on various aspects of the workplace, including satisfaction, culture, and areas for improvement.
""",
"Engagement Activity Calendar": """Create a calendar of employee engagement activities, including team-building events, social gatherings, and professional development opportunities.
""",
"Exit Interview": """Develop an exit interview template to gather feedback from departing employees. Include questions about their experiences, reasons for leaving, and suggestions for improvement.
""",
"Exit Process": """Create a detailed exit process checklist to ensure a smooth transition for departing employees. Include steps for returning company property, finalizing paperwork, and conducting exit interviews.
""",
"Farewell Communication Template": """Generate a template for farewell communications to announce an employee's departure. Ensure it is respectful and expresses gratitude for their contributions.
"""
}

def get_evp_embedment_data_from_chatgpt(company_name, user, all_touchPoints, top_4_themes_data, tagline_data, evp_promise_data, evp_audit_data):
    company = Company.objects.get(name=company_name)
    company_id = company.id

    base_prompt = f"""
                First analyze the Themes Data
                Themes Data: {top_4_themes_data}

                Now analyze the Tagline Data
                Tagline Data : {tagline_data}

                Now analyze the EVP Promise Data
                EVP Promise Data : {evp_promise_data}

                Now analyze the EVP Audit Data
                EVP Audit Data : {evp_audit_data}

                After analyzing all these datasets, give fetch the required information below about the company named {company_name} from the given data:

              """
    
    json_data = {}
    for stage, touchpoints in all_touchPoints.items():
        if len(touchpoints) > 0:
            json_data[stage] = {}
            for touchpoint in touchpoints:
                print(touchpoint)
                information_to_fetch = all_touchpoint_prompts.get(touchpoint, "")
                prompt = base_prompt + information_to_fetch.format(all_touchpoint_prompts[touchpoint])

                completion = chat_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages = [
                    {
                        "role":"system",
                        "content":"""You are an expert in fetching information from the given data.
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
                json_data[stage][touchpoint] = chat_response
    return json_data

    # for stage_name, touchpoints in json_data.items():
    #     stage, created = EVPEmbedmentStage.objects.get_or_create(
    #         user = user,
    #         company = company,
    #         stage_name = stage_name
    #     )
    #     for touchpoint_name, message_content in touchpoints.items():
    #         touchpoint, created = EVPEmbedmentTouchpoint.objects.get_or_create(
    #             user = user,
    #             company = company,
    #             stage = stage,
    #             touchpoint_name = touchpoint_name,
    #         )
    #         EVPEmbedmentMessage.objects.create(
    #             user = user,
    #             company = company,
    #             touchpoint = touchpoint,
    #             message = message_content,
    #         )
    
    # return json_data

def get_evp_handbook_data_from_chatgpt(company_name, user, top_4_themes_data, messaging_hierarchy_data, evp_promise_data, evp_audit_data):

    prompt = f"""
                First analyze the Top 4 Themes Data :

                Top 4 Themes Data : {top_4_themes_data}

                Now analyze the Messaging Hierarchy Data :

                Messaging Hierarchy Data : {messaging_hierarchy_data}

                Now analyze the EVP Promise Data :

                EVP Promise Data : {evp_promise_data}

                Now analyze the EVP Audit Data :

                EVP Audit Data : {evp_audit_data}

                After analyzing the complete given data, generate the data for below :

                Overview
                a.       Introduction - one paragraph on what is this EVP exercise about
                c. Chairman's Letter - email for employees from Ashish Agrawal introducing the EVP
                b.       Journey - Summarises the EVP Narrative section in 3 paragraphs or less 
                c.       Definition of terms - All technical terms used in the all sections
                d.       The EVP - The Positioning Statement ( using tagline ) and 3 Pillars
                e.       The EVP Promise table - create a two line definition and then place the EVP Promise table
                f.        Creative Direction - rationale for why this creative direction, imagery, and colore palette has been used
                g.       Brand voice - what is the brand voice and personality of the EVP
                
                2-      Design Elements
                Typerface - Which font is recommended?
                i.      Usage examples
                b.       Colours - exact shades of colours
                c.       Imagery - examples of images used already and suggested images
                
                3-      Content
                a.       Copy - The main body copy for the EVP positioning
                b.       Do's and Don't's for the ads
                c.       Copy Bank ( Job Ads, Emails)
                d.       Guidelines for writing for  social media
                e.       Employee Testimonial Guide - how should these be created
                4-      Execution Plan
             """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    messages = [
        {
            "role":"system",
            "content":"""You are an expert in fetching information from the given data.
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
        model=AZURE_OPENAI_DEPLOYMENT,
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