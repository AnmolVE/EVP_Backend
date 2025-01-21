import os
import re
import json
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import (
    DesignPrinciples,
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
    DesignPrinciplesSerializer,
    TalentDatasetSerializer,
    TalentInsightsSerializer,
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
"headquarters": """
        Search for the primary location of the company's headquarters. Look for terms like 'head office,' 'corporate office,' or 'main office location.' Identify the city, state, and country where the company's headquarters is located.
        The response should appear as city name, country name.
""",
"established_date": """
        Find the date or year the company was established. Look for phrases like 'founded in,' 'established in,' or 'incorporated on.' Identify the specific year or date of the company's founding.
        The response should only be appear in yyyy format.
""",
"about_the_company": """
        Create a summary which gives the description about the company. Focus on how the company describes itself. Examples include "professional services firm" or "automobile company" or "global consulting company". Include a line on their product or services. Include a line on their clientele. 
""",
"industry": """
        Identify the industry or sector the company operates in. Use terms like 'industry sector,' 'business sector,' or 'industry classification.' Specify the primary market or sector the company is associated with.
""",
"company_financials": """
        Locate financial information about the company. Search for terms like 'financial performance,' 'company revenue,' 'annual report,' or 'financial statement.' Provide data on the company's revenue, profits, and overall financial health.
""",
"company_history": """
                Provide a detailed overview of company's history, focusing on its founding, major milestones, key product developments, and significant shifts in strategy or market presence. Include information on the evolution of its leadership, notable acquisitions or mergers, and any major challenges or controversies it has faced. Summarize how these events have shaped the company's current status in its industry.
""",
"top_3_competitors": """
                Find the top three competitors of the company. Use keywords like 'main competitors,' 'industry competitors,' or 'competitive landscape.' Identify and list the top three companies competing with the organization.
""",
"number_of_employees": """
                Search for the total number of employees in the company. Use terms like 'employee count,' 'number of employees,' or 'company workforce.' Provide the latest available employee count.
                Response should only contain the number of employees. No other words or statements. Numeric response only.
""",
"number_of_geographies": """
                Identify the number of geographical locations where the company has operations. Search for phrases like 'number of locations,' 'geographical presence,' or 'global footprint.' List the distinct regions or countries where the company is active.
                Response should only list the geographies and not have any additional words.
""",
"linked_info": """
                Search for information on the company's LinkedIn profile. Use keywords like 'LinkedIn profile,' 'LinkedIn company page,' or 'LinkedIn followers.' Provide the URL, follower count, and a summary of the company's activity on LinkedIn. Include the frequency of posts, the average number of likes per post, and the average number of comments per post.
""",
"instagram_info": """
                Find information on the company's Instagram profile. Use keywords like 'Instagram profile,' 'Instagram company page,' or 'Instagram followers.' Provide the URL, follower count, and a summary of the company's activity on Instagram.
""",
"facebook_info": """
                Locate information on the company's Facebook page. Use keywords like 'Facebook profile,' 'Facebook page,' or 'Facebook followers.' Provide the URL, follower count, and a summary of the company's activity on Facebook.
""",
"twitter_info": """
                Find information on the company's (X)  Twitter profile. Use keywords like 'Twitter profile,' 'Twitter page,' or 'Twitter followers.' Provide the URL, follower count, and a summary of the company's activity on Twitter.
""",
"glassdoor_score": """
                Locate the company's Glassdoor score. Use terms like 'Glassdoor score,' 'Glassdoor rating,' or 'employee reviews score.' Provide the current rating and a summary of  how many reviews are being considered. Do not summarize the actual reviews.
""",
"employee_value_proposition": """
                Find if there is an existing Employee Value Proposition (EVP) and paste the actual statement here.
""",
"customer_value_proposition": """
                Locate the tagline / CVP of the company.te
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
"""
}

def query_with_langchain(company_name, collection):
    
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
        print(json_data)
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

design_principles_questions = {
    """question_1""":"""What are the strategic goals for the next 3-5 years?""",
    """question_2""":"""Do you have an existing EVP? If so, what aspects of it are working well, and what areas need improvement?""",
    """question_3""":"""What are the key attributes or messages that you want to convey through your EVP?""",
    """question_4""":"""How would you describe your company culture?""",
    """question_5""":"""What values are most important to your organization and its employees?""",
    """question_6""":"""What challenges do you currently face in attracting and retaining top talent?""",
    """question_7""":"""What are the key reasons employees stay at your company? What are the reasons they leave?""",
    """question_8""":"""What talent segment(s) do you most want your EVP to target?""",
    """question_9""":"""How do you differentiate your company's employee experience from competitors?""",
    """question_10""":"""How do you currently measure employee satisfaction and engagement?""",
    """question_11""":"""What channels do you use to communicate with employees and potential candidates?""",
    """question_12""":"""How do you plan to measure the success and impact of the new EVP?""",
    """question_13""":"""How do you believe your company is perceived by potential candidates in the market?""",
    """question_14""":"""What are the key messages you want to convey to the market about working at your company?""",
    """question_15""":"""What are your competitors doing in terms of EVP that you admire or want to differentiate from?""",
}

def get_design_principles(company_name):
    loader = PyPDFLoader(r"media\final_pdf\merged_pdf.pdf")
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
    documents = [text_chunks[i].page_content for i in range(len(text_chunks))]

    embeddings = create_embeddings()

    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    client = chromadb.PersistentClient(path=f"vector_databases/{sanitized_company_name}")
    collection = client.get_or_create_collection(
        name="design_principles",
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

    json_data = {}
    for key, question in design_principles_questions.items():
        print(key)
        query_results = collection.query(
                query_texts=[question],
                n_results=20,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        RESPONSE_JSON = {
            key:question,
        }

        prompt = f"""
                    First analyze the given information and returns the response in json format:

                    Given Information : {fetched_documents}

                    From the given information, fetch the data for below question
                    question : {question}

                    Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
                    Replace the question with the actual data and keys remains as it is.
                """
        
        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={ "type": "json_object" },
        messages = [
                {"role": "system", "content": f"You are an expert Research Analyst."},
                {"role": "user", "content": prompt}
            ],
        temperature=0.1,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            design_principles = json.loads(chat_response)
            json_data[key] = design_principles[key]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            json_data = {}
    return json_data


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

def save_data_to_vector_database(data_to_save, file_path, company_name):
    data = [str(value) for key, value in data_to_save.items() if key not in ["id", "user"]]

    string_data = "\n\n".join(data)

    with open(file_path, "w") as file:
        file.write(string_data)

    loader = TextLoader(file_path)
    document_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(document_data)
    documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
    ids=[f"id{i}" for i in range(len(documents))]

    embeddings = create_embeddings()

    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    client = chromadb.PersistentClient(path=persistent_directory)
    if os.path.exists(os.path.join(persistent_directory)):
        collection = client.get_or_create_collection(
            name="test",
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

def get_attributes_of_great_place_from_chatgpt(company_name):
    embeddings = create_embeddings()
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        develop_collection = chroma_client.get_collection(
            name="test",
            embedding_function=embeddings,
        )
    print("In Attributes of Great Place")

    json_data = {}
    for key, query in attributes_of_great_place_query.items():
        print(key)

        RESPONSE_JSON = {
            key: query,
        }

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        First analyze the given information below and returns the response in json format:

        Given Information: {fetched_documents}

        After completely analyzing the given information, fetch the data for the below query from the given information only.

        Question: {query}.

        Don't create any heading or sub heading, just give the response as a single paragraph.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Add keys as it is and replace the value with the actual data.
        """

        print(prompt)
        print("*************************************************************************************************************")

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
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            attributes = json.loads(chat_response)
            json_data[key] = attributes[key]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            attributes = {}
    return json_data
        

key_themes_query = {
"top_key_themes":"""
        What are some key themes related to building the company's employee value proposition? Focus on themes that help the company stand out as an attractive employer. What are unique aspects about the company that can help to attract and retain the best talent?
"""
}

def get_key_themes_from_chatgpt(company_name):
    embeddings = create_embeddings()
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        develop_collection = chroma_client.get_collection(
            name="test",
            embedding_function=embeddings,
        )
    print("In Key Themes")

    key_themes = {}
    for key, query in key_themes_query.items():
        print(key)

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        RESPONSE_JSON = {
        "top_key_themes": [
            {
                "theme": "value",
                "theme_description": "value",
            },
            {
                "theme": "value",
                "theme_description": "value",
            }
        ]
    }

        prompt = f"""
        First analyze the given information below and return the response in json format:

        Given Information: {fetched_documents}

        After completely analyzing the given information, fetch the data for the below query from the given information only.

        Question: {query}.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Replace value with the actual data.
        """

        print(prompt)
        print("*************************************************************************************************************")

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
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            key_themes = json.loads(chat_response)
            key_themes = key_themes["top_key_themes"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            key_themes = {}
        return key_themes
    
def get_regenerated_themes(company_name, all_key_themes, theme_to_update):
    embeddings = create_embeddings()
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        develop_collection = chroma_client.get_collection(
            name="test",
            embedding_function=embeddings,
        )
    print("In Key Themes")

    key_themes = {}
    for key, query in key_themes_query.items():
        print(key)

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        RESPONSE_JSON = {
            "regenerated_themes": all_key_themes
        }

        prompt = f"""
        First analyze the given information below and return the response in json format:

        Given Information: {fetched_documents}

        My client does not like below theme and wants to get "key_theme" and "key_theme_desc" in below data so that they should not even similar to previous one.
        Don't delete the below object just update the values of "key_theme" and "key_theme_desc" in below data.
        Theme: {theme_to_update}

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Update the data of regenerated theme and let other themes data as it is.

        The number of entries should not exceed the available data.
        """

        print(prompt)
        print("*************************************************************************************************************")

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
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            key_themes = json.loads(chat_response)
            key_themes = key_themes["regenerated_themes"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            key_themes = {}
        return key_themes


audience_wise_messaging_query = {
"Existing Employees":"""
    Create a short summary about what Existing Employees are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Alumni":"""
    Create a short summary about what Alumni are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Targeted Talent":"""
    Create a short summary about what Targeted Talent are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN. 
""",
"Leadership":"""
    Create a short summary about what Leadership are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Recruiters":"""
    Create a short summary about what Recruiters are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Clients":"""
    Create a short summary about what Clients are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Offer Drops":"""
    Create a short summary about what people who interviewed but did not accept the offer to join are saying about company. Only rely on given information or actual external mentions. Do not make up any facts or figures on your own. This is extremely important - DO NOT MAKE UP ANYTHING ON YOUR OWN.
""",
"Exit Interview Feedback Summary": """
                Create a short summary of what exiting employees said during their exit interviews. Remember, do no make anything up. Only summarise findings of that particular document. If no such document is found, please write "Information not found".
""",
"Employee Feedback Summary": """
                Find feedback from current or former employees about the company. Use keywords like 'employee feedback,' 'staff opinions,' or 'employee reviews' or 'HR complaints'.  Provide a summary of how many employees are represented and the topics they have provided feedback on. Do not summarise the actual feedback.
""",
"Engagement Survey Result Summary": """
                Search for results from employee engagement surveys in the documents. Look for terms like 'engagement survey results,' 'employee satisfaction survey,' or 'engagement metrics.' or 'ESat survey' and provide a summary of how many employees are represented and the topics they have provided feedback on. Do not summarize the actual feedback.
""",
"Online Forums Mentions": """
                Crawl the documents and identify any feedback specific to online forums only such as Glassdoor.com or Reddit.com. Then summarise those mentions in a short summary. Do not summarise anything other than mentions on online forums.
"""
}

def get_audience_wise_messaging_from_chatgpt(company_name):
    embeddings = create_embeddings()
    sanitized_company_name = re.sub(r'\s+', '_', company_name)
    persistent_directory = f"vector_databases/{sanitized_company_name}"
    if os.path.exists(os.path.join(persistent_directory)):
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        develop_collection = chroma_client.get_collection(
            name="test",
            embedding_function=embeddings,
        )
    print("In Audience Wise Messaging")

    json_data = {}
    for key, query in audience_wise_messaging_query.items():
        print(key)

        RESPONSE_JSON = {
            key: query,
        }

        query_results = develop_collection.query(
                query_texts=[query],
                n_results=10,
            )
        fetched_documents = " ".join(query_results["documents"][0])

        prompt = f"""
        First analyze the given information below and returns the response in json format:

        Given Information: {fetched_documents}

        After completely analyzing the given information, fetch the data for the below query from the given information only.

        Question: {query}.

        Don't create any heading or sub heading, just give the response as a single paragraph.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Add keys as it is and replace the value with the actual data.
        """

        print(prompt)
        print("*************************************************************************************************************")

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
        temperature=0,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            audiences = json.loads(chat_response)
            json_data[key] = audiences[key]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            audiences = {}
    return json_data

def get_talent_insights_from_chatgpt(company_name):
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

    talent_dataset = {}

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
        talent_dataset = json.loads(chat_response)
        talent_dataset = talent_dataset["talent_dataset"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        talent_dataset = {}

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
        "talent_insights": talent_dataset
    }

    talent_insights = {}

    prompt = f"""First analyze the given Dataset given below and return the response in json format.

        Dataset: {fetched_documents}.

        After analyzing the complete Dataset,
        Search for phrases including and similar to "career drivers" "career motivators" "job motivators" "Professional Aspirations" "Professional Drivers" corresponding to the job title.
        After searching, create a short paragraph to summarize it 100 words.

        Your task is to fill the actual data as the value of key_motivators in each object using the given dataset.
        And do not repeat key_motivators in multiple objects.

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
        talent_insights = json.loads(chat_response)
        talent_insights = talent_insights["talent_insights"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        talent_insights = {}

    return talent_insights

swot_analysis_query = {
"what_is_working_well_for_the_organization": """Identify the attributes that highlight what is working well for the organization. Focus on aspects that employees and external reviewers consistently praise or express satisfaction with. Provide detailed insights on these positive aspects and how they contribute to the overall success and positive reputation of the organization.
""",
"what_is_not_working_well_for_the_organization": """Analyze the provided data to identify the top insights/themes that highlight what is not working well for the organization. Focus on aspects that employees and external reviewers consistently criticize or express concerns about. Provide detailed insights on these negative aspects and how they impact employee satisfaction and the overall performance of the organization.
"""
}

def get_analysis_data_from_vector_chatgpt(company, user):
    attributes_of_great_place_vector = AttributesOfGreatPlace.objects.get(user=user, company=company)
    attributes_of_great_place_vector_serializer = AttributesOfGreatPlaceSerializer(attributes_of_great_place_vector)

    key_themes_vector = KeyThemes.objects.filter(user=user, company=company)
    key_themes_vector_serializer = KeyThemesSerializer(key_themes_vector, many=True)

    audience_wise_messaging_vector = AudienceWiseMessaging.objects.get(user=user, company=company)
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

        RESPONSE_JSON = {
            key: query,
        }

        prompt = f"""
                I want to fetch the information from the given data.
                First analyze the complete data below and returns the response in json format:
                data : {formatted_string}

                Now fetch the below information from the data and give me 5 points on this.
                {query} and returns the response. 

                **Note :** Fetch the complete information from the given data only and don't include anything extra.
                Don't add anything which you don't find in the given data just give the information which is available in the given data.

                Don't create any heading or sub heading, just give the response as a single paragraph.

                Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
                Add keys as it is and replace the value with the actual data.
        """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={ "type": "json_object" },
        messages = [
                {"role": "system", "content": f"You are an expert Research Analyst."},
                {"role": "user", "content": prompt}
            ],
        temperature=0.7,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        try:
            analysis = json.loads(chat_response)
            json_data[key] = analysis[key]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            analysis = {}
    return json_data

def get_alignment_data_from_vector_database(company, user, design_principles):
    attributes_of_great_place_vector = AttributesOfGreatPlace.objects.get(user=user, company=company)
    attributes_of_great_place_vector_serializer = AttributesOfGreatPlaceSerializer(attributes_of_great_place_vector)

    key_themes_vector = KeyThemes.objects.filter(user=user, company=company)
    key_themes_vector_serializer = KeyThemesSerializer(key_themes_vector, many=True)

    audience_wise_messaging_vector = AudienceWiseMessaging.objects.get(user=user, company=company)
    audience_wise_messaging_vector_serializer = AudienceWiseMessagingSerializer(audience_wise_messaging_vector)

    whole_data = {
        "attributes_of_great_place_vector": attributes_of_great_place_vector_serializer.data,
        "key_themes_vector": key_themes_vector_serializer.data,
        "audience_wise_messaging_vector": audience_wise_messaging_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)
    print(len(formatted_string))

    RESPONSE_JSON = {
        "alignment": [
            {
                "theme_name": "",
                "positive_aspects": "",
                "negative_aspects": "",
            },
            {
                "theme_name": "",
                "positive_aspects": "",
                "negative_aspects": "",
            }
        ]
    }
    prompt = f"""First analyze both datasets below and returns the response in json format:

        Review the primary research available in entire Dataset 1 (both sections 'whats working well' and whats not working well).
        **Dataset 1** : {formatted_string}

        **Design Principles**: {design_principles}

        -Identify and extract 5 key themes that the company wants to be known for, based on the design principles questions.
        -Using the information available in Dataset 1,provide a detailed summary that captures both positive and negative aspects for each theme.

        -Output Structure: Each summary should be divided into two sub-sections:
            Positive Aspects: Detail what is working well for the organization. Focus on elements that employees and external reviewers consistently highlight with positivity. Provide specific insights on how these positive aspects contribute to employee satisfaction and overall organizational performance.
            Negative Aspects: Detail what is not working well for the organization. Highlight recurring criticisms or concerns raised by employees and external reviewers. Provide specific insights on how these negative aspects impact employee satisfaction and overall organizational performance.

        -Guidelines:
            Extract information only from Dataset 1. If information is not available, state: "The information is not available."
            Do not rely solely on exact word or phrase matches; intelligently correlate data points by focusing on the underlying meaning and context of the responses.
            Don't create any heading or sub heading in positive and negative aspects, , just give the response as a single paragraph.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Add actual data as the value of keys.
    """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    response_format={ "type": "json_object" },
    messages = [
            {"role": "system", "content": f"You are an expert Research Analyst."},
            {"role": "user", "content": prompt}
        ],
    temperature=0.1,
    max_tokens=4000,
    )
    chat_response = completion.choices[0].message.content
    try:
        alignment = json.loads(chat_response)
        alignment = alignment["alignment"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        alignment = {}
    return alignment

def get_evp_statement_themes_from_chatgpt(company_name, user):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    analysis_vector = SwotAnalysis.objects.get(user=user, company=company_id)
    analysis_vector_serializer = SwotAnalysisSerializer(analysis_vector)

    alignment_vector = Alignment.objects.filter(user=user, company=company_id)
    alignment_vector_serializer = AlignmentSerializer(alignment_vector, many=True)

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
                "theme_name": "heading1",
                "theme_desc": "description1",
            },
            {
                "id": "2",
                "theme_name": "heading2",
                "theme_desc": "description2",
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
        json_response = json_response["themes"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

def get_regenerated_evp_statement_themes(company_name, user, all_evp_statement_themes, evp_statement_theme_to_update):
    company = Company.objects.get(user=user, name=company_name)
    company_id = company.id

    analysis_vector = SwotAnalysis.objects.get(user=user, company=company_id)
    analysis_vector_serializer = SwotAnalysisSerializer(analysis_vector)

    alignment_vector = Alignment.objects.filter(user=user, company=company_id)
    alignment_vector_serializer = AlignmentSerializer(alignment_vector, many=True)

    whole_data = {
        "analysis_vector": analysis_vector_serializer.data,
        "alignment_vector": alignment_vector_serializer.data,
    }

    formatted_string = json.dumps(whole_data)
    
    RESPONSE_JSON = {
        "regenerated_themes": all_evp_statement_themes
    }

    prompt = f"""First analyze the given information completely and return the response in json format

        Given Information: {formatted_string}.

        By using the above given information, do the following

        My client does not like below theme and wants to get "theme_name" and "theme_desc" in below data so that they should not even similar to previous one.
        Don't delete the below object just update the values of "theme_name" and "theme_desc" in below data.
        Theme: {evp_statement_theme_to_update}

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Update the data of regenerated theme and let other themes data as it is.

        The number of entries should not exceed the available data.
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
        json_response = json_response["regenerated_themes"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

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

def get_regenerated_theme(company_name, user, theme_to_regenerate):
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

    RESPONSE_JSON = {
        "regenerated_theme": theme_to_regenerate
    }

    prompt = f"""
                I want to regenerate the given themes using the given information.
                So regenerate the theme and returns the data in json format.

                Given Information : {formatted_string}

                Given Themes : {theme_to_regenerate}

                Regenerate both tab name and tabs data as well as i want to get both to be changed.
                And don't create headings in tabs data only paragraph.

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
        json_response = json_response["regenerated_theme"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

def get_evp_statement(
        main_theme,
        evp_statement_themes_data,
        pillar_1,
        pillar_2,
        pillar_3
):
    query = f"""
                    Act like an advertising expert. Now create a narrative. A narrative is a combination of a Tagline and advertising body copy. The logic for the narrative is that the {main_theme} will become the main theme of that narrative. The remaining themes {pillar_1}, {pillar_2} and {pillar_3} become secondary or supporting themes. 
                    This is how the advertising copy of the narrative will flow. Don't write the subheads below, but follow the instructions.
                    Start with a hook or an engaging statement that captures the reader's attention.
                    Provide a clear and concise explanation of the main theme.
                    Now link the main theme to the supporting themes.
                    Use language that resonates emotionally with employees and potential employees, creating a connection.
                    Emphasize what sets the company apart from competitors.
                    End with a strong call to action, encouraging the audience to take the next step, such as joining the company.
                    Don't include the name of the company in the taglines. But consider the industry of the company.
                """
    
    RESPONSE_JSON = {
        "evp_statement": {
            "tagline": "value",
            "advertising_body_copy": "",
        }
    }
    
    prompt = f"""First analyze the given information below and returns the response in json format
        Given Information: {evp_statement_themes_data}

        After analyzing it, give the response regarding below query.
         
        Query: {query}.

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Add actual data as the value of keys and let keys as it as.
        """

    completion = chat_client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    response_format={ "type": "json_object" },
    messages = [
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
        json_response = json_response["evp_statement"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

def get_regenerated_evp_statement(evp_statement_themes_data, evp_statement_to_update):

    RESPONSE_JSON = {
        "evp_statement": evp_statement_to_update
    }

    prompt = f"""First analyze the given information completely and return the response in json format

        Given Information: {evp_statement_themes_data}.

        By using the above given information, do the following

        My client does not like below statement and wants to get new "tagline" and "tagline_desc" in below data so that they should not even similar to previous one.
        Don't delete the below object just update the values of "tagline" and "tagline_desc" in below data and let other fields as it is
        Statement: {evp_statement_to_update}

        Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
        Update the data of regenerated theme and let other themes data as it is.

        The number of entries should not exceed the available data.
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
        json_response = json_response["evp_statement"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

def get_creative_direction_from_chatgpt(brand_guidelines, tagline):

    RESPONSE_JSON = {
        "creative_direction": {
            "tagline": "value",
            "visual_concept": "value",
        }
    }
    
    prompt = f"""
                First analyze the brand guidelines and tagline given below and returns the response in json format:

                Brand Guidelines : {brand_guidelines}

                Tagline : {tagline}

                Now suggest a single visual that captures the tagline and advertising body copy from the messaging hierarchy section while adhering to the company's brand guidelines in terms of color, style, tone etc as well as the industry of the company. Focus on creating one visual that focuses mainly on the 'overarching theme' and very subtly incorporates the secondary pillars.

                Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
                Replace value with the actual data.
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
        json_response = json_response["creative_direction"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

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
"Social Media Ads": """Create a compelling social media ad campaign that highlights the key pillars of our Employee Value Proposition (EVP). The campaign should target potential candidates at the pre-hire stage, emphasizing [insert specific EVP attributes from the design section]. Include a catchy headline, engaging visuals (described), and a clear call-to-action for candidates to learn more or apply.
""",
"EVP Blog Post": """Write a detailed blog post that showcases our company's unique Employee Value Proposition (EVP). The post should be engaging, informative, and reflect our brand's voice. Focus on [insert specific EVP elements], and explain how these translate into real benefits for potential employees. Provide a strong conclusion that encourages readers to explore job opportunities with us.
""",
"EVP Brochure": """Develop a visually appealing brochure that outlines our Employee Value Proposition (EVP) for prospective candidates. The brochure should cover the core pillars of our EVP, including [insert EVP details], and should include testimonials, benefits, and a clear overview of what makes us stand out as an employer. Ensure the design is aligned with our brand identity.
""",
"EVP Infographic": """Design an infographic that succinctly summarizes our Employee Value Proposition (EVP). The infographic should be visually engaging and easy to understand, highlighting the key benefits and unique aspects of working with our company. Focus on [insert EVP details], and ensure the design is optimized for sharing on social media and in recruitment materials.
""",
"Outreach Letter to Talent": """Craft a personalized outreach letter aimed at top talent in our industry. The letter should introduce our company, emphasize our Employee Value Proposition (EVP), and explain why we believe the recipient would be a great fit. Include specific EVP attributes that align with the recipient's background or interests, and invite them to discuss potential opportunities.
""",
"Culture Video": """Create a script for a culture video that highlights our company’s work environment, values, and Employee Value Proposition (EVP). The video should tell a story that resonates with prospective candidates, showcasing [insert specific aspects of EVP] through employee interviews, office scenes, and examples of company culture in action. Include ideas for visuals and narration.
""",
"Candidate Feedback Survey": """Develop a candidate feedback survey tailored to those who have interacted with our company during the pre-hire stage. The survey should focus on assessing their experience with our recruitment process, particularly how well our Employee Value Proposition (EVP) was communicated. Include questions that capture both quantitative and qualitative feedback.
""",
"Employee Testimonial Format": """Design a format for collecting and sharing employee testimonials that emphasize our Employee Value Proposition (EVP). The format should guide employees to share their experiences in a way that highlights [insert specific EVP attributes], and it should be versatile enough to be used in blog posts, videos, or social media content.
""",
"Referral Letter": """Compose a referral letter template that employees can use to recommend our company to potential candidates. The letter should include an overview of our Employee Value Proposition (EVP), particularly focusing on [insert key EVP points], and explain why the company is a great place to work. Include a section for personal anecdotes or reasons the referrer believes the candidate would be a good fit.
""",
"Webinar - Get to Know Us": """Create an outline for a 'Get to Know Us' webinar aimed at potential candidates. The webinar should introduce our company, walk through our Employee Value Proposition (EVP), and include segments that cover [insert specific EVP topics]. Plan for interactive elements like Q&A, polls, and a virtual tour to engage participants.
""",
"Career Fair": """Develop a plan for our presence at a career fair that highlights our Employee Value Proposition (EVP). The plan should include ideas for booth design, marketing materials, and talking points that focus on [insert specific EVP elements]. Ensure that our EVP is clearly communicated in all interactions and materials, with a focus on attracting top talent.
""",
"Skills Assessment Quiz": """Create a skills assessment quiz that can be used to evaluate potential candidates during the pre-hire stage. The quiz should be aligned with the core skills required for our open positions and should subtly incorporate elements of our Employee Value Proposition (EVP), such as questions that reflect our values or culture. Include a scoring guide that helps identify candidates who align with our EVP.
""",
"Interview Process Overview Document": """Develop a document that outlines the entire interview process for candidates, from initial screening to final interviews. This document should clearly communicate what candidates can expect at each stage and how the process aligns with our Employee Value Proposition (EVP). Include sections that highlight our commitment to [insert EVP elements], such as transparency, candidate care, and fair evaluation.
""",
"Case Study Assignment": """Design a case study assignment that candidates can complete as part of the hiring process. The case study should be relevant to the role and include scenarios that reflect our Employee Value Proposition (EVP). Focus on challenges that highlight [insert specific EVP attributes], such as innovation, problem-solving, or teamwork, and provide clear instructions and expectations.
""",
"Candidate Preparation Guide": """Create a preparation guide for candidates to help them get ready for their interviews. The guide should include tips on what to expect, how to prepare, and insights into our company culture and Employee Value Proposition (EVP). Emphasize how [insert EVP elements] are reflected in our interview process, and provide resources that will help candidates feel confident and well-prepared.
""",
"Offer Stage Communication Plan": """Develop a communication plan template for the offer stage that hiring managers can use to ensure consistent and engaging communication with candidates. The plan should outline key touchpoints, messages, and timing, with a focus on reinforcing our Employee Value Proposition (EVP). Include strategies for addressing candidate concerns and emphasizing [insert EVP points] as reasons to accept the offer.
""",
"Candidate Journey Map": """Design a candidate journey map that visually represents the hiring process from the candidate's perspective. The map should highlight each stage of the process, key interactions, and how our Employee Value Proposition (EVP) is communicated at each step. Use the journey map to identify areas where we can enhance the candidate experience and better align with [insert EVP elements].
""",
"Hiring Manager Toolkit": """Create a comprehensive toolkit for hiring managers to use during the hiring process. The toolkit should include resources, templates, and best practices that align with our Employee Value Proposition (EVP). Focus on helping hiring managers effectively communicate [insert specific EVP attributes] to candidates, conduct interviews, evaluate candidates, and make decisions that support our overall talent strategy.
""",
"Job Description": """Develop a detailed job description template that is tailored to attract high-quality candidates. The job description should include a compelling overview of the role, key responsibilities, required qualifications, and benefits. Most importantly, it should prominently feature our Employee Value Proposition (EVP) to convey what makes our company unique as an employer. Include specific EVP elements such as [insert key EVP attributes] that align with the role, and ensure the tone and language reflect our company culture and values. The job description should not only inform but also inspire potential candidates to apply.
""",
"Offer Letter": """Craft a personalized offer letter template that communicates our excitement to have the candidate join our team. The letter should include key details about the role and compensation, and it should reinforce our Employee Value Proposition (EVP), particularly focusing on [insert specific EVP elements]. Include a welcoming tone and a section that highlights the next steps in the onboarding process.
""",
"Onboarding Guide": """Create an onboarding guide for new hires that introduces them to our company, culture, and Employee Value Proposition (EVP). The guide should be comprehensive yet easy to navigate, covering essential information such as company policies, team introductions, and an overview of [insert EVP details]. Include a warm welcome message and tips for making the most of their first few weeks.
""",
"Welcome Video": """Write a script for a welcome video that will be shown to new hires on their first day. The video should feature messages from key leaders and team members, and it should emphasize our Employee Value Proposition (EVP). Focus on creating a positive, inclusive atmosphere and include specific EVP attributes that align with the candidate's role.
""",
"Role-Specific Training Modules": """Develop content for role-specific training modules that new hires will complete during their onboarding process. The modules should include practical training relevant to their position, while also incorporating elements of our Employee Value Proposition (EVP), such as [insert specific EVP aspects] that relate to career growth, culture, or company values.
""",
"Team Introduction Deck": """Design a PowerPoint deck for introducing new hires to their immediate team. The deck should include profiles of team members, their roles, and how they contribute to the company. Ensure that the content reflects our Employee Value Proposition (EVP), particularly in areas such as collaboration, innovation, and support. Include an icebreaker section to make the introduction more engaging.
""",
"Company Handbook": """Create a company handbook that provides new hires with a comprehensive overview of our policies, procedures, and culture. The handbook should be aligned with our Employee Value Proposition (EVP) and include sections on [insert specific EVP topics], such as our commitment to diversity, professional development opportunities, and work-life balance. Ensure the tone is welcoming and inclusive.
""",
"First 90 Days Plan": """Develop a 'First 90 Days' plan template that managers can customize for new hires. The plan should include specific goals, key milestones, and regular feedback intervals, all aligned with our Employee Value Proposition (EVP). Include tips for success, resources for support, and a focus on integrating the new hire into the company culture.
""",
"Welcome Kit Content": """Design the content for a welcome kit that new hires receive on their first day. The kit should include branded materials, a personal welcome note, and resources that reflect our Employee Value Proposition (EVP). Focus on creating a memorable and positive first impression that aligns with [insert specific EVP aspects].
""",
"Onboarding Survey": """Develop an onboarding survey to be sent to new hires after their first week or month. The survey should assess their initial experience, how well the onboarding process communicated our Employee Value Proposition (EVP), and areas for improvement. Include questions that capture both quantitative and qualitative feedback.
""",
"Buddy Program Overview": """Create an overview document for a buddy program designed to help new hires acclimate to the company. The document should outline the role of the buddy, key activities to encourage integration, and a check-in schedule. Emphasize how this program aligns with our Employee Value Proposition (EVP), particularly in fostering a supportive and inclusive work environment.
""",
"Onboarding Checklist": """Develop an onboarding checklist that new hires can use to ensure they complete all necessary tasks during their first weeks. The checklist should include key tasks, important dates, and required documents. Align the checklist with our Employee Value Proposition (EVP) by including reminders of cultural integration activities and support resources.
""",
"New Hire Announcement Template": """Design a template for new hire announcements that managers can use to introduce new employees to the broader team. The announcement should include the new hire's name, position, background, and a fun fact. Ensure that the tone of the announcement reflects our Employee Value Proposition (EVP), particularly in creating a welcoming and inclusive atmosphere.
""",
"L&D Success Stories, Career Growth Blogs": """Write a blog post that highlights an employee’s success story and career growth within our company. The post should detail key achievements, career milestones, and how our Learning & Development (L&D) programs contributed to their success. Embed our Employee Value Proposition (EVP) by emphasizing the company’s commitment to employee growth and development.
""",
"L&D Program Brochures": """Create a brochure for our L&D program focused on [insert specific topic]. The brochure should include the program’s objectives, content overview, duration, and key benefits for participants. Ensure that our Employee Value Proposition (EVP) is reflected by highlighting how the program aligns with career growth and continuous learning.
""",
"L&D Pathway Infographics": """Design an infographic that visually represents the learning pathway for [insert specific skill/competency]. The infographic should outline key milestones, the sequence of learning activities, and how the pathway supports career development. Reflect our Employee Value Proposition (EVP) by emphasizing the benefits of continuous learning and progression.
""",
"L&D Program Invitations, Training Notifications": """Create an invitation or notification for an upcoming L&D program or training session. The communication should include event details, the target audience, and the key benefits of attending. Embed our Employee Value Proposition (EVP) by highlighting how the program or training session supports employee growth and aligns with our commitment to continuous development.
""",
"L&D Program Videos, Mentorship Intro Videos": """Develop a video for our L&D program or an introduction to our mentorship program. The video should include key messages about the program’s benefits, feature employees or mentors, and maintain a visually engaging style. Reflect our Employee Value Proposition (EVP) by emphasizing how the program supports career development and fosters a learning culture.
""",
"Training Effectiveness Surveys": """Create a survey to assess the effectiveness of a recent training session. The survey should include questions about the quality of content, delivery, and overall experience. Ensure that the survey reflects our Employee Value Proposition (EVP) by asking how well the training met their development needs and aligned with their career goals.
""",
"Leadership Development Guides": """Create a guide for our leadership development program that focuses on [insert specific leadership skills]. The guide should include the program’s objectives, target audience, key competencies to develop, and preferred delivery method. Ensure that the guide aligns with our Employee Value Proposition (EVP), particularly in fostering leadership, innovation, and strategic thinking.
""",
"Recognition Posts, L&D Highlights": """Draft a recognition post or highlight that celebrates employee achievements or the success of an L&D program. The content should include details of the achievement or program, key takeaways, and maintain a tone that reflects our company culture. Embed our Employee Value Proposition (EVP) by emphasizing how these accomplishments align with our commitment to employee development and recognition.
""",
"L&D Program Newsletters": """Create a newsletter focused on Learning & Development (L&D) updates. The newsletter should highlight key topics, feature upcoming programs and events, and share success stories. Ensure the content reflects our Employee Value Proposition (EVP), particularly in promoting a culture of continuous learning and career development.
""",
"L&D Webinars, Leadership Development Workshops": """Create an invitation or content outline for an upcoming L&D webinar or leadership development workshop. The content should include event details, key speakers, and the benefits of attending. Incorporate our Employee Value Proposition (EVP) by highlighting how participating in the event will contribute to the employee’s professional development and align with our company’s values.
""",
"L&D Quizzes, Gamified Training Modules": """Develop a quiz or gamified training module that helps employees learn [insert specific skill/competency]. The content should include key learning objectives, cover important material, and be designed in an engaging, interactive format. Reflect our Employee Value Proposition (EVP) by emphasizing how the activity supports continuous learning and skill development in a fun and motivating way.
""",
"Increment Letter": """Draft an increment letter that communicates an employee’s salary increase. The letter should include the employee’s name, current position, new salary, and the effective date. Embed our Employee Value Proposition (EVP) by emphasizing how the increment reflects the company’s commitment to recognizing and rewarding high performance, and align it with career growth opportunities.
""",
"Promotion Letter": """Create a promotion letter that congratulates an employee on their new role. The letter should detail the employee’s current and new positions, new responsibilities, and any associated salary changes. Reflect our Employee Value Proposition (EVP) by highlighting how the promotion aligns with our commitment to career growth, professional development, and rewarding excellence.
""",
"Periodic Benefits Communication": """Compose a periodic communication that informs employees about the benefits available to them, including any upcoming enrollment periods or changes. The communication should include details about health insurance, retirement plans, and other benefits. Embed our Employee Value Proposition (EVP) by emphasizing how these benefits support employee well-being, financial security, and work-life balance.
""",
"Total Rewards Statement": """Develop a total rewards statement that provides a comprehensive overview of an employee’s compensation package. The statement should include salary details, benefits, bonuses, and any non-monetary rewards. Reflect our Employee Value Proposition (EVP) by showing how the total rewards package aligns with the company’s commitment to holistic employee well-being and recognition.
""",
"Annual Compensation Review Communication": """Create an annual communication that explains the company’s compensation review process and any changes for the year. The communication should provide a summary of company performance, the overall compensation strategy, and how these changes impact employees. Embed our Employee Value Proposition (EVP) by emphasizing the company’s commitment to fair compensation and aligning rewards with business success.
""",
"Benefits Enrollment Guide": """Design a guide for employees to help them navigate the benefits enrollment process. The guide should include detailed descriptions of available benefits, instructions for enrollment, eligibility criteria, and key deadlines. Ensure that the guide reflects our Employee Value Proposition (EVP) by highlighting how the benefits support employee health, financial security, and overall well-being.
""",
"Bonuses and Incentives Announcement": """Compose an announcement regarding bonuses and incentives for employees. The communication should outline the criteria for earning bonuses, relevant performance metrics, the payout timeline, and any tax implications. Reflect our Employee Value Proposition (EVP) by connecting these rewards to the company’s recognition of employee contributions and their alignment with business goals.
""",
"Recognition and Reward Program Communication": """Draft a communication that introduces or updates employees on the company’s recognition and reward program. The content should include program details, eligibility criteria, available reward options, and key dates. Embed our Employee Value Proposition (EVP) by emphasizing how the program aligns with our values of recognizing and celebrating employee achievements and contributions.
""",
"Promotion Criteria and Pathways Communication": """Create a communication that explains the criteria for promotion and the available career pathways within the company. The content should include detailed criteria, examples of potential career paths, and resources available to support employee development. Reflect our Employee Value Proposition (EVP) by highlighting how the company supports career growth and provides clear opportunities for advancement.
""",
"Compensation FAQs Document": """Develop a Frequently Asked Questions (FAQs) document that addresses common employee inquiries related to compensation and benefits. The document should provide clear, concise answers and include additional resources for further assistance. Ensure the document reflects our Employee Value Proposition (EVP) by focusing on transparency, fairness, and the company’s commitment to supporting employee financial well-being.
""",
"Retirement Benefits Overview": """Create an overview document that explains the retirement benefits available to employees. The document should detail the different retirement plan options, contribution details, any company matching programs, and instructions for enrollment. Embed our Employee Value Proposition (EVP) by emphasizing how these benefits support long-term financial security and align with the company’s commitment to employee well-being.
""",
"Health and Wellness Program Communication": """Draft a communication that informs employees about the company’s health and wellness programs. The content should include details about available programs, how employees can participate, the benefits of participation, and the enrollment process. Reflect our Employee Value Proposition (EVP) by highlighting how these programs contribute to employee health, work-life balance, and overall well-being.
""",
"Company-Wide Announcements": """Create a company-wide announcement that communicates [insert key message]. The announcement should clearly convey the important information, specify the target audience, and include any relevant resources or links. Embed our Employee Value Proposition (EVP) by highlighting how the announcement aligns with our company’s values and impacts employees.
""",
"Quarterly Newsletters": """Draft a quarterly newsletter that provides updates on company achievements, upcoming events, and other important information. The newsletter should also include employee spotlights and key dates to remember. Reflect our Employee Value Proposition (EVP) by emphasizing our commitment to transparency, employee recognition, and fostering a connected workplace.
""",
"Leadership Communication Emails": """Compose an email from company leadership that communicates [insert key message]. The email should provide context or background for the message, and outline any next steps or action items for employees. Embed our Employee Value Proposition (EVP) by aligning the message with our company’s vision, goals, and commitment to employee engagement.
""",
"Crisis Communication Plan": """Develop a crisis communication plan that outlines how to manage and communicate during [insert specific crisis scenario]. The plan should include details on key stakeholders, communication channels, and response steps. Ensure the plan reflects our Employee Value Proposition (EVP) by emphasizing the company’s commitment to transparency, employee safety, and support during challenging times.
""",
"Internal Survey Invitations": """Create an invitation for employees to participate in [insert specific internal survey]. The invitation should explain the purpose of the survey, key areas of focus, and include a link to the survey. Embed our Employee Value Proposition (EVP) by highlighting the importance of employee feedback and how it will be used to improve the workplace.
""",
"Employee Town Hall Invitations and Agendas": """Draft an invitation and agenda for an upcoming employee town hall. The invitation should include the town hall’s objectives, key topics to be covered, featured speakers, and details about the Q&A session. Reflect our Employee Value Proposition (EVP) by emphasizing the company’s commitment to open communication, transparency, and employee engagement.
""",
"Policy Update Communications": """Compose a communication that informs employees about [insert specific policy change]. The communication should detail the changes, the effective date, and how it impacts employees. Ensure that the message reflects our Employee Value Proposition (EVP) by aligning the policy update with our commitment to fairness, compliance, and employee support.
""",
"Employee Recognition Communications": """Draft a communication that recognizes an employee or team for their achievements. The message should detail the achievements, include a personalized recognition message, and outline any next steps (e.g., a celebratory event). Embed our Employee Value Proposition (EVP) by highlighting how the recognition aligns with our values of excellence, teamwork, and employee appreciation.
""",
"Diversity and Inclusion (D&I) Updates": """Create an update on the company’s Diversity and Inclusion (D&I) efforts. The communication should highlight recent initiatives, upcoming events, key achievements, and opportunities for employee participation. Reflect our Employee Value Proposition (EVP) by emphasizing the company’s commitment to fostering an inclusive, diverse, and supportive work environment.
""",
"Employee Feedback Response Communications": """Compose a communication that responds to employee feedback collected through [insert specific feedback mechanism]. The message should summarize the feedback, outline the actions taken or planned, and provide a timeline for implementation. Ensure the communication aligns with our Employee Value Proposition (EVP) by showing how employee voices are valued and acted upon.
""",
"Internal Campaign Announcements": """Develop an announcement for an internal campaign focused on [insert specific objective, e.g., health and wellness, sustainability]. The announcement should include the campaign’s objectives, key messages, timeline, and details on how employees can participate. Embed our Employee Value Proposition (EVP) by aligning the campaign with our company’s values and employee engagement goals.
""",
"Internal Event Invitations (e.g., Team Building, Celebrations)": """Create an invitation for an internal event such as a team-building activity or celebration. The invitation should include event details, the purpose of the event, and any special notes (e.g., attire, RSVP instructions). Reflect our Employee Value Proposition (EVP) by emphasizing how the event supports a positive workplace culture, employee connection, and company values.
""",
"Engagement Tips, Recognition Program Highlights": """Create a document that shares practical engagement tips and highlights key aspects of our recognition program. Include success stories that demonstrate the program’s impact. Reflect our Employee Value Proposition (EVP) by emphasizing how these strategies align with our commitment to fostering a positive and motivating workplace.
""",
"Wellness Program Brochures": """Design a brochure that outlines the wellness programs available to employees. The brochure should detail the benefits of each program, how to participate, and the positive impact on well-being. Embed our Employee Value Proposition (EVP) by highlighting how these programs support employee health, work-life balance, and overall satisfaction.
""",
"Recognition Program Infographics": """Develop an infographic that visually communicates the key elements of our recognition program. Include statistics or metrics that demonstrate the program’s success. Ensure the infographic reflects our Employee Value Proposition (EVP) by showcasing how recognition aligns with our values and supports a culture of appreciation.
""",
"Recognition Emails, Team Event Invites": """Draft a recognition email or team event invitation that highlights [insert achievement or event details]. The email should include personalized elements to make the recipient feel valued. Reflect our Employee Value Proposition (EVP) by emphasizing how the recognition or event aligns with our company’s commitment to celebrating success and fostering team spirit.
""",
"Employee Spotlight Videos, Team Event Highlights": """Create a video that spotlights an employee’s achievements or highlights a recent team event. The video should feature key moments or contributions and be engaging in both style and content. Embed our Employee Value Proposition (EVP) by emphasizing how these stories align with our values of excellence, teamwork, and community.
""",
"Pulse Surveys, Recognition Feedback": """Design a pulse survey or feedback form that gathers employee input on [insert specific focus area, e.g., engagement, recognition]. The survey should include targeted questions that align with our Employee Value Proposition (EVP) by focusing on how well employees feel recognized and supported within the company.
""",
"Mentorship Guides, Team-Building Guides": """Create a guide for our mentorship or team-building program that outlines the program’s objectives, key activities, and criteria for participation. The guide should include practical tips and resources. Ensure that the guide reflects our Employee Value Proposition (EVP) by highlighting how the program supports professional development, collaboration, and team cohesion.
""",
"Campaign Posts, Wellness Program Highlights": """Develop a series of campaign posts or highlights that promote our wellness programs. The content should focus on key messages, include engaging visuals, and encourage participation. Embed our Employee Value Proposition (EVP) by aligning the campaign with our commitment to employee well-being, health, and a balanced lifestyle.
""",
"Engagement Newsletters, Recognition Program Updates": """Create a newsletter that updates employees on engagement initiatives and recognition program activities. Include details about upcoming events and spotlight employees who have been recognized. Reflect our Employee Value Proposition (EVP) by emphasizing how these efforts contribute to a supportive and motivated workplace.
""",
"Team-Building Events, Engagement Webinars": """Draft an invitation or outline for a team-building event or engagement webinar. The content should include event details, objectives, and key activities that will take place. Reflect our Employee Value Proposition (EVP) by highlighting how the event supports team cohesion, learning, and a positive work environment.
""",
"Interactive Employee Recognition Platforms": """Design a communication plan to introduce an interactive employee recognition platform. The plan should include details about the platform’s features, how employees can participate, and the timeline for launch. Ensure the communication reflects our Employee Value Proposition (EVP) by emphasizing how the platform will make recognition more accessible and aligned with our values of appreciation and excellence.
""",
"Exit Announcement (Internal)": """Draft an internal exit announcement to inform the team about [insert employee’s name] departure. The announcement should include the employee’s role, reason for departure (if applicable), last working day, and a message of appreciation for their contributions. Reflect our Employee Value Proposition (EVP) by emphasizing our gratitude and maintaining a respectful tone.
""",
"Exit Interview Questionnaire": """Create an exit interview questionnaire designed to gather feedback from departing employees. The questionnaire should focus on understanding their reasons for leaving, overall satisfaction, and suggestions for improvement. Ensure the questionnaire reflects our Employee Value Proposition (EVP) by showing a genuine interest in learning from their experience and improving the work environment.
""",
"Separation Checklist": """Develop a separation checklist that outlines the tasks to be completed before an employee’s departure. The checklist should include items such as returning company property, completing final payroll, and any other necessary steps. Reflect our Employee Value Proposition (EVP) by ensuring the process is smooth, organized, and respectful of the departing employee.
""",
"Final Pay and Benefits Communication": """Compose a communication that explains the final pay and benefits situation for departing employees. The communication should include details on final pay, any continuation or termination of benefits, and who to contact with questions. Embed our Employee Value Proposition (EVP) by ensuring clarity, fairness, and support during this transition.
""",
"Thank You and Farewell Letter": """Draft a personalized thank you and farewell letter to be given to the departing employee. The letter should express gratitude for their contributions, highlight positive memories, and offer encouragement for their future endeavors. Reflect our Employee Value Proposition (EVP) by showing appreciation and fostering a positive relationship, even as they leave the company.
""",
"Knowledge Transfer Document": """Create a knowledge transfer document that the departing employee can use to pass on critical information to their successor or team. The document should include details on key projects, responsibilities, and handover instructions. Ensure the process reflects our Employee Value Proposition (EVP) by emphasizing the importance of a smooth transition and continued team success.
""",
"Exit Process Overview (For Managers)": """Develop an overview document for managers to guide them through the exit process. The document should outline the steps to be followed, key responsibilities, and communication guidelines to ensure the process is handled professionally and empathetically. Reflect our Employee Value Proposition (EVP) by emphasizing a respectful and supportive approach to employee departures.
""",
"Alumni Network Invitation": """Draft an invitation for departing employees to join the company’s alumni network. The communication should include details about the network, benefits of joining, and instructions for signing up. Embed our Employee Value Proposition (EVP) by highlighting the value of staying connected and the continued support the company offers even after departure.
""",
"References and Recommendations Letter": """Create a reference or recommendation letter for a departing employee. The letter should highlight the employee’s role, key achievements, and include contact details for verification. Reflect our Employee Value Proposition (EVP) by ensuring the letter is positive, supportive, and aligned with our commitment to recognizing employee contributions.
""",
"Exit Survey Invitation": """Compose an invitation for the departing employee to complete an exit survey. The invitation should explain the purpose of the survey, provide a link, and assure the employee of confidentiality. Ensure the communication aligns with our Employee Value Proposition (EVP) by emphasizing the value of their feedback and our commitment to improving the employee experience.
""",
"Transition Communication Plan": """Develop a communication plan for informing key stakeholders about an employee’s departure and the transition process. The plan should include a timeline, key messages, and guidelines for ensuring a smooth transition. Reflect our Employee Value Proposition (EVP) by focusing on clear, respectful, and timely communication throughout the transition.
""",
"Exit Package Overview": """Create an overview document that explains the exit package being offered to the departing employee. The document should detail the components of the package, payment schedule, and any legal considerations. Embed our Employee Value Proposition (EVP) by ensuring the package is presented transparently, fairly, and with respect for the employee’s contributions to the company.
""",
"Alumni Newsletter": """Create a newsletter specifically for alumni that includes key updates about the company, highlights alumni achievements, and provides information about upcoming events. Embed our Employee Value Proposition (EVP) by showcasing the ongoing connection between the company and its alumni, and by emphasizing opportunities for continued engagement.
""",
"Alumni Portal Welcome Message": """Draft a welcome message for the company’s alumni portal. The message should introduce the portal’s features, highlight the key benefits of joining, and provide clear instructions for registration. Reflect our Employee Value Proposition (EVP) by emphasizing the value of staying connected and the resources available to alumni through the portal.
""",
"Alumni Event Invitations": """Compose an invitation for an upcoming alumni event. The invitation should include all necessary event details, explain the purpose of the event, and provide RSVP instructions. Ensure that the communication reflects our Employee Value Proposition (EVP) by emphasizing how the event supports networking, learning, and continued engagement with the company.
""",
"Alumni Success Stories": """Create a series of success stories that feature alumni who have achieved significant milestones since leaving the company. The stories should highlight their achievements and how their time at the company contributed to their success. Embed our Employee Value Proposition (EVP) by showing the lasting impact of the company’s culture and development opportunities.
""",
"Alumni Networking Opportunities": """Draft a communication that informs alumni about upcoming networking opportunities. The message should include details about the event, key participants, and the purpose of the networking session. Reflect our Employee Value Proposition (EVP) by emphasizing the benefits of staying connected and how these opportunities can contribute to ongoing professional growth.
""",
"Alumni Feedback Surveys": """Create a feedback survey for alumni to gather their insights on [insert specific topic, e.g., alumni engagement, event satisfaction]. The survey should include targeted questions, a link to participate, and an assurance of confidentiality. Ensure that the survey aligns with our Employee Value Proposition (EVP) by emphasizing the importance of alumni feedback in shaping future initiatives.
""",
"Alumni Social Media Content": """Develop a series of social media posts aimed at engaging alumni. The content should include key messages, engaging visuals, and a clear call-to-action for alumni to interact or participate in upcoming activities. Reflect our Employee Value Proposition (EVP) by showcasing how the company values its alumni and encourages continued connection and involvement.
""",
"Re-engagement Campaigns": """Create a re-engagement campaign aimed at alumni who have not been active recently. The campaign should focus on key messages that encourage them to reconnect with the company, and include any incentives for re-engagement. Embed our Employee Value Proposition (EVP) by emphasizing the value of the alumni network and the mutual benefits of staying involved.
""",
"Alumni Recognition Programs": """Develop a recognition program that celebrates the achievements of alumni. The program should include details about how alumni can be recognized, eligibility criteria, and the methods of recognition (e.g., awards, features in newsletters). Reflect our Employee Value Proposition (EVP) by emphasizing the company’s ongoing commitment to celebrating success and maintaining strong ties with its former employees.
""",
"Alumni Mentorship Program": """Create a communication plan for an alumni mentorship program that connects former employees with current staff or other alumni. The plan should outline the program’s objectives, criteria for matching mentors and mentees, and the benefits of participation. Ensure that the program reflects our Employee Value Proposition (EVP) by fostering continuous learning and professional development across the alumni network.
""",
"Career Opportunities Updates for Alumni": """Compose a periodic update that informs alumni about career opportunities within the company. The update should highlight available job openings, key qualifications needed, and application instructions. Reflect our Employee Value Proposition (EVP) by emphasizing any alumni-specific benefits or advantages in rejoining the company, and how these opportunities align with their career growth.
""",
"Alumni Referral Program": """Draft a communication about the alumni referral program. The message should detail how the program works, the incentives for successful referrals, eligibility criteria, and instructions for submitting referrals. Embed our Employee Value Proposition (EVP) by emphasizing the value of alumni contributions in bringing in top talent and maintaining a strong connection to the company’s success.
""",
}

def get_evp_embedment_data_from_chatgpt(company_name, user, stage, touchpoint, evp_statement_themes, tagline_data, evp_promise_data, evp_audit_data):
    company = Company.objects.get(name=company_name)
    company_id = company.id

    RESPONSE_JSON = {
        "touchpoint_data": {
            "stage": stage,
            "touchpoint": touchpoint,
            "message": all_touchpoint_prompts.get(touchpoint, "")
        }
    }

    prompt = f"""
                First analyze the Themes Data
                Themes Data: {evp_statement_themes}

                Now analyze the Tagline Data
                Tagline Data : {tagline_data}

                Now analyze the EVP Promise Data
                EVP Promise Data : {evp_promise_data}

                Now analyze the EVP Audit Data
                EVP Audit Data : {evp_audit_data}

                Using the above given data, Fetch the data for the value of "message" key and returns the response in json format

                Make sure to format the response exactly like {RESPONSE_JSON} and use it as a guide.
                Just replace the value of key "message" with the actual data of the query and let other fields as it is.
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
        json_response = json_response["touchpoint_data"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        json_response = {}

    return json_response

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