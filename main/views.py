import os
import re
import shutil
from dotenv import load_dotenv
load_dotenv()

from django.contrib.auth import login

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from rest_framework.permissions import IsAuthenticated

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import *

from .serializers import *

from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import chromadb
from openai import AzureOpenAI

AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_TYPE = os.environ["AZURE_OPENAI_TYPE"]
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]
AZURE_EMBEDDING_MODEL = os.environ["AZURE_EMBEDDING_MODEL"]

from .utils.bing_search import bing_query_data, get_data_from_bing, save_data_to_database
from .utils.chatgpt import chatgpt_1_query_data, get_data_from_chatgpt_1
from .utils.handle_documents import save_documents, merge_documents
from .utils.langchaining import (
    save_documents_to_master_vector_database,
    get_talent_dataset_from_chatgpt,
    testing_data,
    create_embeddings,
    query_with_langchain,
    save_pgData_to_vector_database,
    get_develop_data_from_vector_database,
    get_talent_insights_from_chatgpt,
    get_dissect_data_from_vector_database,
    get_design_data_from_database, get_tagline,
    get_regenerated_theme,
    get_creative_direction_from_chatgpt,
    get_evp_definition_from_chatgpt,
    get_evp_promise_from_chatgpt,
    get_evp_audit_from_chatgpt,
    get_evp_embedment_data_from_chatgpt,
    get_evp_handbook_data_from_chatgpt,
)
from .utils.email_send import send_email_to_users

chat_client = AzureOpenAI(
    azure_endpoint = AZURE_ENDPOINT, 
    api_key=AZURE_OPENAI_KEY,  
    api_version=AZURE_OPENAI_API_VERSION
)

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["email"] = user.email
        return token
    
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)

    return {
        "refresh": str(refresh),
        "access": str(refresh.access_token),
        "email": str(user.email),
        "role": str(user.role),
    }

class LoginAPIView(APIView):
    def post(self, request):
        data = request.data
        serializer = UserLoginSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.validated_data['user']
            login(request, user)
            tokens = get_tokens_for_user(user)
            return Response({"tokens": tokens}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class IsAdmin(permissions.BasePermission):
    def has_permission(self, request, view):
        user = request.user
        return user.role == "Admin"


class Testing(APIView):
    def post(self, request):
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({'error': 'company_name parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        if "documents" in request.FILES:
            uploaded_documents = request.FILES.getlist("documents")
            all_documents = []
            for document in uploaded_documents:
                response = save_documents(document, "documents")
                all_documents.append(response)
            merge_documents("media\documents", "final_pdf", "merged_pdf.pdf")
            loader = PyPDFLoader(r"media\final_pdf\merged_pdf.pdf")
            document_data = loader.load()
    
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(document_data)
            documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
            ids=[f"id{i}" for i in range(len(documents))]

        embeddings = create_embeddings()

        embedded_documents = embeddings([documents[i] for i in range(len(documents))])

        chroma_client = chromadb.PersistentClient(path="vector_databases/Testing")
        collection = chroma_client.get_or_create_collection(
            name="test",
            embedding_function=embeddings,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            embeddings=embedded_documents,
            documents=documents,
            ids=ids,
        )

        # for doc, id in zip(documents, ids):
        #     print(f"Document: {doc[:100]} - ID: {id}")

        # mismatches = []
        # for i, (original_doc, more_doc) in enumerate(zip(documents, more_documents)):
        #     print("Original Document: ", original_doc[:100])  # Print first 100 chars for brevity
        #     print("Retrieved Document: ", more_doc[:100])    # Print first 100 chars for brevity
        #     if original_doc != more_doc:
        #         mismatches.append(i)
        #     else:
        #         print(f"Document stored correctly at index {i}: {original_doc[:100]}")

        # if mismatches:
        #     return Response({'error': f'Mismatches found at indices: {mismatches}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        # else:
        #     return Response("Documents stored correctly")

        get_data = testing_data(collection)

        print(get_data)
        return Response(get_data)
    
# *****************************Admin Space**********************************
class MasterVectorDatabaseAPIView(APIView):
    permission_classes = [IsAuthenticated, IsAdmin]

    def post(self, request):
        user = request.user

        if "documents" in request.FILES:
            if os.path.exists(r"media\admin_documents"):
                for filename in os.listdir(r"media\admin_documents"):
                    file_path = os.path.join(r"media\admin_documents", filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            uploaded_documents = request.FILES.getlist("documents")
            all_documents = []
            for document in uploaded_documents:
                response = save_documents(document, "admin_documents")
                all_documents.append(response)
            merge_documents("media/admin_documents", "admin_merged_pdf", "merged_pdf.pdf")
        else:
            uploaded_documents = None

        if uploaded_documents:
            master_database_response = save_documents_to_master_vector_database()
            return Response(master_database_response, status=status.HTTP_200_OK)
        
        return Response({"error": "Please upload at least one document"}, status=status.HTTP_400_BAD_REQUEST)
    
class homePageAPIView(APIView):
    def post(self, request):
        company_name = request.data.get("company_name")

        RESPONSE_JSON = {
            "facts": {
                "fact1": "response",
                "fact2": "response",
            }
        }

        prompt = f"""
                    I want to find specific facts about the company named {company_name}.

                    So, find the facts about the company above and returns the response in json format:

                    Come up with 9 interesting facts about the company.
                    Showcase a mix of talent and employees-centric facts as well as business and strategy-centric facts.
                    Each fact should be no more than 15 to 20 words.

                    Make sure to format your response exactly like {RESPONSE_JSON} and use it as a guide.
                    Replace response with the actual fact and every response should contain the company name.
                """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages = [
            {
                "role":"system",
                "content":"You are an expert research analyst"
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
            json_response = json_response["facts"]
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            json_response = {}
        return Response(json_response, status=status.HTTP_200_OK)

class TalentInsightsHomeAPIView(APIView):
    def post(self, request):
        
        talent_insights = request.data.get("talent_insights")
        skill = talent_insights["skill"]
        sub_skill = talent_insights["sub_skill"]
        role = talent_insights["role"]
        geography = talent_insights["geography"]

        RESPONSE_JSON = {
            "name": "Name of the person",
            "age": "Age of the person",
            "location": "location",
            "highest_qualification": "Qualification in not more than 20 words",
            "work_experience": "Work experience in not more then 20 words",
            "previous_companies": "Previous companies in not more than 20 words",
            "salary_inr": "Salary package",
            "personality": "Use the Briggs Myer Personality Index to determine what personality this typical candidate persona would fall under and create it in 100 words",
            "goals": "Goals in 3 points and points in numbers",
            "frustration": "Frustration in 3 points and points in numbers",
            "bio": "Bio description within 50 words",
            "motivation": "3 key motivators and points in numbers",
            "topics_of_interest": "Create within 3 points and points in numbers",
            "preferred_channels": "3 preferred channels and points in numbers",
        }

        prompt = f"""
                    Create a candidate persona according to the below query and returns response in json format:

                    Selected Skill : {skill}
                    Selected Sub skill : {sub_skill}
                    Selected Role : {role}
                    Selected Geography : {geography}

                    Create a candidate persona for the selected Skill, Sub Skill, Role and geography.
                    It will contain Name, Age, Location, Highest Qualification, Work Experience, Previous Companies, Salary INR, Personality, Goals, Frustration, Bio, Motivation, Topics of Interest and Preferred Channels.

                    Make sure to format your response exactly like {RESPONSE_JSON} and use it as a guide.

                    The keys will remain same and the value will be replaced with the actual information.

                    And do not create set or anything for points, I just want a string with line gap after each point
                """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages = [
            {
                "role":"system",
                "content":"You are an expert research analyst"
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
            print(json_response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            json_response = {}
        return Response(
            json_response,
            status=status.HTTP_200_OK
        )
    
class IndustryTrendsHomeAPIView(APIView):
    def post(self, request):
        
        industry_trends = request.data.get("industry_trends")
        industry = industry_trends["industry"]
        sub_industry = industry_trends["sub_industry"]

        prompt = f"""
                    I want to find the trends according to below query:

                    Selected Industry: {industry}
                    Selected Sub Industry: {sub_industry}

                    Source the latest trends in the selected Sub Industry.
                    Focus on market forces, latest news, technologies, geo-political forces, demographic inputs.
                    Then create a section on possibilities and implications for the talent market from a hiring and retention standpoint.

                """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role":"system",
                "content":"You are an expert research analyst"
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
        return Response(
            {"latest_trends": chat_response},
            status=status.HTTP_200_OK
        )

class SearchWebsiteView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({'error': 'company_name parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        if "documents" in request.FILES:
            uploaded_documents = request.FILES.getlist("documents")
            all_documents = []
            for document in uploaded_documents:
                response = save_documents(document, "documents")
                all_documents.append(response)
            merge_documents("media\documents", "final_pdf", "merged_pdf.pdf")
        else:
            uploaded_documents = None

        final_data = {}
        if uploaded_documents:
            print("In if block")
            data_from_langchain = query_with_langchain(company_name)
            # return Response(data_from_langchain)

            data_with_values_from_langchain = {field: value for field, value in data_from_langchain.items() if not re.search(r'not\s*found', value, re.IGNORECASE)}

            final_data.update(data_with_values_from_langchain)

            empty_fields_from_langchain = [field for field, value in data_from_langchain.items() if re.search(r'not\s*found', value, re.IGNORECASE)]

            fields_to_query_with_bing = [field for field in bing_query_data if field in empty_fields_from_langchain]
            # # fields_to_query_with_bing = bing_query_data

            data_from_bing = get_data_from_bing(company_name, fields_to_query_with_bing)

            data_with_values_from_bing = {field: value for field, value in data_from_bing.items() if not re.search(r'not\s*found', value, re.IGNORECASE)}

            final_data.update(data_with_values_from_bing)

            # fields_to_query_with_chatgpt_1 = {field: "" for field in chatgpt_1_query_data if field in empty_fields_from_langchain}

            # if (len(fields_to_query_with_chatgpt_1) > 0):
            #     data_from_chatgpt_1 = get_data_from_chatgpt_1(company_name, fields_to_query_with_chatgpt_1)
            #     final_data.update(data_from_chatgpt_1)

        else:
            print("In else block")
            data_from_bing = get_data_from_bing(company_name, bing_query_data)

            final_data.update(data_from_bing)

            # data_from_chatgpt_1 = get_data_from_chatgpt_1(company_name, chatgpt_1_query_data)

            # final_data.update(data_from_chatgpt_1)

        saved_data_in_database_in_string = save_data_to_database(final_data, company_name, user)
        
        # with open(r"media\pgData.txt", "w") as file:
        #     file.write(saved_data_in_database_in_string)
        # save_pgData_to_vector_database(r"media\pgData.txt", company_name)
        # print("Success")

        # return Response(data_from_langchain)
        return Response(final_data)

class DevelopAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({'error': 'company_name parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
        company_id = company.id

        if (AttributesOfGreatPlace.objects.filter(user=user, company=company_id).exists() and 
                KeyThemes.objects.filter(user=user, company=company_id).exists() and 
                AudienceWiseMessaging.objects.filter(company=company_id).exists()):
            return Response("Instance already created", status=status.HTTP_200_OK)
        
        try:
            develop_data_from_vector_database_in_string = get_develop_data_from_vector_database(company_name, user)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # return Response(develop_data_from_vector_database_in_string, status=status.HTTP_200_OK)
        return Response("develop_data_from_vector_database_in_string", status=status.HTTP_200_OK)
    
class DissectAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get('company_name')
        if not company_name:
            return Response({'error': 'company_name parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
        company_id = company.id

        if (SwotAnalysis.objects.filter(user=user, company=company_id).exists() and 
                Alignment.objects.filter(user=user, company=company_id).exists()):
            return Response("Instance already created", status=status.HTTP_200_OK)
        
        try:
            design_principles_instance = DesignPrinciples.objects.get(user=user, company=company)
            design_principles = design_principles_instance.design_principles
        except DesignPrinciples.DoesNotExist:
            return Response({'error': 'Design Principles not found'}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            dissect_data_from_vector_database = get_dissect_data_from_vector_database(company_name, user, design_principles)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response("dissect_data_from_vector_database", status=status.HTTP_200_OK)
    
class DesignAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        if not company_name:
            return Response({"error": "company_name parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)

        if MessagingHierarchyTabs.objects.filter(user=user, company=company).exists():
            messaging_hierarchy_tabs = MessagingHierarchyTabs.objects.filter(user=user, company=company)
            serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        themes_data = request.data.get("themes_data")
        if themes_data:
            for data in themes_data:
                MessagingHierarchyTabs.objects.create(
                    company=company,
                    user=user,
                    tab_name=data["tab_name"],
                    tabs_data=data["tabs_data"]
                )
            messaging_hierarchy_tabs = MessagingHierarchyTabs.objects.filter(user=user, company=company)
            serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        try:
            design_data_from_vector_database = get_design_data_from_database(company_name, user)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(design_data_from_vector_database, status=status.HTTP_200_OK)
    
class Top4ThemesRegenerateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        if not company_name:
            return Response({"error": "company_name parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)

        # if MessagingHierarchyTabs.objects.filter(user=user, company=company).exists():
        #     messaging_hierarchy_tabs = MessagingHierarchyTabs.objects.filter(user=user, company=company)
        #     serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs, many=True)
        #     return Response(serializer.data, status=status.HTTP_200_OK)

        theme_to_regenerate = request.data.get("theme_to_regenerate")
        print(theme_to_regenerate)
        regenerate_theme = get_regenerated_theme(company_name, user, theme_to_regenerate)

        return Response(
            regenerate_theme,
            status=status.HTTP_200_OK
        )

class ChatBotAPIView(APIView):
    permission_classes = [IsAuthenticated]

    embeddings = create_embeddings()
    def post(self, request):
        
        if "documents" in request.FILES:
            if os.path.exists("media\documents_chatbot"):
                for filename in os.listdir("media\documents_chatbot"):
                    file_path = os.path.join("media\documents_chatbot", filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            uploaded_documents = request.FILES.getlist("documents")
            all_documents = []
            for document in uploaded_documents:
                response = save_documents(document, "documents_chatbot")
                all_documents.append(response)
            merge_documents("media\documents_chatbot", "final_pdf_chatbot", "merged_pdf.pdf")
            loader = PyPDFLoader(r"media\final_pdf_chatbot\merged_pdf.pdf")
            document_data = loader.load()
    
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
            text_chunks = text_splitter.split_documents(document_data)
            documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
            ids=[f"id{i}" for i in range(len(documents))]
            embedded_documents = self.embeddings([documents[i] for i in range(len(documents))])
        else:
            text_chunks = []
       
        user_query = request.data["user_query"]
        company_name = request.data["company_name"]
 
        sanitized_company_name = re.sub(r'\s+', '_', company_name)

        persistent_directory = f"vector_databases/{sanitized_company_name}"
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        if os.path.exists(os.path.join(persistent_directory)):
            if "test_chatbot" in [collection.name for collection in chroma_client.list_collections()]:
                chroma_client.delete_collection(name="test_chatbot")

            chatbot_collection = chroma_client.create_collection(
                name="test_chatbot",
                embedding_function=self.embeddings,
            )

            if text_chunks:
                chatbot_collection.add(
                    embeddings=embedded_documents,
                    documents=documents,
                    ids=ids,
                )

            query_results = chatbot_collection.query(
                    query_texts=[user_query],
                    n_results=10,
                )
            fetched_documents = " ".join(query_results["documents"][0])

            prompt = f"""
            Information: {fetched_documents} \n \n Question: {user_query}.
            """

            chat_client = AzureOpenAI(
                azure_endpoint = AZURE_ENDPOINT, 
                api_key=AZURE_OPENAI_KEY,  
                api_version=AZURE_OPENAI_API_VERSION
            )

            completion = chat_client.chat.completions.create(
            model="VEGPT35",
            messages = [
                {
                    "role":"system",
                    "content":"""You are a helpful expert research assistant. Your users are asking questions about information contained in the given data.
                                 You will be shown the user's question, and the relevant information from the data.
                                 After analyzing the complete information, your task is to answer the user's question using only this information.
                                 IF YOU DON'T FIND THE ANSWER IN THE GIVEN INFORMATION PLEASE SAY -- "I don't know".
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
        return Response(chat_response)

class SendMailAPIView(APIView):
    def post(self, request):
        input_emails = request.data.get("emails")
        if not input_emails:
            return Response("Invalid email address provided", status=400)

        try:
            send_email_to_users(input_emails)
            return Response("Email sent successfully")
        except Exception as e:
            return Response(f"Failed to send email: {str(e)}", status=500)
        
class TranscriptAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, company_name):

        if "documents" in request.FILES:
            if os.path.exists("media\documents"):
                for filename in os.listdir("media\documents"):
                    file_path = os.path.join("media\documents", filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            uploaded_documents = request.FILES.getlist("documents")
            all_documents = []
            for document in uploaded_documents:
                response = save_documents(document, "documents")
                all_documents.append(response)
            merge_documents("media\documents", "final_pdf", "merged_pdf.pdf")
            loader = PyPDFLoader(r"media\final_pdf\merged_pdf.pdf")
            document_data = loader.load()
    
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(document_data)
            documents = [text_chunks[i].page_content for i in range(len(text_chunks))]
            embeddings = create_embeddings()
        else:
            text_chunks = []
 
        sanitized_company_name = re.sub(r'\s+', '_', company_name)

        persistent_directory = f"vector_databases/{sanitized_company_name}"
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        if os.path.exists(os.path.join(persistent_directory)):
            if text_chunks:
                collection = chroma_client.get_or_create_collection(
                    name="test",
                    embedding_function=embeddings,
                )

                current_count = collection.count()
                ids = [f"id{current_count + i}" for i in range(len(documents))]
                embedded_documents = embeddings([documents[i] for i in range(len(documents))])

                collection.add(
                    embeddings=embedded_documents,
                    documents=documents,
                    ids=ids,
                )
                return Response("Transcript added successfully", status=status.HTTP_201_CREATED)
        return Response("Please upload at least one document", status=status.HTTP_400_BAD_REQUEST)

class DesignPrinciplesAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"message": "Company does not exist"}, status=status.HTTP_404_NOT_FOUND)

        design_principles_list = request.data.get("design_principles")
        design_principles_string = "\n".join([f"{i + 1}. {item}" for i, item in enumerate(design_principles_list)])

        existing_design_principles = DesignPrinciples.objects.filter(user=user, company=company).first()

        if existing_design_principles:
            return Response(
                "Design Principles already exist",
                status=status.HTTP_200_OK
            )

        DesignPrinciples.objects.create(
            user=user,
            company=company,
            design_principles = design_principles_string,
        )

        return Response(
            "Design Principles saved successfully",
            status=status.HTTP_201_CREATED
        )
    
class CompanySpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"message": "Company does not exist"}, status=status.HTTP_404_NOT_FOUND)
        serializer = CompanySerializer(company)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = CompanySerializer(company, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class TalentDatasetAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        if TalentDataset.objects.filter(user=user, company=company).exists():
            talent_datasets = TalentDataset.objects.filter(user=user, company=company)
            serializer = TalentDatasetSerializer(talent_datasets, many=True)
            return Response(serializer.data)
        
        talent_dataset_from_chatgpt = get_talent_dataset_from_chatgpt(company_name, user)
        return Response(talent_dataset_from_chatgpt, status=status.HTTP_200_OK)
    
class PerceptionSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            perception = Perception.objects.get(user=user, company=company)
        except Perception.DoesNotExist:
            return Response({'error': 'Perception does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = PerceptionSerializer(perception)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            perception = Perception.objects.get(user=user, company=company)
        except Perception.DoesNotExist:
            return Response({'error': 'Perception does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = PerceptionSerializer(perception, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class LoyaltySpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            loyalty = Loyalty.objects.get(user=user, company=company)
        except Loyalty.DoesNotExist:
            return Response({'error': 'Loyalty does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = LoyaltySerializer(loyalty)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            loyalty = Loyalty.objects.get(user=user, company=company)
        except Loyalty.DoesNotExist:
            return Response({'error': 'Loyalty does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = LoyaltySerializer(loyalty, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class AdvocacySpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            advocacy = Advocacy.objects.get(user=user, company=company)
        except Advocacy.DoesNotExist:
            return Response({'error': 'Advocacy does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AdvocacySerializer(advocacy)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            advocacy = Advocacy.objects.get(user=user, company=company)
        except Advocacy.DoesNotExist:
            return Response({'error': 'Advocacy does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AdvocacySerializer(advocacy, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class AttractionSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            attraction = Attraction.objects.get(user=user, company=company)
        except Attraction.DoesNotExist:
            return Response({'error': 'Attraction does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AttractionSerializer(attraction)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            attraction = Attraction.objects.get(user=user, company=company)
        except Attraction.DoesNotExist:
            return Response({'error': 'Attraction does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AttractionSerializer(attraction, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class InfluenceSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            influence = Influence.objects.get(user=user, company=company)
        except Influence.DoesNotExist:
            return Response({'error': 'Influence does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = InfluenceSerializer(influence)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            influence = Influence.objects.get(user=user, company=company)
        except Influence.DoesNotExist:
            return Response({'error': 'Influence does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = InfluenceSerializer(influence, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class BrandSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            brand = Brand.objects.get(user=user, company=company)
        except Brand.DoesNotExist:
            return Response({'error': 'Brand does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = BrandSerializer(brand)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            brand = Brand.objects.get(user=user, company=company)
        except Brand.DoesNotExist:
            return Response({'error': 'Brand does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = BrandSerializer(brand, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class AttributesOfGreatPlaceSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            attribute_of_great_place = AttributesOfGreatPlace.objects.get(user=user, company=company)
        except AttributesOfGreatPlace.DoesNotExist:
            return Response({'error': 'Attribute of Great Place does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AttributesOfGreatPlaceSerializer(attribute_of_great_place)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            attribute_of_great_place = AttributesOfGreatPlace.objects.get(user=user, company=company)
        except AttributesOfGreatPlace.DoesNotExist:
            return Response({'error': 'Attribute of Great Place does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AttributesOfGreatPlaceSerializer(attribute_of_great_place, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class KeyThemesSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            key_themes = KeyThemes.objects.get(user=user, company=company)
        except KeyThemes.DoesNotExist:
            return Response({'error': 'Key Themes does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = KeyThemesSerializer(key_themes)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            key_themes = KeyThemes.objects.get(user=user, company=company)
        except KeyThemes.DoesNotExist:
            return Response({'error': 'Key Themes does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = KeyThemesSerializer(key_themes, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class AudienceWiseMessagingSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            audience_wise_messaging = AudienceWiseMessaging.objects.get(user=user, company=company)
        except AudienceWiseMessaging.DoesNotExist:
            return Response({'error': 'Audience Wise Messaging does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AudienceWiseMessagingSerializer(audience_wise_messaging)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            audience_wise_messaging = AudienceWiseMessaging.objects.get(user=user, company=company)
        except AudienceWiseMessaging.DoesNotExist:
            return Response({'error': 'Audience Wise Messaging does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AudienceWiseMessagingSerializer(audience_wise_messaging, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class TalentInsightsAPIView(APIView):
    def post(self, request):
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)
        
        talent_dataset = TalentDataset.objects.filter(company=company)

        if not talent_dataset.exists():
            return Response({"error": "Talent Dataset does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = TalentInsightsSerializer(talent_dataset, many=True)
        all_talent_dataset = serializer.data

        talent_insights_from_chatgpt = get_talent_insights_from_chatgpt(company_name, all_talent_dataset)

        return Response(talent_insights_from_chatgpt, status=status.HTTP_200_OK)

class SwotAnalysisSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            swot_analysis = SwotAnalysis.objects.get(user=user, company=company)
        except SwotAnalysis.DoesNotExist:
            return Response({'error': 'Swot Analysis does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = SwotAnalysisSerializer(swot_analysis)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            swot_analysis = SwotAnalysis.objects.get(user=user, company=company)
        except SwotAnalysis.DoesNotExist:
            return Response({'error': 'Swot Analysis does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = SwotAnalysisSerializer(swot_analysis, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class AlignmentSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            alignment = Alignment.objects.get(user=user, company=company)
        except Alignment.DoesNotExist:
            return Response({'error': 'Alignment does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AlignmentSerializer(alignment)
        return Response(serializer.data)
    
    def patch(self, request, company_name):
        data = request.data
        user = request.user

        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({'error': 'Company does not exist'}, status=status.HTTP_404_NOT_FOUND)

        try:
            alignment = Alignment.objects.get(user=user, company=company)
        except Alignment.DoesNotExist:
            return Response({'error': 'Alignment does not exist'}, status=status.HTTP_404_NOT_FOUND)

        serializer = AlignmentSerializer(alignment, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class TaglineAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        company_id = company.id

        main_theme = request.data.get("main_theme")
        pillars = request.data.get("pillars", [])
        tagline = request.data.get("tagline")

        if MessagingHierarchyData.objects.filter(user=user, company=company_id).exists():
            existing_messaging_hierarchy_data = MessagingHierarchyData.objects.get(user=user, company=company_id)
            serializer = MessagingHierarchyDataSerializer(existing_messaging_hierarchy_data)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        if tagline:
            messaging_hierarchy_data = MessagingHierarchyData(
                user=user,
                company = company,
                main_theme = main_theme,
                pillar_1 = pillars[0] if len(pillars) > 0 else None,
                pillar_2 = pillars[1] if len(pillars) > 1 else None,
                pillar_3 = pillars[2] if len(pillars) > 2 else None,
                tagline = tagline,
            )
            print(messaging_hierarchy_data.pillar_1)
            print(messaging_hierarchy_data.pillar_2)
            print(messaging_hierarchy_data.pillar_3)
            messaging_hierarchy_data.save()
            serializer = MessagingHierarchyDataSerializer(messaging_hierarchy_data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        else:
            tabs_data_instances = MessagingHierarchyTabs.objects.filter(user=user, company=company_id)
            combined_tabs_data = " ".join(instance.tabs_data for instance in tabs_data_instances)

        tagline = get_tagline(
            main_theme,
            combined_tabs_data,
            pillars,
        )
        return Response({"tagline": tagline})
    
class MessagingHierarchySpecificAPIView(APIView):
    def get(self, request, company_name):
        # user = request.user

        try:
            company = Company.objects.get(name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            messaging_hierarchy = MessagingHierarchyData.objects.get(company=company)
        except MessagingHierarchyData.DoesNotExist:
            return Response({"error": "Messaging Hierarchy Data does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = MessagingHierarchyDataSerializer(messaging_hierarchy)
        return Response(serializer.data, status=status.HTTP_200_OK)

class CreativeDirectionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            creative_direction = CreativeDirection.objects.get(user=user, company=company)
        except CreativeDirection.DoesNotExist:
            creative_direction = None

        if creative_direction:
            serializer = CreativeDirectionSerializer(creative_direction)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        creative_direction_data = request.data.get("creative_direction_data")

        if creative_direction_data:
            creative_direction = CreativeDirection.objects.create(
                company=company,
                user=user,
                creative_direction_data=creative_direction_data,
            )
            serializer = CreativeDirectionSerializer(creative_direction)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        try:
            messaging_hierarchy_data = MessagingHierarchyData.objects.get(user=user, company=company)
        except MessagingHierarchyData.DoesNotExist:
            return Response({"error": "Messaging hierarchy data not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)
        
        brand_guidelines = company.brand_guidelines
        tagline = messaging_hierarchy_data.tagline

        try:
            creative_direction_from_chatgpt = get_creative_direction_from_chatgpt(
                brand_guidelines,
                tagline
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"creative_direction_data": creative_direction_from_chatgpt})
    
class EVPDefinitionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        if EVPDefinition.objects.filter(company=company).exists():
            evp_audit = EVPDefinition.objects.filter(company=company)
            serializer = EVPDefinitionSerializer(evp_audit, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        try:
            analysis_instance = SwotAnalysis.objects.get(user=user, company=company)
            serializer = SwotAnalysisSerializer(analysis_instance)
            analysis_data = serializer.data
        except SwotAnalysis.DoesNotExist:
            return Response({"error": "SWOT analysis not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)

        try:
            alignment_instance = Alignment.objects.get(user=user, company=company)
            serializer = AlignmentSerializer(alignment_instance)
            alignment_data = serializer.data
        except Alignment.DoesNotExist:
            return Response({"error": "Alignment data not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            messaging_hierarchy_data_instance = MessagingHierarchyData.objects.get(user=user, company=company)
        except MessagingHierarchyData.DoesNotExist:
            return Response({"error": "Messaging Hierarchy Data does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        themes_data_list = []
        if messaging_hierarchy_data_instance.main_theme is not None and len(messaging_hierarchy_data_instance.main_theme) > 0:
            themes_data_list.append(messaging_hierarchy_data_instance.main_theme)
        if messaging_hierarchy_data_instance.pillar_1 is not None and len(messaging_hierarchy_data_instance.pillar_1) > 0:
            themes_data_list.append(messaging_hierarchy_data_instance.pillar_1)
        if messaging_hierarchy_data_instance.pillar_2 is not None and len(messaging_hierarchy_data_instance.pillar_2) > 0:
            themes_data_list.append(messaging_hierarchy_data_instance.pillar_2)
        if messaging_hierarchy_data_instance.pillar_3 is not None and len(messaging_hierarchy_data_instance.pillar_3) > 0:
            themes_data_list.append(messaging_hierarchy_data_instance.pillar_3)

        all_themes = ", ".join(themes_data_list)

        try:
            evp_definition_from_chatgpt = get_evp_definition_from_chatgpt(company_name, user, analysis_data, alignment_data, all_themes)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(evp_definition_from_chatgpt)

    
class EVPPromiseAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)

        if EVPPromise.objects.filter(user=user, company=company).exists():
            evp_promise = EVPPromise.objects.filter(user=user, company=company)
            serializer = EVPPromiseSerializer(evp_promise, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        messaging_hierarchy_data = MessagingHierarchyData.objects.get(user=user, company=company)
        all_themes = []
        if messaging_hierarchy_data.main_theme is not None and len(messaging_hierarchy_data.main_theme):
            all_themes.append(messaging_hierarchy_data.main_theme)
        if messaging_hierarchy_data.pillar_1 is not None and len(messaging_hierarchy_data.pillar_1):
            all_themes.append(messaging_hierarchy_data.pillar_1)
        if messaging_hierarchy_data.pillar_2 is not None and len(messaging_hierarchy_data.pillar_2):
            all_themes.append(messaging_hierarchy_data.pillar_2)
        if messaging_hierarchy_data.pillar_3 is not None and len(messaging_hierarchy_data.pillar_3):
            all_themes.append(messaging_hierarchy_data.pillar_3)

        try:
            evp_promise_from_chatgpt = get_evp_promise_from_chatgpt(company_name, user, all_themes)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(evp_promise_from_chatgpt)
    
class EVPAuditAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        if EVPAudit.objects.filter(user=user, company=company).exists():
            evp_audit = EVPAudit.objects.filter(user=user, company=company)
            serializer = EVPAuditSerializer(evp_audit, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        try:
            analysis_instance = SwotAnalysis.objects.get(user=user, company=company)
            serializer = SwotAnalysisSerializer(analysis_instance)
            analysis_data = serializer.data
        except SwotAnalysis.DoesNotExist:
            return Response({"error": "SWOT analysis not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)

        try:
            alignment_instance = Alignment.objects.get(user=user, company=company)
            serializer = AlignmentSerializer(alignment_instance)
            alignment_data = serializer.data
        except Alignment.DoesNotExist:
            return Response({"error": "Alignment data not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)

        messaging_hierarchy_data = MessagingHierarchyData.objects.get(user=user, company=company)
        all_themes = []
        if messaging_hierarchy_data.main_theme is not None and len(messaging_hierarchy_data.main_theme):
            all_themes.append(messaging_hierarchy_data.main_theme)
        if messaging_hierarchy_data.pillar_1 is not None and len(messaging_hierarchy_data.pillar_1):
            all_themes.append(messaging_hierarchy_data.pillar_1)
        if messaging_hierarchy_data.pillar_2 is not None and len(messaging_hierarchy_data.pillar_2):
            all_themes.append(messaging_hierarchy_data.pillar_2)
        if messaging_hierarchy_data.pillar_3 is not None and len(messaging_hierarchy_data.pillar_3):
            all_themes.append(messaging_hierarchy_data.pillar_3)

        try:
            evp_audit_from_chatgpt = get_evp_audit_from_chatgpt(company_name, user, analysis_data, alignment_data, all_themes)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(evp_audit_from_chatgpt)
    
class EVPEmbedmentAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        all_touchPoints = request.data.get("touchpoints")

        try:
            top_4_themes_instances = MessagingHierarchyTabs.objects.filter(user=user, company=company)
        except MessagingHierarchyTabs.DoesNotExist:
            return Response({"error": "Messaging Hierarchy Tabs does not exist"}, status=status.HTTP_404_NOT_FOUND)

        try:
            tagline_instance = MessagingHierarchyData.objects.get(user=user, company=company)
        except MessagingHierarchyData.DoesNotExist:
            return Response({"error": "Messaging Hierarchy Data does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            evp_promise_instances = EVPPromise.objects.filter(user=user, company=company)
        except EVPPromise.DoesNotExist:
            return Response({"error": "EVP Promise does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            evp_audit_instances = EVPAudit.objects.filter(user=user, company=company)
        except EVPAudit.DoesNotExist:
            return Response({"error": "EVP Audit does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        top_4_themes_data = " ".join(instance.tabs_data for instance in top_4_themes_instances)
        
        tagline_data = tagline_instance.tagline

        evp_promise_data_list = []
        for instance in evp_promise_instances:
            data = f"{instance.what_employees_can_expect} {instance.what_is_expected_of_employees}"
            evp_promise_data_list.append(data)

        evp_promise_data = " ".join(evp_promise_data_list)

        evp_audit_data_list = []
        for instance in evp_audit_instances:
            data = f"{instance.what_makes_this_credible} {instance.where_do_we_need_to_stretch}"
            evp_audit_data_list.append(data)

        evp_audit_data = " ".join(evp_audit_data_list)
        
        evp_embedment_data_from_chatgpt = get_evp_embedment_data_from_chatgpt(
            company_name,
            user,
            all_touchPoints,
            top_4_themes_data,
            tagline_data,
            evp_promise_data,
            evp_audit_data
        )

        return Response(evp_embedment_data_from_chatgpt)
    
class EVPNarrativeAPIView(APIView):
    def get(self, request):
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        attribute_of_great_place = AttributesOfGreatPlace.objects.get(company=company)
        serializer = AttributesOfGreatPlaceSerializer(attribute_of_great_place)
        attribute_of_great_place_data = serializer.data

        key_themes = KeyThemes.objects.get(company=company)
        serializer = KeyThemesSerializer(key_themes)
        key_themes_data = serializer.data

        audience_wise_messaging = AudienceWiseMessaging.objects.get(company=company)
        serializer = AudienceWiseMessagingSerializer(audience_wise_messaging)
        audience_wise_messaging_data = serializer.data
        print("hello 1")

        prompt1 = f"""
                First analyze the Attribute Data :

                Attribute Data : {attribute_of_great_place_data}

                Now analyze the Key Themes Data :

                Key Themes Data : {key_themes_data}

                Now analyze the Audience Wise Messaging Data :

                Audience Wise Messaging Data : {audience_wise_messaging_data}

                Now create a summary of whole above data within 300 words
             """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role":"system",
                "content":"""You are an expert summary creator from the given data.
                            """
            },
            {
                "role":"user",
                "content":prompt1
            }
        ],
        temperature=0.3,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        data_1 = chat_response

        analysis = SwotAnalysis.objects.get(company=company)
        serializer = SwotAnalysisSerializer(analysis)
        analysis_data = serializer.data

        alignment = Alignment.objects.get(company=company)
        serializer = AlignmentSerializer(alignment)
        alignment_data = serializer.data
        print("hello 2")

        prompt2 = f"""
                First analyze the Analysis Data :

                Analysis Data : {analysis_data}

                Now analyze the Alignment Data :

                Alignment Data : {alignment_data}

                Now create a summary of whole above data within 300 words
             """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role":"system",
                "content":"""You are an expert summary creator from the given data.
                            """
            },
            {
                "role":"user",
                "content":prompt2
            }
        ],
        temperature=0.3,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        data_2 = chat_response

        top_4_themes = MessagingHierarchyTabs.objects.filter(company=company)
        serializer = MessagingHierarchyTabsSerializer(top_4_themes, many=True)
        top_4_themes_data = serializer.data

        messaging_hierarchy = MessagingHierarchyData.objects.get(company=company)
        serializer = MessagingHierarchyDataSerializer(messaging_hierarchy)
        messaging_hierarchy_data = serializer.data

        evp_promise = EVPPromise.objects.filter(company=company)
        serializer = EVPPromiseSerializer(evp_promise, many=True)
        evp_promise_data = serializer.data
        print("hello 3")

        prompt3 = f"""
                First analyze the Top 4 Themes Data :

                Top 4 Themes Data : {top_4_themes_data}

                Now analyze the Messaging Hierarchy Data :

                Messaging Hierarchy Data : {messaging_hierarchy_data}

                Now analyze the EVP Promise Data :

                EVP Promise Data : {evp_promise_data}

                Now create a summary of whole above data within 300 words
             """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role":"system",
                "content":"""You are an expert summary creator from the given data.
                            """
            },
            {
                "role":"user",
                "content":prompt3
            }
        ],
        temperature=0.3,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        data_3 = chat_response
        print("hello 4")

        prompt4 = f"""
                First analyze the Dataset 1 :

                Dataset 1 : {data_1}

                Now analyze the Dataset 2 :

                Dataset 2 : {data_2}

                Now analyze the Dataset 3 :

                Dataset 3 : {data_3}

                After analyzing all the above datasets, create a summary.
             """

        completion = chat_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages = [
            {
                "role":"system",
                "content":"""You are an expert summary creator from the given data.
                            """
            },
            {
                "role":"user",
                "content":prompt4
            }
        ],
        temperature=0.3,
        max_tokens=4000,
        )
        chat_response = completion.choices[0].message.content
        final_data = chat_response

        return Response(final_data)
    
class EVPHandBookAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        company_name = request.data.get("company_name")
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            evp_handbook = EVPHandbook.objects.get(user=user, company=company)
        except EVPHandbook.DoesNotExist:
            evp_handbook = None

        if evp_handbook:
            serializer = EVPHandbookSerializer(evp_handbook)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        handbook_data = request.data.get("handbook_data")
        if handbook_data:
            evp_handbook, created = EVPHandbook.objects.get_or_create(
                company = company,
                user = user,
                defaults = {"handbook_data": handbook_data}
            )
            if not created:
                evp_handbook.handbook_data = handbook_data
                evp_handbook.save()
            serializer = EVPHandbookSerializer(evp_handbook)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        top_4_themes = MessagingHierarchyTabs.objects.filter(user=user, company=company)
        serializer = MessagingHierarchyTabsSerializer(top_4_themes, many=True)
        top_4_themes_data = serializer.data

        messaging_hierarchy = MessagingHierarchyData.objects.get(user=user, company=company)
        serializer = MessagingHierarchyDataSerializer(messaging_hierarchy)
        messaging_hierarchy_data = serializer.data

        evp_promise = EVPPromise.objects.filter(user=user, company=company)
        serializer = EVPPromiseSerializer(evp_promise, many=True)
        evp_promise_data = serializer.data

        evp_audit = EVPAudit.objects.filter(user=user, company=company)
        serializer = EVPAuditSerializer(evp_audit, many=True)
        evp_audit_data = serializer.data

        evp_handbook_data_from_chatgpt = get_evp_handbook_data_from_chatgpt(
            company_name,
            user,
            top_4_themes_data,
            messaging_hierarchy_data,
            evp_promise_data,
            evp_audit_data,
        )

        return Response({"handbook_data": evp_handbook_data_from_chatgpt}, status=status.HTTP_200_OK)

class EVPStatementAndPillarsSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            evp_statement = EVPStatementAndPillars.objects.get(user=user, company=company)
        except EVPStatementAndPillars.DoesNotExist:
            return Response({"error": "EVPStatementAndPillars does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = EVPStatementAndPillarsSerializer(evp_statement)
        return Response(serializer.data)

    
class EVPExecutionPlanSpecificAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, company_name):
        user = request.user
        try:
            company = Company.objects.get(user=user, name=company_name)
        except Company.DoesNotExist:
            return Response({"error": "Company does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        stages = EVPEmbedmentStage.objects.filter(user=user, company=company)
        response_data = []

        for stage in stages:
            touchpoints = EVPEmbedmentTouchpoint.objects.filter(stage=stage)
            touchpoint_data = []
            for touchpoint in touchpoints:
                try:
                    message = EVPEmbedmentMessage.objects.get(touchpoint=touchpoint)
                    touchpoint_data.append({
                        "touchpoint": touchpoint.touchpoint_name,
                        "messaging_or_recommendation": message.message
                    })
                except EVPEmbedmentMessage.DoesNotExist:
                    touchpoint_data.append({
                        "touchpoint": touchpoint.touchpoint_name,
                        "messaging_or_recommendation": ""
                    })
            if touchpoint_data:
                response_data.append({
                    "stage": stage.stage_name,
                    "touchpoints": touchpoint_data
                })

        return Response(response_data, status=status.HTTP_200_OK)
    
from crawlbase import CrawlingAPI
import json
        
class TestCrawlBase(APIView):
    def post(self, request):
        api = CrawlingAPI({'token': '0T_Q68sxGTE-hmTNXnN5NQ'})

        targetURL = 'https://www.reddit.com/r/pics/comments/5bx4bx/thanks_obama/'

        response = api.get(targetURL, {'autoparse': 'true'})
        if response['status_code'] == 200:
            data = response["body"]
            try:
                json_response = json.loads(data)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                json_response = {}

            # return Response(data)
            return Response(json_response)
        return Response({"error": f"Some error occurred"})
