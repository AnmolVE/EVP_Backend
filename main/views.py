import os
import re
import shutil
from dotenv import load_dotenv
load_dotenv()

from django.contrib.auth import login

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    NewUser,
    Company,
    Perception,
    Loyalty,
    Advocacy,
    Attraction,
    Influence,
    Brand,
    AttributesOfGreatPlace,
    KeyThemes,
    AudienceWiseMessaging,
    SwotAnalysis,
    Alignment,
    MessagingHierarchyTabs,
    MessagingHierarchyData,
    EVPPromise,
    EVPAudit,
)
from .serializers import (
    NewUserSerializer,
    UserLoginSerializer,
    CompanySerializer,
    PerceptionSerializer,
    LoyaltySerializer,
    AdvocacySerializer,
    AttractionSerializer,
    InfluenceSerializer,
    BrandSerializer,
    AttributesOfGreatPlaceSerializer,
    KeyThemesSerializer,
    AudienceWiseMessagingSerializer,
    SwotAnalysisSerializer,
    AlignmentSerializer,
    MessagingHierarchyTabsSerializer,
    MessagingHierarchyDataSerializer,
    EVPPromiseSerializer,
    EVPAuditSerializer,
)

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
    testing_data,
    create_embeddings,
    query_with_langchain,
    save_pgData_to_vector_database,
    get_develop_data_from_vector_database,
    get_dissect_data_from_vector_database,
    get_design_data_from_database, get_tagline,
    get_creative_direction_from_chatgpt,
    get_evp_promise_from_chatgpt,
    get_evp_audit_from_chatgpt,
)
from .utils.email_send import send_email_to_users

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
    }

class LoginAPIView(APIView):
    def post(self, request):
        data = request.data
        serializer = UserLoginSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.validated_data['user']
            login(request, user)
            tokens = get_tokens_for_user(user)
            return Response({"tokens": tokens, "email": user.email}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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

            fields_to_query_with_chatgpt_1 = {field: "" for field in chatgpt_1_query_data if field in empty_fields_from_langchain}

            if (len(fields_to_query_with_chatgpt_1) > 0):
                data_from_chatgpt_1 = get_data_from_chatgpt_1(company_name, fields_to_query_with_chatgpt_1)
                final_data.update(data_from_chatgpt_1)

        else:
            print("In else block")
            data_from_bing = get_data_from_bing(company_name, bing_query_data)

            final_data.update(data_from_bing)

            data_from_chatgpt_1 = get_data_from_chatgpt_1(company_name, chatgpt_1_query_data)

            final_data.update(data_from_chatgpt_1)

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
            dissect_data_from_vector_database = get_dissect_data_from_vector_database(company_name, user)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # return Response(dissect_data_from_vector_database, status=status.HTTP_200_OK)
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
        
        company_id = company.id

        if MessagingHierarchyTabs.objects.filter(user=user, company=company_id).exists():
            messaging_hierarchy_tabs = MessagingHierarchyTabs.objects.filter(user=user, company=company_id)
            serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        try:
            design_data_from_vector_database = get_design_data_from_database(company_name, user)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(design_data_from_vector_database, status=status.HTTP_200_OK)

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
            ids=[f"id{i}" for i in range(len(documents))]
            embeddings = create_embeddings()
            embedded_documents = embeddings([documents[i] for i in range(len(documents))])
        else:
            text_chunks = []
 
        sanitized_company_name = re.sub(r'\s+', '_', company_name)

        persistent_directory = f"vector_databases/{sanitized_company_name}"
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        if os.path.exists(os.path.join(persistent_directory)):
            if text_chunks:
                chatbot_collection = chroma_client.get_or_create_collection(
                    name="test",
                    embedding_function=embeddings,
                )

                chatbot_collection.add(
                    embeddings=embedded_documents,
                    documents=documents,
                    ids=ids,
                )
                return Response("Transcript added successfully", status=status.HTTP_201_CREATED)
        return Response("Please upload at least one document", status=status.HTTP_400_BAD_REQUEST)

class DesignPrinciplesAPIView(APIView):
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
            ids=[f"id{i}" for i in range(len(documents))]
            embeddings = create_embeddings()
            embedded_documents = embeddings([documents[i] for i in range(len(documents))])
        else:
            text_chunks = []
 
        sanitized_company_name = re.sub(r'\s+', '_', company_name)

        persistent_directory = f"vector_databases/{sanitized_company_name}"
        chroma_client = chromadb.PersistentClient(path=persistent_directory)
        if os.path.exists(os.path.join(persistent_directory)):
            if text_chunks:
                chatbot_collection = chroma_client.get_or_create_collection(
                    name="test_design",
                    embedding_function=embeddings,
                )

                chatbot_collection.add(
                    embeddings=embedded_documents,
                    documents=documents,
                    ids=ids,
                )
                return Response("Design Principles added successfully", status=status.HTTP_201_CREATED)
        return Response("Please upload at least one document", status=status.HTTP_400_BAD_REQUEST)

class CompanyAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            company = Company.objects.get(user=user)
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = CompanySerializer(company)
        return Response(serializer.data)

    def post(self, request):
        data = request.data
        user = request.user
        data["user"] = request.user.id
        serializer = CompanySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
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
    
class PerceptionAPIView(APIView):
    def get(self, request):
        perceptions = Perception.objects.all()
        serializer = PerceptionSerializer(perceptions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = PerceptionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
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
    
class LoyaltyAPIView(APIView):
    def get(self, request):
        loyalties = Loyalty.objects.all()
        serializer = LoyaltySerializer(loyalties, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = LoyaltySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class AdvocacyAPIView(APIView):
    def get(self, request):
        advocacies = Advocacy.objects.all()
        serializer = AdvocacySerializer(advocacies, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = AdvocacySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class AttractionAPIView(APIView):
    def get(self, request):
        attractions = Attraction.objects.all()
        serializer = AttractionSerializer(attractions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = AttractionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class InfluenceAPIView(APIView):
    def get(self, request):
        influences = Influence.objects.all()
        serializer = InfluenceSerializer(influences, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = InfluenceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class BrandAPIView(APIView):
    def get(self, request):
        brands = Brand.objects.all()
        serializer = BrandSerializer(brands, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = BrandSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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

class AttributesOfGreatPlaceAPIView(APIView):
    def get(self, request):
        attributes_of_great_place = AttributesOfGreatPlace.objects.all()
        serializer = AttributesOfGreatPlaceSerializer(attributes_of_great_place, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = AttributesOfGreatPlaceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class KeyThemesAPIView(APIView):
    def get(self, request):
        key_themes = KeyThemes.objects.all()
        serializer = KeyThemesSerializer(key_themes, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = KeyThemesSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
    
class AudienceWiseMessagingAPIView(APIView):
    def get(self, request):
        audience_wise_messaging = AudienceWiseMessaging.objects.all()
        serializer = AudienceWiseMessagingSerializer(audience_wise_messaging, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = AudienceWiseMessagingSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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

class SwotAnalysisAPIView(APIView):
    def get(self, request):
        swot_analysis = SwotAnalysis.objects.all()
        serializer = SwotAnalysisSerializer(swot_analysis, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = SwotAnalysisSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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
    
class AlignmentAPIView(APIView):
    def get(self, request):
        alignment = Alignment.objects.all()
        serializer = AlignmentSerializer(alignment, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = AlignmentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
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
        pillar_1 = pillar_2 = pillar_3 = None
        if len(pillars) >= 3:
            pillar_1 = pillars[0]
            pillar_2 = pillars[1]
            pillar_3 = pillars[2]
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
                pillar_1 = pillar_1,
                pillar_2 = pillar_2,
                pillar_3 = pillar_3,
                tagline = tagline,
            )
            messaging_hierarchy_data.save()
            serializer = MessagingHierarchyDataSerializer(messaging_hierarchy_data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        else:
            tabs_data_instances = MessagingHierarchyTabs.objects.filter(user=user, company=company_id)
            combined_tabs_data = ""
            for instance in tabs_data_instances:
                combined_tabs_data = combined_tabs_data + " " + instance.tabs_data

        if pillar_1 and pillar_2 and pillar_3:
            tagline = get_tagline(company_name, user, main_theme, pillar_1, pillar_2, pillar_3, combined_tabs_data)
        else:
            tagline = get_tagline(company_name, main_theme, combined_tabs_data)
        return Response({"tagline": tagline})

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
            brand = Brand.objects.get(user=user, company=company)
        except Brand.DoesNotExist:
            return Response({"error": "Brand not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            messaging_hierarchy_data = MessagingHierarchyData.objects.get(user=user, company=company)
        except MessagingHierarchyData.DoesNotExist:
            return Response({"error": "Messaging hierarchy data not found for the specified company"}, status=status.HTTP_404_NOT_FOUND)
        
        brand_guidelines = brand.brand_guidelines
        tagline = messaging_hierarchy_data.tagline

        try:
            creative_direction_from_chatgpt = get_creative_direction_from_chatgpt(brand_guidelines, tagline)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"creative_direction": creative_direction_from_chatgpt})
    
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

        messaging_hierarchy_tabs_data = MessagingHierarchyTabs.objects.filter(user=user, company=company)
        serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs_data, many=True)
        themes_data = serializer.data

        try:
            evp_promise_from_chatgpt = get_evp_promise_from_chatgpt(company_name, user, themes_data)
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

        messaging_hierarchy_tabs_data = MessagingHierarchyTabs.objects.filter(user=user, company=company)
        serializer = MessagingHierarchyTabsSerializer(messaging_hierarchy_tabs_data, many=True)
        themes_data = serializer.data

        themes_data_list = [theme["tab_name"] for theme in themes_data]
        four_themes = ", ".join(themes_data_list)

        try:
            evp_audit_from_chatgpt = get_evp_audit_from_chatgpt(company_name, user, analysis_data, alignment_data, four_themes)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(evp_audit_from_chatgpt)
    
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
