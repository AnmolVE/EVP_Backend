from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.llms import AzureOpenAI
from langchain.vectorstores import Pinecone as PC
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_API_KEY=os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT=os.environ["AZURE_OPENAI_ENDPOINT"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]


loader = PyPDFDirectoryLoader("pdfs")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)


embeddings = AzureOpenAIEmbeddings(api_key=AZURE_OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "testing"

index = pc.Index(index_name)

docsearch = PC.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

query = "YOLOv7 outperforms which model?"

docs = docsearch.similarity_search(query)


llm = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)

docsearch.as_retriever()
print(docsearch)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), verbose=True)
query = "YOLOv7 outperforms which model?"
print(qa.invoke(query))


# import sys
# while True:
#     user_input = input(f"Input Prompt: ")
#     if user_input == "exit":
#         print("Exiting")
#         sys.exit()
#     if user_input == "":
#         continue
#     result = qa.invoke({"query": user_input})
#     print(f"Answer: {result['result']}")
