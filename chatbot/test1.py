import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader, PdfWriter
import os
import glob

def merge_pdfs(source_folder, destination_folder, output_filename):
    pdf_writer = PdfWriter()

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    output_path = os.path.join(destination_folder, output_filename)
    
    # Adjusted to read PDFs from the source_folder
    paths = glob.glob(os.path.join(source_folder, '*.pdf'))
    paths.sort()

    for path in paths:
        pdf_reader = PdfReader(path)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output_path, 'wb') as out:
        pdf_writer.write(out)

merge_pdfs("documents", "final_pdf", "merged_pdf.pdf")
pdfreader = PdfReader("final_pdf/merged_pdf.pdf")

loader = PyPDFDirectoryLoader("final_pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=20)
text = text_splitter.split_documents(data)

persistent_directory = "db"

embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=text, embedding=embeddings, persist_directory=persistent_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("")

llm = OpenAI()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "When was the titanic released?"

llm_response = qa_chain(query)

def process_llm_response(llm_response):
    print("Answer :", (llm_response["result"]))
    print("\n\nSources: ")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])

process_llm_response(llm_response)
