from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader, PdfWriter
import os
import glob

load_dotenv()

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

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Testing the embedding model")

from langchain.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vector_db"
COLLECTION_NAME = "testing"

db = PGVector.from_documents(embedding=embeddings, documents=text, collection_name=COLLECTION_NAME, connection_string=CONNECTION_STRING)
