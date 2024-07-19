import os
import glob
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader, PdfWriter

from django.conf import settings

def save_documents(uploaded_document, directory_path):
    documents_folder = os.path.join(settings.MEDIA_ROOT, directory_path)
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
        
    file_path = os.path.join(documents_folder, uploaded_document.name)

    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_document.chunks():
            destination.write(chunk)
    return f"File {uploaded_document.name} saved successfully"

def merge_documents(source_folder, destination_folder, output_filename):
    pdf_writer = PdfWriter()

    # Construct the full path for the destination folder within MEDIA_ROOT
    destination_path = os.path.join(settings.MEDIA_ROOT, destination_folder)

    # Ensure the destination folder exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    output_path = os.path.join(destination_path, output_filename)
    
    # Adjusted to read PDFs from the source_folder
    paths = glob.glob(os.path.join(source_folder, '*.pdf'))
    paths.sort()

    for path in paths:
        pdf_reader = PdfReader(path)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output_path, 'wb') as out:
        pdf_writer.write(out)
    return "Hello"
