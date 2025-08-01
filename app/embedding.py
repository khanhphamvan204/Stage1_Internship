import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pdfplumber
import pandas as pd
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from paddleocr import PaddleOCR
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from .config import Config

# Khởi tạo OCR
ocr = PaddleOCR(lang="vi", use_textline_orientation=True)

# Định nghĩa mô hình Pydantic cho metadata
from pydantic import BaseModel
class AddVectorRequest(BaseModel):
    id: str
    title: str
    description: str
    uploader: str

def extract_text_with_paddle(file_path):
    try:
        result = ocr.ocr(file_path)
        text = "\n".join([line[1][0] for line in result[0] if len(line) > 1]) if result and result[0] else ""
        return text
    except Exception as e:
        print(f"OCR Error for {file_path}: {e}")
        return ""

def process_pdf(file_path):
    tables, texts = [], []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted_tables = page.extract_tables()
                for table in extracted_tables or []:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        tables.append(df)
                text = page.extract_text()
                if not text:
                    full_text = extract_text_with_paddle(file_path)
                    if full_text:
                        texts.append(full_text)
                        break
                else:
                    texts.append(text)
    except Exception as e:
        print(f"PDF Processing Error for {file_path}: {e}")
    return tables, texts

def load_metadata():
    metadata_file = os.path.join(Config.DATA_PATH, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return {item["file_name"]: AddVectorRequest(**item) for item in json.load(f)}
    return {}

def load_new_documents(file_path, metadata):
    documents = []
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_path.lower())[1]
    supported_extensions = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.docx': Docx2txtLoader, '.csv': CSVLoader, '.xlsx': UnstructuredExcelLoader, '.xls': UnstructuredExcelLoader}

    if extension in supported_extensions:
        try:
            if extension == '.pdf':
                tables, texts = process_pdf(file_path)
                for table in tables:
                    table_text = table.to_csv(index=False)
                    documents.append(Document(page_content=table_text, metadata={
                        "id": metadata.id,
                        "title": metadata.title,
                        "description": metadata.description,
                        "uploader": metadata.uploader
                    }))
                for text in texts:
                    documents.append(Document(page_content=text, metadata={
                        "id": metadata.id,
                        "title": metadata.title,
                        "description": metadata.description,
                        "uploader": metadata.uploader
                    }))
            else:
                loader = supported_extensions[extension](file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append(Document(page_content=doc.page_content, metadata={
                        "id": metadata.id,
                        "title": metadata.title,
                        "description": metadata.description,
                        "uploader": metadata.uploader
                    }))
            print(f"Loaded {len(documents)} documents from {file_path}")
            for i, doc in enumerate(documents):
                print(f"Document {i}: page_content length={len(doc.page_content)}, metadata={doc.metadata}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents

def add_to_embedding(file_path, metadata):
    documents = load_new_documents(file_path, metadata)
    if not documents:
        print(f"No documents loaded from {file_path}, skipping embedding.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks from {file_path}")

    if not chunks:
        print(f"No chunks created from {file_path}, skipping embedding.")
        return

    # Load FAISS index hiện có nếu tồn tại, nếu không thì tạo mới
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_exists = os.path.exists(os.path.join(Config.VECTOR_DB_PATH, "index.faiss")) and os.path.exists(os.path.join(Config.VECTOR_DB_PATH, "index.pkl"))
    
    try:
        if index_exists:
            print(f"Loading existing FAISS index from {Config.VECTOR_DB_PATH}")
            db = FAISS.load_local(Config.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            print(f"No existing FAISS index found, creating new one at {Config.VECTOR_DB_PATH}")
            db = FAISS.from_documents(chunks, embedding_model)  # Khởi tạo với chunks thay vì danh sách rỗng
            print(f"FAISS index initialized successfully.")

        # Thêm các chunk mới
        db.add_documents(chunks)
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        db.save_local(Config.VECTOR_DB_PATH)
        print(f"Successfully saved FAISS index to {Config.VECTOR_DB_PATH}")
    except Exception as e:
        print(f"Error processing FAISS index: {e}")
        raise