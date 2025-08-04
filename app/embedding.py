import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import pdfplumber
import pandas as pd
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
from app.config import Config
from pydantic import BaseModel, Field
from typing import Dict, List
from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo OCR
try:
    ocr = PaddleOCR(lang="vi", use_textline_orientation=True)
    logger.info("PaddleOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
    raise

# Định nghĩa mô hình Pydantic cho metadata
class AddVectorRequest(BaseModel):
    id: str = Field(alias="_id")  # Sử dụng alias để map từ _id
    filename: str
    url: str
    uploaded_by: str
    role: Dict[str, List[str]]
    createdAt: str
    
    class Config:
        allow_population_by_field_name = True  # Cho phép sử dụng cả field name và alias

def save_metadata(metadata: AddVectorRequest):
    """Lưu metadata vào MongoDB, fallback vào metadata.json nếu có lỗi."""
    import os
    metadata_file = f"{Config.DATA_PATH}/metadata.json"
    try:
        # Kết nối MongoDB
        logger.info(f"Attempting to connect to MongoDB at mongodb://localhost:27017/")
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        # Chuyển đổi metadata thành dict với _id
        metadata_dict = metadata.dict(by_alias=True)
        
        # Lưu metadata vào MongoDB
        collection.insert_one(metadata_dict)
        logger.info(f"Successfully saved metadata to MongoDB for _id: {metadata.id}")
        client.close()
    except PyMongoError as e:
        logger.error(f"Failed to save metadata to MongoDB: {str(e)}")
        # Fallback: Lưu vào metadata.json
        existing_metadata = []
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"No existing metadata.json found, creating new")
            existing_metadata = []
        
        metadata_dict = metadata.dict(by_alias=True)
        existing_metadata.append(metadata_dict)
        
        try:
            os.makedirs(Config.DATA_PATH, exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Fallback: Successfully saved metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata to JSON file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save metadata to JSON file: {str(e)}")

def extract_text_with_paddle(file_path: str) -> str:
    """Trích xuất văn bản từ file sử dụng PaddleOCR."""
    import os
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Extracting text with PaddleOCR from {file_path}")
        result = ocr.ocr(file_path)
        text = "\n".join([line[1][0] for line in result[0] if len(line) > 1]) if result and result[0] else ""
        logger.info(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"OCR Error for {file_path}: {str(e)}")
        return ""

def process_pdf(file_path: str) -> tuple[list, list]:
    """Xử lý file PDF, trích xuất bảng và văn bản."""
    import os
    tables, texts = [], []
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Processing PDF: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
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
                        logger.info(f"Used PaddleOCR for text extraction: {file_path}")
                        break
                else:
                    texts.append(text)
        logger.info(f"Extracted {len(tables)} tables and {len(texts)} text segments")
    except Exception as e:
        logger.error(f"PDF Processing Error for {file_path}: {str(e)}")
    return tables, texts

def load_new_documents(file_path: str, metadata: AddVectorRequest) -> list:
    """Tải và xử lý tài liệu từ file_path, gắn metadata."""
    import os
    documents = []
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return documents

    extension = file_path.lower().split('.')[-1]
    supported_extensions = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'docx': Docx2txtLoader,
        'csv': CSVLoader,
        'xlsx': UnstructuredExcelLoader,
        'xls': UnstructuredExcelLoader
    }

    if extension in supported_extensions:
        try:
            logger.info(f"Loading document: {file_path} with extension {extension}")
            if extension == 'pdf':
                tables, texts = process_pdf(file_path)
                metadata_dict = metadata.dict(by_alias=True)
                for table in tables:
                    table_text = table.to_csv(index=False)
                    documents.append(Document(page_content=table_text, metadata=metadata_dict))
                for text in texts:
                    documents.append(Document(page_content=text, metadata=metadata_dict))
            else:
                loader = supported_extensions[extension](file_path)
                loaded_docs = loader.load()
                metadata_dict = metadata.dict(by_alias=True)
                for doc in loaded_docs:
                    documents.append(Document(page_content=doc.page_content, metadata=metadata_dict))
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    return documents

def add_to_embedding(file_path: str, metadata: AddVectorRequest):
    """Thêm tài liệu vào FAISS vector store."""
    import os
    documents = load_new_documents(file_path, metadata)
    if not documents:
        logger.warning(f"No documents loaded from {file_path}, skipping embedding.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks from {file_path}")

    if not chunks:
        logger.warning(f"No chunks created from {file_path}, skipping embedding.")
        return

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_exists = os.path.exists(f"{Config.VECTOR_DB_PATH}/index.faiss") and os.path.exists(f"{Config.VECTOR_DB_PATH}/index.pkl")
        
        if index_exists:
            logger.info(f"Loading existing FAISS index from {Config.VECTOR_DB_PATH}")
            db = FAISS.load_local(Config.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            logger.info(f"No existing FAISS index found, creating new one at {Config.VECTOR_DB_PATH}")
            db = FAISS.from_documents(chunks, embedding_model)
            logger.info(f"FAISS index initialized successfully.")

        db.add_documents(chunks)
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        db.save_local(Config.VECTOR_DB_PATH)
        logger.info(f"Successfully saved FAISS index to {Config.VECTOR_DB_PATH}")
    except Exception as e:
        logger.error(f"Error processing FAISS index: {str(e)}")
        raise