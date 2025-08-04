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
    file_type: str 
    createdAt: str
    
    class Config:
        allow_population_by_field_name = True  # Cho phép sử dụng cả field name và alias

def get_file_paths(file_type: str, filename: str) -> tuple[str, str]:
    """
    Trả về đường dẫn file và vector database dựa trên file_type
    
    Args:
        file_type: 'public', 'student', 'teacher', 'admin'
        filename: tên file
    
    Returns:
        tuple: (file_path, vector_db_path)
    """
    base_path = Config.DATA_PATH if hasattr(Config, 'DATA_PATH') else "data"
    
    type_mapping = {
        'public': {
            'file_folder': f"{base_path}/Public_Rag_Info/File_Folder",
            'vector_folder': f"{base_path}/Public_Rag_Info/Faiss_Folder"
        },
        'student': {
            'file_folder': f"{base_path}/Student_Rag_Info/File_Folder", 
            'vector_folder': f"{base_path}/Student_Rag_Info/Faiss_Folder"
        },
        'teacher': {
            'file_folder': f"{base_path}/Teacher_Rag_Info/File_Folder",
            'vector_folder': f"{base_path}/Teacher_Rag_Info/Faiss_Folder"
        },
        'admin': {
            'file_folder': f"{base_path}/Admin_Rag_Info/File_Folder",
            'vector_folder': f"{base_path}/Admin_Rag_Info/Faiss_Folder"
        }
    }
    
    if file_type not in type_mapping:
        raise ValueError(f"Invalid file_type: {file_type}. Must be one of: {list(type_mapping.keys())}")
    
    file_path = os.path.join(type_mapping[file_type]['file_folder'], filename)
    vector_db_path = type_mapping[file_type]['vector_folder']
    
    return file_path, vector_db_path

def save_metadata(metadata: AddVectorRequest):
    """Lưu metadata vào MongoDB, fallback vào metadata.json nếu có lỗi."""
    # Tạo đường dẫn metadata dựa trên file_type
    try:
        _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
        metadata_file = os.path.join(vector_db_path, "metadata.json")
    except ValueError as e:
        logger.error(f"Invalid file_type in metadata: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
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
        # Fallback: Lưu vào metadata.json theo file_type
        existing_metadata = []
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"No existing metadata.json found at {metadata_file}, creating new")
            existing_metadata = []
        
        metadata_dict = metadata.dict(by_alias=True)
        existing_metadata.append(metadata_dict)
        
        try:
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Fallback: Successfully saved metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata to JSON file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save metadata to JSON file: {str(e)}")

def extract_text_with_paddle(file_path: str) -> str:
    """Trích xuất văn bản từ file sử dụng PaddleOCR."""
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
    """Thêm tài liệu vào FAISS vector store dựa trên file_type."""
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
        # Lấy vector database path dựa trên file_type
        _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
        
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_exists = os.path.exists(f"{vector_db_path}/index.faiss") and os.path.exists(f"{vector_db_path}/index.pkl")
        
        if index_exists:
            logger.info(f"Loading existing FAISS index from {vector_db_path}")
            db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            logger.info(f"No existing FAISS index found, creating new one at {vector_db_path}")
            db = FAISS.from_documents(chunks, embedding_model)
            logger.info(f"FAISS index initialized successfully.")

        db.add_documents(chunks)
        os.makedirs(vector_db_path, exist_ok=True)
        db.save_local(vector_db_path)
        logger.info(f"Successfully saved FAISS index to {vector_db_path}")
    except Exception as e:
        logger.error(f"Error processing FAISS index: {str(e)}")
        raise