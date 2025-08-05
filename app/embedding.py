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
import shutil

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
    file_path = file_path.replace("\\", "/")
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

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata: AddVectorRequest) -> bool:
    """
    Cập nhật metadata của document trong FAISS vector store.
    Cách tiếp cận: Xóa tất cả chunks cũ và thêm lại với metadata mới.
    
    Args:
        doc_id: ID của document cần cập nhật
        old_metadata: Metadata cũ
        new_metadata: Metadata mới
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        logger.info(f"Updating metadata in vector store for document: {doc_id}")
        
        # Lấy đường dẫn vector database từ old_metadata
        old_file_type = old_metadata.get('file_type')
        old_filename = old_metadata.get('filename')
        
        if not old_file_type or not old_filename:
            logger.error(f"Missing file_type or filename in old_metadata for doc_id: {doc_id}")
            return False
        
        _, old_vector_db_path = get_file_paths(old_file_type, old_filename)
        
        # Kiểm tra sự tồn tại của vector database cũ
        if not (os.path.exists(f"{old_vector_db_path}/index.faiss") and 
                os.path.exists(f"{old_vector_db_path}/index.pkl")):
            logger.warning(f"Old vector database not found at {old_vector_db_path}")
            return False
        
        # 1. Xóa document cũ khỏi vector database cũ
        success = delete_from_faiss_index(old_vector_db_path, doc_id)
        if not success:
            logger.error(f"Failed to delete old document from vector database")
            return False
        
        # 2. Thêm document với metadata mới vào vector database mới
        file_path = new_metadata.url
        if os.path.exists(file_path):
            add_to_embedding(file_path, new_metadata)
            logger.info(f"Successfully updated metadata in vector store for document: {doc_id}")
            return True
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata: AddVectorRequest) -> bool:
    """
    Cập nhật chỉ metadata trong các vector chunks hiện có mà không thay đổi embeddings.
    Phương pháp này nhanh hơn nhưng yêu cầu truy cập trực tiếp vào docstore.
    
    Args:
        doc_id: ID của document cần cập nhật
        new_metadata: Metadata mới
    
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        logger.info(f"Updating metadata only for document: {doc_id}")
        
        # Lấy vector database path
        _, vector_db_path = get_file_paths(new_metadata.file_type, new_metadata.filename)
        
        # Kiểm tra sự tồn tại của vector database
        if not (os.path.exists(f"{vector_db_path}/index.faiss") and 
                os.path.exists(f"{vector_db_path}/index.pkl")):
            logger.warning(f"Vector database not found at {vector_db_path}")
            return False
        
        # Tải FAISS index
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        
        # Tìm và cập nhật metadata cho tất cả chunks của document
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        updated_count = 0
        new_metadata_dict = new_metadata.dict(by_alias=True)
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                # Cập nhật metadata
                doc.metadata = new_metadata_dict
                docstore.add({docstore_id: doc})
                updated_count += 1
        
        if updated_count > 0:
            # Lưu lại FAISS index với metadata đã cập nhật
            db.save_local(vector_db_path)
            logger.info(f"Successfully updated metadata for {updated_count} chunks of document: {doc_id}")
            return True
        else:
            logger.warning(f"No chunks found for document: {doc_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata only: {str(e)}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata: AddVectorRequest, 
                         force_re_embed: bool = False) -> bool:
    """
    Cập nhật metadata thông minh: chọn phương pháp tối ưu dựa trên loại thay đổi.
    
    Args:
        doc_id: ID của document
        old_metadata: Metadata cũ
        new_metadata: Metadata mới
        force_re_embed: Buộc re-embed toàn bộ document
    
    Returns:
        bool: True nếu thành công
    """
    try:
        # Kiểm tra loại thay đổi
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        # Nếu file_type hoặc filename thay đổi, hoặc force_re_embed = True -> re-embed toàn bộ
        if file_type_changed or filename_changed or force_re_embed:
            logger.info(f"File type or filename changed, performing full re-embedding for doc_id: {doc_id}")
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata)
        else:
            # Chỉ thay đổi metadata đơn giản (uploaded_by, role) -> cập nhật metadata only
            logger.info(f"Only metadata changed, performing metadata-only update for doc_id: {doc_id}")
            success = update_metadata_only(doc_id, new_metadata)
            
            # Nếu metadata-only update thất bại, fallback sang full re-embed
            if not success:
                logger.warning(f"Metadata-only update failed, falling back to full re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata)
            
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False

def delete_metadata(doc_id: str) -> bool:
    """Xóa metadata từ MongoDB và metadata.json fallback."""
    success = False
    
    # Thử xóa từ MongoDB trước
    try:
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        result = collection.delete_one({"_id": doc_id})
        if result.deleted_count > 0:
            logger.info(f"Successfully deleted metadata from MongoDB for _id: {doc_id}")
            success = True
        else:
            logger.warning(f"No document found in MongoDB with _id: {doc_id}")
        client.close()
    except PyMongoError as e:
        logger.error(f"Failed to delete metadata from MongoDB: {str(e)}")
    
    # Fallback: Xóa từ tất cả các file metadata.json
    if not success:
        base_path = Config.DATA_PATH if hasattr(Config, 'DATA_PATH') else "data"
        metadata_paths = [
            f"{base_path}/Public_Rag_Info/Faiss_Folder/metadata.json",
            f"{base_path}/Student_Rag_Info/Faiss_Folder/metadata.json",
            f"{base_path}/Teacher_Rag_Info/Faiss_Folder/metadata.json",
            f"{base_path}/Admin_Rag_Info/Faiss_Folder/metadata.json"
        ]
        
        for metadata_file in metadata_paths:
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                    
                    # Tìm và xóa metadata
                    original_length = len(metadata_list)
                    metadata_list = [item for item in metadata_list if item.get('_id') != doc_id]
                    
                    if len(metadata_list) < original_length:
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                        logger.info(f"Successfully deleted metadata from {metadata_file}")
                        success = True
            except Exception as e:
                logger.error(f"Error deleting from {metadata_file}: {str(e)}")
    
    return success

def find_document_info(doc_id: str) -> dict:
    """Tìm thông tin document để xác định đường dẫn file và vector DB."""
    # Thử tìm từ MongoDB trước
    try:
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        doc_info = collection.find_one({"_id": doc_id})
        client.close()
        
        if doc_info:
            return doc_info
    except PyMongoError as e:
        logger.error(f"Failed to find document in MongoDB: {str(e)}")
    
    # Fallback: Tìm từ metadata.json files
    base_path = Config.DATA_PATH if hasattr(Config, 'DATA_PATH') else "data"
    metadata_paths = [
        f"{base_path}/Public_Rag_Info/Faiss_Folder/metadata.json",
        f"{base_path}/Student_Rag_Info/Faiss_Folder/metadata.json", 
        f"{base_path}/Teacher_Rag_Info/Faiss_Folder/metadata.json",
        f"{base_path}/Admin_Rag_Info/Faiss_Folder/metadata.json"
    ]
    
    for metadata_file in metadata_paths:
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                
                for item in metadata_list:
                    if item.get('_id') == doc_id:
                        return item
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")
    
    return None

def delete_from_faiss_index(vector_db_path: str, doc_id: str) -> bool:
    """
    Xóa tài liệu khỏi chỉ mục FAISS dựa trên doc_id.
    
    Args:
        vector_db_path (str): Đường dẫn đến thư mục chứa chỉ mục FAISS.
        doc_id (str): ID của tài liệu cần xóa (tương ứng với metadata._id).
    
    Returns:
        bool: True nếu xóa thành công hoặc không tìm thấy chỉ mục/tài liệu, False nếu có lỗi.
    """
    try:
        # Kiểm tra sự tồn tại của chỉ mục FAISS
        index_path = f"{vector_db_path}/index.faiss"
        pkl_path = f"{vector_db_path}/index.pkl"
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            logger.warning(f"Không tìm thấy chỉ mục FAISS tại {vector_db_path}")
            return True  # Không có chỉ mục thì coi như đã xóa thành công
        
        # Tải chỉ mục FAISS
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        
        # Lấy tất cả docstore_id và kiểm tra metadata._id
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        ids_to_delete = []
        
        # Tìm docstore_id tương ứng với doc_id trong metadata
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                ids_to_delete.append(docstore_id)
        
        # Xóa tài liệu nếu tìm thấy
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            db.save_local(vector_db_path)
            logger.info(f"Đã xóa thành công {len(ids_to_delete)} tài liệu với _id: {doc_id} khỏi chỉ mục FAISS tại {vector_db_path}")
        else:
            logger.warning(f"Không tìm thấy tài liệu nào với _id: {doc_id} trong chỉ mục FAISS")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi xóa khỏi chỉ mục FAISS: {str(e)}")
        return False