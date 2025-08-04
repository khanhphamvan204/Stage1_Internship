import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from app.embedding import add_to_embedding, delete_from_faiss_index, delete_metadata, find_document_info, save_metadata, AddVectorRequest, get_file_paths
from app.config import Config
import shutil
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình môi trường
try:
    os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
    logger.info("GOOGLE_API_KEY configured successfully")
except AttributeError as e:
    logger.error(f"Config error: {str(e)}")
    raise HTTPException(status_code=500, detail="Configuration error: Missing GEMINI_API_KEY")

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái API."""
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post("/documents/vector/add", response_model=dict)
async def add_vector_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form(...),
    file_type: str = Form(...),  # Thêm trường file_type bắt buộc
    role_user: str = Form(default="[]"),
    role_subject: str = Form(default="[]")
):
    """
    API endpoint để thêm tài liệu, lưu metadata và embedding vào FAISS.
    
    Args:
        file: File upload
        uploaded_by: Người upload
        file_type: Loại file ('public', 'student', 'teacher', 'admin')
        role_user: JSON string chứa danh sách user roles
        role_subject: JSON string chứa danh sách subject roles
    """
    try:
        logger.info(f"Processing file: {file.filename}, uploaded_by: {uploaded_by}, file_type: {file_type}")
        
        # Validate file_type
        valid_file_types = ['public', 'student', 'teacher', 'admin']
        if file_type not in valid_file_types:
            logger.error(f"Invalid file_type: {file_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file_type. Must be one of: {valid_file_types}"
            )
        
        # Tạo metadata
        generated_id = str(uuid.uuid4())
        vietnam_tz = timezone(timedelta(hours=7))
        created_at = datetime.now(vietnam_tz).isoformat()
        file_name = file.filename
        
        # Lấy đường dẫn file và vector database dựa trên file_type
        try:
            file_path, vector_db_path = get_file_paths(file_type, file_name)
        except ValueError as e:
            logger.error(f"Error getting file paths: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        file_url = f"{file_path}"
        
        logger.info(f"Generated metadata: _id={generated_id}, createdAt={created_at}")
        logger.info(f"File will be saved to: {file_path}")
        logger.info(f"Vector DB will be saved to: {vector_db_path}")

        try:
            role = {
                "user": json.loads(role_user),
                "subject": json.loads(role_subject)
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON format for role_user or role_subject")

        # Tạo metadata object với _id được map thành id field
        metadata = AddVectorRequest(
            _id=generated_id,  # Sử dụng _id như alias
            filename=file_name,
            url=file_url,
            uploaded_by=uploaded_by,
            role=role,
            file_type=file_type,  # Thêm file_type vào metadata
            createdAt=created_at
        )

        # Kiểm tra định dạng file
        supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file_name.lower())[1]
        if file_extension not in supported_extensions:
            logger.error(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail=f"File format {file_extension} not supported")

        # Lưu file vào thư mục tương ứng với file_type
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            if not os.path.exists(file_path):
                logger.error(f"Failed to save file: {file_path}")
                raise HTTPException(status_code=500, detail=f"Failed to save file: {file_path}")
            logger.info(f"File saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

        # Lưu metadata
        try:
            save_metadata(metadata)
            logger.info(f"Metadata saved for _id: {generated_id}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")

        # Embedding
        try:
            add_to_embedding(file_path, metadata)
            logger.info(f"Embedding completed for file: {file_path}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

        return {
            "message": "Vector added successfully", 
            "_id": generated_id,
            "file_type": file_type,
            "file_path": file_path,
            "vector_db_path": vector_db_path
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/documents/types")
async def get_file_types():
    """API endpoint để lấy danh sách các loại file được hỗ trợ."""
    return {
        "file_types": [
            {
                "value": "public",
                "label": "Thông báo chung (Public)",
                "description": "Tài liệu công khai cho tất cả người dùng"
            },
            {
                "value": "student", 
                "label": "Sinh viên (Student)",
                "description": "Tài liệu dành cho sinh viên"
            },
            {
                "value": "teacher",
                "label": "Giảng viên (Teacher)", 
                "description": "Tài liệu dành cho giảng viên"
            },
            {
                "value": "admin",
                "label": "Quản trị viên (Admin)",
                "description": "Tài liệu dành cho quản trị viên"
            }
        ]
    }
@app.delete("/documents/vector/{doc_id}")
async def delete_vector_document(doc_id: str):
    """
    API endpoint để xóa tài liệu, bao gồm file, vector DB và metadata.
    
    Args:
        doc_id: ID của document cần xóa
    """
    try:
        logger.info(f"Deleting document with ID: {doc_id}")
        
        # Tìm thông tin document
        doc_info = find_document_info(doc_id)
        if not doc_info:
            logger.error(f"Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        file_type = doc_info.get('file_type')
        filename = doc_info.get('filename')
        file_path = doc_info.get('url')
        
        if not all([file_type, filename]):
            logger.error(f"Incomplete document information for ID: {doc_id}")
            raise HTTPException(status_code=400, detail="Incomplete document information")
        
        # Lấy đường dẫn vector database
        try:
            _, vector_db_path = get_file_paths(file_type, filename)
        except ValueError as e:
            logger.error(f"Error getting file paths: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        deletion_results = {
            "file_deleted": False,
            "metadata_deleted": False,
            "vector_deleted": False
        }
        
        # 1. Xóa file vật lý
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                deletion_results["file_deleted"] = True
                logger.info(f"Successfully deleted file: {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")
                deletion_results["file_deleted"] = True  # Coi như đã xóa nếu không tồn tại
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        # 2. Xóa từ FAISS vector database
        try:
            deletion_results["vector_deleted"] = delete_from_faiss_index(vector_db_path, doc_id)
        except Exception as e:
            logger.error(f"Error deleting from vector database: {str(e)}")
        
        # 3. Xóa metadata
        try:
            deletion_results["metadata_deleted"] = delete_metadata(doc_id)
        except Exception as e:
            logger.error(f"Error deleting metadata: {str(e)}")
        
        # Kiểm tra kết quả
        if all(deletion_results.values()):
            logger.info(f"Successfully deleted document: {doc_id}")
            return {
                "message": "Document deleted successfully",
                "_id": doc_id,
                "file_type": file_type,
                "filename": filename,
                "deletion_results": deletion_results
            }
        else:
            logger.warning(f"Partial deletion for document: {doc_id}, results: {deletion_results}")
            return {
                "message": "Document partially deleted",
                "_id": doc_id,
                "file_type": file_type,
                "filename": filename,
                "deletion_results": deletion_results,
                "warning": "Some components could not be deleted"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/documents/list")
async def list_documents(file_type: str = None, limit: int = 100, skip: int = 0):
    """
    API endpoint để lấy danh sách tài liệu.
    
    Args:
        file_type: Lọc theo loại file (optional)
        limit: Số lượng documents trả về (default: 100)
        skip: Số lượng documents bỏ qua (default: 0)
    """
    try:
        documents = []
        
        # Thử lấy từ MongoDB trước
        try:
            client = MongoClient(Config.DATABASE_URL)
            db = client["faiss_db"]
            collection = db["metadata"]
            
            # Tạo filter
            filter_dict = {}
            if file_type:
                filter_dict["file_type"] = file_type
            
            # Lấy documents với pagination
            cursor = collection.find(filter_dict).skip(skip).limit(limit).sort("createdAt", -1)
            documents = list(cursor)
            client.close()
            
            if documents:
                logger.info(f"Retrieved {len(documents)} documents from MongoDB")
                return {
                    "documents": documents,
                    "total": len(documents),
                    "source": "mongodb"
                }
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
        
        # Fallback: Lấy từ metadata.json files
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
                    
                    # Filter theo file_type nếu có
                    if file_type:
                        metadata_list = [item for item in metadata_list if item.get('file_type') == file_type]
                    
                    documents.extend(metadata_list)
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")
        
        # Sort theo createdAt (newest first) và apply pagination
        documents.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        total = len(documents)
        documents = documents[skip:skip + limit]
        
        logger.info(f"Retrieved {len(documents)} documents from JSON files")
        return {
            "documents": documents,
            "total": total,
            "source": "json",
            "showing": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)