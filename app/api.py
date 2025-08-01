import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from app.embedding import load_new_documents, add_to_embedding, AddVectorRequest
from app.config import Config
import shutil

# Khởi tạo FastAPI
app = FastAPI()

# Thêm CORS middleware để cho phép gọi API từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo môi trường
os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY

def save_metadata(metadata: AddVectorRequest, file_name: str):
    """
    Lưu metadata vào metadata.json.
    
    Parameters:
    - metadata: Metadata của tài liệu (AddVectorRequest).
    - file_name: Tên file để gắn với metadata.
    """
    metadata_file = os.path.join(Config.DATA_PATH, "metadata.json")
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    existing_metadata = []
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except json.JSONDecodeError:
            existing_metadata = []
        except Exception:
            existing_metadata = []
    
    new_metadata = {
        "file_name": file_name,
        "id": metadata.id,
        "title": metadata.title,
        "description": metadata.description,
        "uploader": metadata.uploader
    }
    existing_metadata.append(new_metadata)
    
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")

@app.post("/add", response_model=dict)
async def add_vector(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(...),
    uploader: str = Form(...)
):
    """
    API endpoint để thêm file và metadata từ client, tự động tạo id bằng UUID, thực hiện embedding vào FAISS.
    
    Parameters:
    - file: File upload (PDF, TXT, DOCX, CSV, XLSX).
    - title: Tiêu đề tài liệu.
    - description: Mô tả tài liệu.
    - uploader: Người upload.
    
    Returns:
    - JSON: {"message": "Vector added", "id": "<generated_uuid>"}
    """
    try:
        # Tạo ID tự động bằng UUID
        generated_id = str(uuid.uuid4())

        # Tạo metadata từ form data và UUID
        metadata = AddVectorRequest(id=generated_id, title=title, description=description, uploader=uploader)

        # Kiểm tra định dạng file
        supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file.filename.lower())[1]
        if file_extension not in supported_extensions:
            raise HTTPException(status_code=400, detail=f"File format {file_extension} not supported")

        # Lưu file trực tiếp vào thư mục data với tên gốc
        file_path = os.path.join(Config.DATA_PATH, file.filename)
        os.makedirs(Config.DATA_PATH, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Lưu metadata
        save_metadata(metadata, file.filename)

        # Thêm file mới vào FAISS index
        try:
            add_to_embedding(file_path, metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

        return {"message": "Vector added", "id": generated_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Thêm phần này để có thể chạy trực tiếp bằng F5
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)