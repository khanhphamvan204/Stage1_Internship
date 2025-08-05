from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class AddVectorRequest(BaseModel):
    id: str = Field(alias="_id")
    filename: str
    url: str
    uploaded_by: str
    role: Dict[str, List[str]]
    file_type: str 
    createdAt: str
    
    class Config:
        allow_population_by_field_name = True

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Câu truy vấn tìm kiếm")
    k: int = Field(default=5, ge=1, le=100, description="Số lượng kết quả trả về (1-100)")
    file_type: str = Field(..., description="Loại tài liệu (public, student, teacher, admin)")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Ngưỡng độ tương quan (0.0-1.0)")

class SearchResult(BaseModel):
    content: str
    metadata: Dict

class VectorSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    k_requested: int
    file_type: str
    similarity_threshold: float
    search_time_ms: float

class DocumentInfo(BaseModel):
    id: str = Field(alias="_id")
    filename: str
    url: str
    uploaded_by: str
    role: Dict[str, List[str]]
    file_type: str
    createdAt: str
    file_exists: Optional[bool] = None
    vector_exists: Optional[bool] = None
    file_size: Optional[int] = None
    
    class Config:
        allow_population_by_field_name = True

class DeleteResult(BaseModel):
    file_deleted: bool
    metadata_deleted: bool
    vector_deleted: bool

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
    source: str
    showing: Optional[int] = None

class FileTypeInfo(BaseModel):
    value: str
    label: str
    description: str

class FileTypesResponse(BaseModel):
    file_types: List[FileTypeInfo]

class UploadFileRequest(BaseModel):
    uploaded_by: str
    file_type: str
    role_user: str = "[]"
    role_subject: str = "[]"