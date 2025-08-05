import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import (
    VectorSearchRequest, VectorSearchResponse, DocumentInfo,
    FileTypesResponse, FileTypeInfo, UploadFileRequest
)
from app.services.document_service import DocumentService
from app.services.search_service import SearchService
from app.config.settings import settings
from app.config.constants import Constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Document Vector API",
    description="API for document embedding and vector search",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure environment
try:
    os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
    logger.info("GOOGLE_API_KEY configured successfully")
except Exception as e:
    logger.error(f"Config error: {str(e)}")
    raise HTTPException(status_code=500, detail="Configuration error: Missing GEMINI_API_KEY")

# Initialize services
document_service = DocumentService()
search_service = SearchService()

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Document Vector API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "timestamp": settings.GEMINI_API_KEY[:10] + "..." if settings.GEMINI_API_KEY else "not_configured"
    }

@app.post("/documents/vector/add", response_model=dict, tags=["Documents"])
async def add_vector_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form(...),
    file_type: str = Form(...),
    role_user: str = Form(default="[]"),
    role_subject: str = Form(default="[]")
):
    """
    Add document with vector embedding
    
    - **file**: Document file to upload
    - **uploaded_by**: User who uploaded the document
    - **file_type**: Type of document (public, student, teacher, admin)
    - **role_user**: JSON array of user roles (optional)
    - **role_subject**: JSON array of subject roles (optional)
    """
    request = UploadFileRequest(
        uploaded_by=uploaded_by,
        file_type=file_type,
        role_user=role_user,
        role_subject=role_subject
    )
    return document_service.add_document(file, request)

@app.get("/documents/vector/{doc_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_vector_document(doc_id: str):
    """
    Get document information by ID
    
    - **doc_id**: Document ID to retrieve
    """
    return document_service.get_document(doc_id)

@app.delete("/documents/vector/{doc_id}", response_model=dict, tags=["Documents"])
async def delete_vector_document(doc_id: str):
    """
    Delete document by ID
    
    - **doc_id**: Document ID to delete
    """
    return document_service.delete_document(doc_id)

@app.get("/documents/list", tags=["Documents"])
async def list_documents(file_type: str = None, limit: int = 100, skip: int = 0):
    """
    List documents with filtering and pagination
    
    - **file_type**: Filter by document type (optional)
    - **limit**: Maximum number of documents to return (1-1000)
    - **skip**: Number of documents to skip for pagination
    """
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    return document_service.list_documents(file_type, limit, skip)

@app.get("/documents/types", response_model=FileTypesResponse, tags=["Configuration"])
async def get_file_types():
    """Get supported file types"""
    file_types = [
        FileTypeInfo(**info) for info in Constants.FILE_TYPE_DESCRIPTIONS.values()
    ]
    return FileTypesResponse(file_types=file_types)

@app.post("/documents/vector/search", response_model=VectorSearchResponse, tags=["Search"])
async def search_vector_documents(request: VectorSearchRequest):
    """
    Search documents using vector similarity
    
    - **query**: Search query text
    - **k**: Number of results to return (1-100)
    - **file_type**: Type of documents to search in
    - **similarity_threshold**: Minimum similarity threshold (0.0-1.0)
    """
    return search_service.search_vectors(request)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )