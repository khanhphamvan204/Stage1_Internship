import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from fastapi import HTTPException, UploadFile
from app.models.schemas import (
    AddVectorRequest, DocumentInfo, DeleteResult, 
    DocumentListResponse, UploadFileRequest
)
from app.services.metadata_service import MetadataService
from app.services.file_service import FileService
from app.services.embedding_service import EmbeddingService
from app.config.constants import Constants
from app.config.paths import path_manager

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.metadata_service = MetadataService()
        self.file_service = FileService()
        self.embedding_service = EmbeddingService()
        self.path_manager = path_manager
    
    def add_document(self, file: UploadFile, request: UploadFileRequest) -> Dict:
        """Add new document with file upload, metadata saving, and embedding"""
        try:
            # Validate file type
            if request.file_type not in self.path_manager.valid_file_types:
                raise HTTPException(
                    status_code=400,
                    detail=Constants.ERROR_MESSAGES["INVALID_FILE_TYPE"].format(
                        file_types=self.path_manager.valid_file_types
                    )
                )
            
            # Validate file extension
            if not self.file_service.validate_file_extension(file.filename):
                file_extension = self.file_service.get_file_extension(file.filename)
                raise HTTPException(
                    status_code=400,
                    detail=Constants.ERROR_MESSAGES["UNSUPPORTED_FILE_FORMAT"].format(
                        extension=file_extension
                    )
                )
            
            # Generate metadata
            generated_id = str(uuid.uuid4())
            vietnam_tz = timezone(timedelta(hours=7))
            created_at = datetime.now(vietnam_tz).isoformat()
            
            # Save file
            file_path = self.file_service.save_uploaded_file(file.file, request.file_type, file.filename)
            
            # Parse role data
            try:
                role = {
                    "user": json.loads(request.role_user),
                    "subject": json.loads(request.role_subject)
                }
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail="Invalid JSON format for role_user or role_subject")
            
            # Create metadata
            metadata = AddVectorRequest(
                _id=generated_id,
                filename=file.filename,
                url=file_path,
                uploaded_by=request.uploaded_by,
                role=role,
                file_type=request.file_type,
                createdAt=created_at
            )
            
            # Save metadata
            if not self.metadata_service.save_metadata(metadata):
                raise HTTPException(
                    status_code=500,
                    detail=Constants.ERROR_MESSAGES["METADATA_SAVE_FAILED"].format(error="Database error")
                )
            
            # Create embeddings
            if not self.embedding_service.add_to_embedding(file_path, metadata):
                raise HTTPException(
                    status_code=500,
                    detail=Constants.ERROR_MESSAGES["EMBEDDING_FAILED"].format(error="Embedding creation failed")
                )
            
            logger.info(f"Successfully added document: {generated_id}")
            return {
                "message": Constants.SUCCESS_MESSAGES["DOCUMENT_ADDED"],
                "_id": generated_id,
                "file_type": request.file_type,
                "filename": file.filename
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    def get_document(self, doc_id: str) -> DocumentInfo:
        """Get document information by ID"""
        try:
            # Find document metadata
            doc_info = self.metadata_service.find_document(doc_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=Constants.ERROR_MESSAGES["DOCUMENT_NOT_FOUND"].format(doc_id=doc_id)
                )
            
            # Get file information
            file_path = doc_info.get('url')
            file_info = self.file_service.get_file_info(file_path)
            
            # Check vector database
            vector_exists = self.file_service.check_vector_exists(doc_info.get('file_type'))
            
            # Create response with additional info
            doc_info.update({
                "file_exists": file_info["exists"],
                "vector_exists": vector_exists,
                "file_size": file_info["size"]
            })
            
            return DocumentInfo(**doc_info)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")
    
    def delete_document(self, doc_id: str) -> Dict:
        """Delete document including file, metadata, and vector embeddings"""
        try:
            # Find document info
            doc_info = self.metadata_service.find_document(doc_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404,
                    detail=Constants.ERROR_MESSAGES["DOCUMENT_NOT_FOUND"].format(doc_id=doc_id)
                )
            
            file_type = doc_info.get('file_type')
            filename = doc_info.get('filename')
            file_path = doc_info.get('url')
            
            deletion_results = DeleteResult(
                file_deleted=False,
                metadata_deleted=False,
                vector_deleted=False
            )
            
            # Delete physical file
            deletion_results.file_deleted = self.file_service.delete_file(file_path)
            
            # Delete from vector database
            try:
                vector_db_path = self.path_manager.get_vector_path(file_type)
                deletion_results.vector_deleted = self.embedding_service.delete_from_faiss_index(vector_db_path, doc_id)
            except Exception as e:
                logger.error(f"Error deleting from vector database: {str(e)}")
            
            # Delete metadata
            deletion_results.metadata_deleted = self.metadata_service.delete_metadata(doc_id)
            
            # Check results
            if all([deletion_results.file_deleted, deletion_results.metadata_deleted, deletion_results.vector_deleted]):
                logger.info(f"Successfully deleted document: {doc_id}")
                return {
                    "message": Constants.SUCCESS_MESSAGES["DOCUMENT_DELETED"],
                    "_id": doc_id,
                    "file_type": file_type,
                    "filename": filename,
                    "deletion_results": deletion_results.dict()
                }
            else:
                logger.warning(f"Partial deletion for document: {doc_id}")
                return {
                    "message": "Document partially deleted",
                    "_id": doc_id,
                    "file_type": file_type,
                    "filename": filename,
                    "deletion_results": deletion_results.dict(),
                    "warning": "Some components could not be deleted"
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
    def list_documents(self, file_type: Optional[str] = None, limit: int = 100, skip: int = 0) -> DocumentListResponse:
        """List documents with filtering and pagination"""
        try:
            # Validate file_type if provided
            if file_type and file_type not in self.path_manager.valid_file_types:
                raise HTTPException(
                    status_code=400,
                    detail=Constants.ERROR_MESSAGES["INVALID_FILE_TYPE"].format(
                        file_types=self.path_manager.valid_file_types
                    )
                )
            
            # Get documents from metadata service
            result = self.metadata_service.list_documents(file_type, limit, skip)
            
            # Convert to DocumentInfo objects
            documents = [DocumentInfo(**doc) for doc in result["documents"]]
            
            return DocumentListResponse(
                documents=documents,
                total=result["total"],
                source=result["source"],
                showing=result.get("showing")
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")
