import time
import logging
from typing import List
from app.models.schemas import VectorSearchRequest, VectorSearchResponse, SearchResult
from app.services.embedding_service import EmbeddingService
from app.config.constants import Constants

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Perform vector search with timing"""
        start_time = time.time()
        
        try:
            # Validate file_type
            if request.file_type not in Constants.SUPPORTED_FILE_TYPES:
                raise ValueError(f"Invalid file_type. Must be one of: {Constants.SUPPORTED_FILE_TYPES}")
            
            # Perform search
            search_docs = self.embedding_service.search_documents(
                query=request.query,
                file_type=request.file_type,
                k=request.k
            )
            
            # Convert to SearchResult objects
            results = [
                SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in search_docs
            ]
            
            search_time_ms = round((time.time() - start_time) * 1000, 2)
            
            response = VectorSearchResponse(
                query=request.query,
                results=results,
                total_found=len(results),
                k_requested=request.k,
                file_type=request.file_type,
                similarity_threshold=request.similarity_threshold,
                search_time_ms=search_time_ms
            )
            
            logger.info(f"Search completed in {search_time_ms}ms, returned {len(results)} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise