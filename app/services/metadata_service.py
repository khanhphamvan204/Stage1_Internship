import json
import os
import logging
from typing import Optional, List, Dict
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.models.schemas import AddVectorRequest, DocumentInfo
from app.config.settings import settings
from app.config.paths import path_manager

logger = logging.getLogger(__name__)

class MetadataService:
    def __init__(self):
        self.database_url = settings.DATABASE_URL
        self.database_name = settings.MONGODB_DATABASE
        self.collection_name = settings.MONGODB_COLLECTION
        self.path_manager = path_manager
    
    def save_metadata(self, metadata: AddVectorRequest) -> bool:
        """Save metadata to MongoDB with JSON fallback"""
        # Try MongoDB first
        if self._save_to_mongodb(metadata):
            return True
        
        # Fallback to JSON
        return self._save_to_json(metadata)
    
    def _save_to_mongodb(self, metadata: AddVectorRequest) -> bool:
        """Save to MongoDB"""
        try:
            client = MongoClient(self.database_url)
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            metadata_dict = metadata.dict(by_alias=True)
            collection.insert_one(metadata_dict)
            
            logger.info(f"Successfully saved metadata to MongoDB for _id: {metadata.id}")
            client.close()
            return True
        except PyMongoError as e:
            logger.error(f"Failed to save metadata to MongoDB: {str(e)}")
            return False
    
    def _save_to_json(self, metadata: AddVectorRequest) -> bool:
        """Save to JSON file"""
        try:
            _, vector_db_path = self.path_manager.get_paths(metadata.file_type, metadata.filename)
            metadata_file = os.path.join(vector_db_path, "metadata.json")
            
            existing_metadata = self._load_json_metadata(metadata_file)
            metadata_dict = metadata.dict(by_alias=True)
            existing_metadata.append(metadata_dict)
            
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Fallback: Successfully saved metadata to {metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to JSON file: {str(e)}")
            return False
    
    def _load_json_metadata(self, metadata_file: str) -> List[Dict]:
        """Load existing metadata from JSON file"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"No existing metadata found at {metadata_file}, creating new")
            return []
    
    def find_document(self, doc_id: str) -> Optional[Dict]:
        """Find document by ID in MongoDB or JSON files"""
        # Try MongoDB first
        doc_info = self._find_in_mongodb(doc_id)
        if doc_info:
            return doc_info
        
        # Fallback to JSON files
        return self._find_in_json_files(doc_id)
    
    def _find_in_mongodb(self, doc_id: str) -> Optional[Dict]:
        """Find document in MongoDB"""
        try:
            client = MongoClient(self.database_url)
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            doc_info = collection.find_one({"_id": doc_id})
            client.close()
            return doc_info
        except PyMongoError as e:
            logger.error(f"Failed to find document in MongoDB: {str(e)}")
            return None
    
    def _find_in_json_files(self, doc_id: str) -> Optional[Dict]:
        """Find document in JSON files"""
        for metadata_file in self.path_manager.get_all_metadata_paths():
            try:
                if os.path.exists(metadata_file):
                    metadata_list = self._load_json_metadata(metadata_file)
                    for item in metadata_list:
                        if item.get('_id') == doc_id:
                            return item
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")
        return None
    
    def delete_metadata(self, doc_id: str) -> bool:
        """Delete metadata from MongoDB and JSON files"""
        mongodb_success = self._delete_from_mongodb(doc_id)
        json_success = self._delete_from_json_files(doc_id)
        return mongodb_success or json_success
    
    def _delete_from_mongodb(self, doc_id: str) -> bool:
        """Delete from MongoDB"""
        try:
            client = MongoClient(self.database_url)
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            result = collection.delete_one({"_id": doc_id})
            client.close()
            
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted metadata from MongoDB for _id: {doc_id}")
                return True
            return False
        except PyMongoError as e:
            logger.error(f"Failed to delete metadata from MongoDB: {str(e)}")
            return False
    
    def _delete_from_json_files(self, doc_id: str) -> bool:
        """Delete from JSON files"""
        success = False
        for metadata_file in self.path_manager.get_all_metadata_paths():
            try:
                if os.path.exists(metadata_file):
                    metadata_list = self._load_json_metadata(metadata_file)
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
    
    def list_documents(self, file_type: Optional[str] = None, limit: int = 100, skip: int = 0) -> Dict:
        """List documents with pagination"""
        # Try MongoDB first
        documents = self._list_from_mongodb(file_type, limit, skip)
        if documents:
            return {
                "documents": documents,
                "total": len(documents),
                "source": "mongodb",
                "showing": len(documents)
            }
        
        # Fallback to JSON files
        return self._list_from_json_files(file_type, limit, skip)
    
    def _list_from_mongodb(self, file_type: Optional[str], limit: int, skip: int) -> List[Dict]:
        """List documents from MongoDB"""
        try:
            client = MongoClient(self.database_url)
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            filter_dict = {}
            if file_type:
                filter_dict["file_type"] = file_type
            
            cursor = collection.find(filter_dict).skip(skip).limit(limit).sort("createdAt", -1)
            documents = list(cursor)
            client.close()
            
            logger.info(f"Retrieved {len(documents)} documents from MongoDB")
            return documents
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
            return []
    
    def _list_from_json_files(self, file_type: Optional[str], limit: int, skip: int) -> Dict:
        """List documents from JSON files"""
        documents = []
        
        for metadata_file in self.path_manager.get_all_metadata_paths():
            try:
                if os.path.exists(metadata_file):
                    metadata_list = self._load_json_metadata(metadata_file)
                    
                    # Filter by file_type if specified
                    if file_type:
                        metadata_list = [item for item in metadata_list if item.get('file_type') == file_type]
                    
                    documents.extend(metadata_list)
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")
        
        # Sort by createdAt (newest first) and apply pagination
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