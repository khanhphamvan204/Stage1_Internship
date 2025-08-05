import os
import shutil
import logging
from typing import Dict
from app.config.settings import settings
from app.config.constants import Constants
from app.config.paths import path_manager

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.path_manager = path_manager
        self.supported_extensions = Constants.SUPPORTED_EXTENSIONS
    
    def validate_file_extension(self, filename: str) -> bool:
        """Validate if file extension is supported"""
        file_extension = os.path.splitext(filename.lower())[1]
        return file_extension in self.supported_extensions
    
    def get_file_extension(self, filename: str) -> str:
        """Get file extension"""
        return os.path.splitext(filename.lower())[1]
    
    def save_uploaded_file(self, file, file_type: str, filename: str) -> str:
        """Save uploaded file to appropriate directory"""
        file_path, _ = self.path_manager.get_paths(file_type, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        # Verify file was saved
        if not os.path.exists(file_path):
            raise Exception(f"Failed to save file: {file_path}")
        
        logger.info(f"File saved successfully: {file_path}")
        return file_path
    
    def delete_file(self, file_path: str) -> bool:
        """Delete physical file"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return True  # Consider as success if file doesn't exist
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict:
        """Get file information"""
        info = {"exists": False, "size": None}
        
        if file_path and os.path.exists(file_path):
            info["exists"] = True
            try:
                info["size"] = os.path.getsize(file_path)
            except Exception as e:
                logger.warning(f"Could not get file size for {file_path}: {str(e)}")
        
        return info
    
    def check_vector_exists(self, file_type: str) -> bool:
        """Check if vector database files exist"""
        try:
            vector_db_path = self.path_manager.get_vector_path(file_type)
            return (os.path.exists(f"{vector_db_path}/index.faiss") and 
                   os.path.exists(f"{vector_db_path}/index.pkl"))
        except Exception as e:
            logger.error(f"Error checking vector existence: {str(e)}")
            return False