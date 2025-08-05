import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    
    # Model paths
    MODEL_EMBEDDING: str = "model/vinallama-7b-chat_q5_0.gguf"
    MODEL_PADDLEOCR: str = "model/.paddlex"
    
    # Data paths
    DATA_PATH: str = "Root_Folder"
    
    # Database
    DATABASE_URL: str = "mongodb://localhost:27017/"
    MONGODB_DATABASE: str = "faiss_db"
    MONGODB_COLLECTION: str = "metadata"
    
    # Embedding settings
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "models/embedding-001"
    
    # OCR settings
    OCR_LANGUAGE: str = "vi"
    OCR_USE_TEXTLINE_ORIENTATION: bool = True
    
    # Search settings
    DEFAULT_SEARCH_K: int = 5
    MAX_SEARCH_K: int = 100
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()