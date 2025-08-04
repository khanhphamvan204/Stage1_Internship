from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "data")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "model/vinallama-7b-chat_q5_0.gguf")
    MODEL_PADDLEOCR = os.getenv("MODEL_PADDLEOCR", "model/.paddlex")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")