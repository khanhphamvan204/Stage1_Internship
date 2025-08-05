import os
import logging
import pandas as pd
import pdfplumber
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
)
from paddleocr import PaddleOCR
from app.models.schemas import AddVectorRequest
from app.config.settings import settings
from app.config.paths import path_manager

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.path_manager = path_manager
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize OCR
        try:
            self.ocr = PaddleOCR(
                lang=settings.OCR_LANGUAGE,
                use_textline_orientation=settings.OCR_USE_TEXTLINE_ORIENTATION
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            self.ocr = None
    
    def extract_text_with_paddle(self, file_path: str) -> str:
        """Extract text from file using PaddleOCR"""
        if not self.ocr:
            logger.error("PaddleOCR not initialized")
            return ""
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Extracting text with PaddleOCR from {file_path}")
            result = self.ocr.ocr(file_path)
            text = "\n".join([line[1][0] for line in result[0] if len(line) > 1]) if result and result[0] else ""
            logger.info(f"Extracted text length: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"OCR Error for {file_path}: {str(e)}")
            return ""
    
    def process_pdf(self, file_path: str) -> Tuple[List, List]:
        """Process PDF file, extract tables and text"""
        tables, texts = [], []
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Processing PDF: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract tables
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables or []:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                            tables.append(df)
                    
                    # Extract text
                    text = page.extract_text()
                    if not text:
                        # Fallback to OCR
                        full_text = self.extract_text_with_paddle(file_path)
                        if full_text:
                            texts.append(full_text)
                            logger.info(f"Used PaddleOCR for text extraction: {file_path}")
                            break
                    else:
                        texts.append(text)
            
            logger.info(f"Extracted {len(tables)} tables and {len(texts)} text segments")
        except Exception as e:
            logger.error(f"PDF Processing Error for {file_path}: {str(e)}")
        
        return tables, texts
    
    def load_documents(self, file_path: str, metadata: AddVectorRequest) -> List[Document]:
        """Load and process documents from file_path with metadata"""
        documents = []
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return documents

        extension = file_path.lower().split('.')[-1]
        supported_loaders = {
            'pdf': PyPDFLoader,
            'txt': TextLoader,
            'docx': Docx2txtLoader,
            'csv': CSVLoader,
            'xlsx': UnstructuredExcelLoader,
            'xls': UnstructuredExcelLoader
        }

        if extension in supported_loaders:
            try:
                logger.info(f"Loading document: {file_path} with extension {extension}")
                metadata_dict = metadata.dict(by_alias=True)
                
                if extension == 'pdf':
                    # Special handling for PDF
                    tables, texts = self.process_pdf(file_path)
                    
                    # Add tables as documents
                    for table in tables:
                        table_text = table.to_csv(index=False)
                        documents.append(Document(page_content=table_text, metadata=metadata_dict))
                    
                    # Add texts as documents
                    for text in texts:
                        documents.append(Document(page_content=text, metadata=metadata_dict))
                else:
                    # Standard document loading
                    loader = supported_loaders[extension](file_path)
                    loaded_docs = loader.load()
                    
                    for doc in loaded_docs:
                        documents.append(Document(page_content=doc.page_content, metadata=metadata_dict))
                
                logger.info(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        else:
            logger.warning(f"Unsupported file extension: {extension}")
        
        return documents
    
    def add_to_embedding(self, file_path: str, metadata: AddVectorRequest) -> bool:
        """Add document to FAISS vector store"""
        try:
            # Load documents
            documents = self.load_documents(file_path, metadata)
            if not documents:
                logger.warning(f"No documents loaded from {file_path}, skipping embedding.")
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks from {file_path}")

            if not chunks:
                logger.warning(f"No chunks created from {file_path}, skipping embedding.")
                return False

            # Get vector database path
            _, vector_db_path = self.path_manager.get_paths(metadata.file_type, metadata.filename)
            
            # Check if index exists
            index_exists = (os.path.exists(f"{vector_db_path}/index.faiss") and 
                           os.path.exists(f"{vector_db_path}/index.pkl"))
            
            if index_exists:
                logger.info(f"Loading existing FAISS index from {vector_db_path}")
                db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
                db.add_documents(chunks)
            else:
                logger.info(f"Creating new FAISS index at {vector_db_path}")
                db = FAISS.from_documents(chunks, self.embedding_model)
            
            # Save the index
            os.makedirs(vector_db_path, exist_ok=True)
            db.save_local(vector_db_path)
            logger.info(f"Successfully saved FAISS index to {vector_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing FAISS index: {str(e)}")
            return False
    
    def delete_from_faiss_index(self, vector_db_path: str, doc_id: str) -> bool:
        """Delete document from FAISS index by doc_id"""
        try:
            # Check if index exists
            index_path = f"{vector_db_path}/index.faiss"
            pkl_path = f"{vector_db_path}/index.pkl"
            
            if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
                logger.warning(f"FAISS index not found at {vector_db_path}")
                return True  # Consider as success if index doesn't exist
            
            # Load FAISS index
            db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
            
            # Find docstore_ids that match the doc_id
            docstore = db.docstore
            index_to_docstore_id = db.index_to_docstore_id
            ids_to_delete = []
            
            for index, docstore_id in index_to_docstore_id.items():
                doc = docstore.search(docstore_id)
                if doc and doc.metadata.get('_id') == doc_id:
                    ids_to_delete.append(docstore_id)
            
            # Delete documents if found
            if ids_to_delete:
                db.delete(ids=ids_to_delete)
                db.save_local(vector_db_path)
                logger.info(f"Successfully deleted {len(ids_to_delete)} documents with _id: {doc_id} from FAISS index at {vector_db_path}")
            else:
                logger.warning(f"No documents found with _id: {doc_id} in FAISS index")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from FAISS index: {str(e)}")
            return False
    
    def search_documents(self, query: str, file_type: str, k: int = 5) -> List[Document]:
        """Search documents in FAISS vector store"""
        try:
            vector_db_path = self.path_manager.get_vector_path(file_type)
            
            # Check if vector database exists
            if not (os.path.exists(f"{vector_db_path}/index.faiss") and 
                   os.path.exists(f"{vector_db_path}/index.pkl")):
                logger.warning(f"Vector database not found at {vector_db_path}")
                return []
            
            # Load database and search
            db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
            search_docs = db.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(search_docs)} documents for query: {query}")
            return search_docs
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return []
