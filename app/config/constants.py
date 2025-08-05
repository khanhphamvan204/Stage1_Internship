from enum import Enum
from typing import Dict, List

class FileType(str, Enum):
    PUBLIC = "public"
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class Constants:
    # File types
    SUPPORTED_FILE_TYPES = [ft.value for ft in FileType]
    
    # File extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
    
    # File type descriptions
    FILE_TYPE_DESCRIPTIONS = {
        FileType.PUBLIC.value: {
            "value": FileType.PUBLIC.value,
            "label": "Thông báo chung (Public)",
            "description": "Tài liệu công khai cho tất cả người dùng"
        },
        FileType.STUDENT.value: {
            "value": FileType.STUDENT.value,
            "label": "Sinh viên (Student)",
            "description": "Tài liệu dành cho sinh viên"
        },
        FileType.TEACHER.value: {
            "value": FileType.TEACHER.value,
            "label": "Giảng viên (Teacher)", 
            "description": "Tài liệu dành cho giảng viên"
        },
        FileType.ADMIN.value: {
            "value": FileType.ADMIN.value,
            "label": "Quản trị viên (Admin)",
            "description": "Tài liệu dành cho quản trị viên"
        }
    }
    
    # Response messages
    SUCCESS_MESSAGES = {
        "DOCUMENT_ADDED": "Vector added successfully",
        "DOCUMENT_DELETED": "Document deleted successfully", 
        "DOCUMENT_UPDATED": "Document updated successfully"
    }
    
    ERROR_MESSAGES = {
        "INVALID_FILE_TYPE": "Invalid file_type. Must be one of: {file_types}",
        "UNSUPPORTED_FILE_FORMAT": "File format {extension} not supported",
        "DOCUMENT_NOT_FOUND": "Document with ID {doc_id} not found",
        "FILE_SAVE_FAILED": "Failed to save file: {file_path}",
        "EMBEDDING_FAILED": "Failed to create embeddings: {error}",
        "METADATA_SAVE_FAILED": "Failed to save metadata: {error}"
    }