import os
from typing import List, NamedTuple
from config.settings import settings
from config.constants import FileType

class PathInfo(NamedTuple):
    file_folder: str
    vector_folder: str

class PathManager:
    def __init__(self, base_path: str = None):
        self.base_path = base_path or settings.DATA_PATH
        self._path_mapping = {
            FileType.PUBLIC: PathInfo(
                f"{self.base_path}/Public_Rag_Info/File_Folder",
                f"{self.base_path}/Public_Rag_Info/Faiss_Folder"
            ),
            FileType.STUDENT: PathInfo(
                f"{self.base_path}/Student_Rag_Info/File_Folder",
                f"{self.base_path}/Student_Rag_Info/Faiss_Folder"
            ),
            FileType.TEACHER: PathInfo(
                f"{self.base_path}/Teacher_Rag_Info/File_Folder",
                f"{self.base_path}/Teacher_Rag_Info/Faiss_Folder"
            ),
            FileType.ADMIN: PathInfo(
                f"{self.base_path}/Admin_Rag_Info/File_Folder",
                f"{self.base_path}/Admin_Rag_Info/Faiss_Folder"
            )
        }
    
    def get_paths(self, file_type: str, filename: str) -> tuple[str, str]:
        """Get file path and vector database path"""
        if file_type not in [ft.value for ft in FileType]:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of: {[ft.value for ft in FileType]}")
        
        path_info = self._path_mapping[FileType(file_type)]
        file_path = os.path.join(path_info.file_folder, filename).replace("\\", "/")
        return file_path, path_info.vector_folder
    
    def get_all_metadata_paths(self) -> List[str]:
        """Get all metadata.json file paths"""
        return [
            os.path.join(path_info.vector_folder, "metadata.json")
            for path_info in self._path_mapping.values()
        ]
    
    def get_vector_path(self, file_type: str) -> str:
        """Get vector database path for specific file type"""
        if file_type not in [ft.value for ft in FileType]:
            raise ValueError(f"Invalid file_type: {file_type}")
        return self._path_mapping[FileType(file_type)].vector_folder
    
    @property
    def valid_file_types(self) -> List[str]:
        return [ft.value for ft in FileType]

# Global path manager instance
path_manager = PathManager()