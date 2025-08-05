# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from app.embedding import create_embedding_chunks
# from app.config import Config

# if __name__ == "__main__":
#     create_embedding_chunks()

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import Config


# Khởi tạo môi trường
os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY

def load_vector_db():
    """
    Load FAISS vector database từ đường dẫn đã lưu.
    
    Returns:
    - FAISS vector store hoặc None nếu có lỗi.
    """
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(Config.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        print(f"Đã load vector DB từ: {Config.VECTOR_DB_PATH}")
        return db
    except Exception as e:
        print(f"Lỗi khi load FAISS index: {e}")
        return None

def retrieve_with_title_filter(query, title_filter, k=5):
    """
    Truy vấn FAISS với bộ lọc theo title.
    
    Parameters:
    - query: Chuỗi truy vấn để tìm kiếm.
    - title_filter: Giá trị title cần lọc (chuỗi chính xác hoặc một phần).
    - k: Số lượng kết quả trả về.
    
    Returns:
    - Danh sách các tài liệu phù hợp với truy vấn và bộ lọc title.
    """
    # Load vector DB
    db = load_vector_db()
    if db is None:
        return []

    # Tạo bộ lọc metadata để lọc theo title
    def metadata_filter(metadata_dict):
        return title_filter.lower() in metadata_dict.get("uploaded_by", "").lower()

    # Thực hiện truy vấn với bộ lọc
    try:
        results = db.similarity_search(query, k=k, filter=metadata_filter)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in results
        ]
    except Exception as e:
        print(f"Lỗi khi thực hiện truy vấn: {e}")
        return []

# Ví dụ sử dụng
if __name__ == "__main__":
    query = "Thông tin"
    title_filter = "1323"  # Thay bằng title bạn muốn lọc
    results = retrieve_with_title_filter(query, title_filter, k=5)
    
    # In kết quả
    for i, result in enumerate(results):
        print(f"Kết quả {i+1}:")
        print(f"Title: {result['metadata']['uploaded_by']}")
        print(f"Content: {result['content'][:200]}...")  # In 200 ký tự đầu tiên
        print(f"Metadata: {result['metadata']}")
        print("-" * 50)

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from app.embedding import create_embedding_chunks
# from app.config import Config
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS

# # Khởi tạo môi trường
# os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY

# def delete_chunks_by_doc_id(doc_id):
#     """
#     Xóa các chunk trong FAISS index dựa trên doc_id.

#     Parameters:
#     - doc_id: ID của tài liệu cần xóa (khớp với trường 'id' trong metadata).

#     Returns:
#     - True nếu xóa thành công, False nếu có lỗi hoặc không tìm thấy chunk.
#     """
#     try:
#         # Khởi tạo embedding model
#         embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
#         # Load FAISS index
#         db = FAISS.load_local(Config.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
#         print(f"Đã load vector DB từ: {Config.VECTOR_DB_PATH}")

#         # Lấy tất cả tài liệu từ docstore
#         documents = db.docstore._dict
#         ids_to_delete = []

#         # Tìm các ID nội bộ tương ứng với doc_id
#         for doc_id_in_index, doc in documents.items():
#             if doc.metadata.get("id", "") == doc_id:
#                 ids_to_delete.append(doc_id_in_index)

#         if not ids_to_delete:
#             print(f"Không tìm thấy chunk nào với doc_id: {doc_id}")
#             return False

#         # Xóa các chunk bằng phương thức delete
#         db.delete(ids=ids_to_delete)
        
#         # Lưu lại index
#         db.save_local(Config.VECTOR_DB_PATH)
#         print(f"Đã xóa {len(ids_to_delete)} chunk với doc_id: {doc_id}")
#         return True
#     except Exception as e:
#         print(f"Lỗi khi xóa chunk: {e}")
#         return False

# # Ví dụ sử dụng
# if __name__ == "__main__":
#     doc_id_to_delete = "doc_001"  # Thay bằng doc_id thực tế
#     delete_chunks_by_doc_id(doc_id_to_delete)