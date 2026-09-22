try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
DOCS_DIR = "./docs"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def init_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="zepto_policies",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    if collection.count() == 0:
        doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
        documents = []
        metadatas = []
        ids = []
        
        for file_path in doc_files:
            doc_id = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if content:
                documents.append(content)
                metadatas.append({"source": doc_id})
                ids.append(doc_id)
        
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Successfully indexed {len(documents)} document(s) into ChromaDB.")
        else:
            print(f"Warning: No valid .txt documents found in '{DOCS_DIR}'.")
    
    return collection

def query_vector_store(query_text: str, top_k: int = 3):
    collection = init_vector_store()
    
    if collection.count() == 0:
        print("Warning: Vector store is empty. Returning no chunks.")
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    retrieved_chunks = []
    if results and results.get("ids") and results["ids"][0]:
        ids = results["ids"][0]
        documents = results["documents"][0]
        
        for doc_id, doc_text in zip(ids, documents):
            retrieved_chunks.append({"id": doc_id, "text": doc_text})
        
    return retrieved_chunks