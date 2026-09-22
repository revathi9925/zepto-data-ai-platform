import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_documents():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="zepto_support")

    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} not found.")
        return

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            # Split document into chunks
            chunks = text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_id": i})
                ids.append(f"{filename}_chunk_{i}")

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully ingested {len(documents)} chunks into ChromaDB.")
    else:
        print("No documents found to ingest.")

if __name__ == "__main__":
    ingest_documents()