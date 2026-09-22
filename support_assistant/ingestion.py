from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# When processing your files:
documents = []
metadatas = []
ids = []

for file_path in doc_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Split the document content into smaller chunks
    chunks = text_splitter.split_text(content)
    
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": file_path, "chunk_id": i})
        ids.append(f"{file_path}_chunk_{i}")

# Add the chunked documents to ChromaDB collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)