import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "./data_docs"
CHROMA_PATH = "./chroma_db"

def load_and_chunk():
    docs = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def create_vector_store():
    print("1. Loading and splitting documents...")
    chunks = load_and_chunk()
    print(f"   Created {len(chunks)} chunks.")

    print("2. Embedding chunks into ChromaDB via nomic-embed-text...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"   Vector store persisted at: {CHROMA_PATH}")
    return vector_store

def verify_retriever(vector_store):
    print("\n3. Testing Retriever (search_kwargs={'k': 3}) on numeric query...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Test query strictly targeting financial metrics
    test_query = "What was the revenue and profit in Q2 FY 2024-25?"
    print(f"   Query: '{test_query}'\n")
    
    results = retriever.invoke(test_query)
    for idx, doc in enumerate(results, 1):
        print(f"--- Result {idx} ---")
        print(doc.page_content.strip()[:350])
        print("--------------------\n")

if __name__ == "__main__":
    store = create_vector_store()
    verify_retriever(store)
    print("SUCCESS: Vector store created and retriever verified!")