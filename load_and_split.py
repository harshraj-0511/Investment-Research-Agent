import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "./data_docs"

def load_documents():
    docs = []
    if not os.path.exists(DOCS_DIR):
        print(f"Directory {DOCS_DIR} not found.")
        return docs

    for file in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, file)
        if file.endswith(".docx"):
            print(f"Loading Word Doc: {file}...")
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            print(f"Loading PDF: {file}...")
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            print(f"Loading Text: {file}...")
            loader = TextLoader(file_path)
            docs.extend(loader.load())

    return docs

def split_documents(docs):
    # Chunk size and overlap tuned so tabular metrics remain with headers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    raw_documents = load_documents()
    print(f"Loaded {len(raw_documents)} raw document(s).")
    if raw_documents:
        chunks = split_documents(raw_documents)
        print(f"Split into {len(chunks)} chunks.")
        print("\n--- Preview of Chunk 1 ---")
        print(chunks[0].page_content[:300])
        print("--------------------------")
        print("SUCCESS: Ready for vector store indexing!")
    else:
        print("No documents found in data_docs/. Ensure your .docx or .pdf is inside data_docs/.")