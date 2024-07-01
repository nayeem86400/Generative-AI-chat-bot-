import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_db(data_path, db_faiss_path):
    if not any(fname.endswith('.pdf') for fname in os.listdir(data_path)):
        print("No .pdf files found in the data directory. Please upload a .pdf file.")
        return

    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    
    try:
        db.save_local(db_faiss_path)
        print("Vector database created and saved successfully.")
        return 1
    except :
        print("Vector database creation failed")
        return 0
    
    return db
if __name__ == "__main__":
    create_vector_db()
