import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def test_query():
    if not os.path.exists(CHROMA_PATH):
        print(" ERROR: ChromaDB folder not found. Did ingestion run?")
        return
    
    print("Loading embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print("Loading Chroma DB...")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print(" Running a test search...")
    query = "What is FAB's net profit in 2023?"
    results = db.similarity_search(query, k=3)

    print(f"\nRetrieved {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"--- RESULT {i+1} ---")
        print("Content:", r.page_content[:300], "...")
        print("Metadata:", r.metadata)
        print()

if __name__ == "__main__":
    test_query()


