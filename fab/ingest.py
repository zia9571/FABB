import os
import json
from pydantic import BaseModel
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings

DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

class DocumentMetadata(BaseModel):
    """Pydantic model for structured metadata."""
    source: str
    year: int
    quarter: str
    report_type: str
    financial_section: str = "General"

def extract_metadata_from_filename(file_path: str) -> Dict[str, Any]:
    """
    Extracts structured metadata (Year, Quarter, Report Type) from the filename.
    
    Filenames MUST follow a strict convention, e.g., 'FAB_Annual_2023.pdf' or 'FAB_Q3_2024_Earnings.pdf'.
    This is CRITICAL for temporal reasoning later.
    """
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    
    year = next((int(p) for p in parts if p.isdigit() and len(p) == 4), 0)
    
    if 'Annual' in parts:
        report_type = 'Annual Report'
        quarter = 'Q4'
    elif 'Q' in filename:
        quarter_match = next((p for p in parts if p.startswith('Q') and len(p) == 2 and p[1].isdigit()), 'Unknown')
        report_type = 'Quarterly Report'
        quarter = quarter_match
    else:
        quarter = 'Unknown'
        report_type = 'General'

    return DocumentMetadata(
        source=filename,
        year=year,
        quarter=quarter,
        report_type=report_type
    ).dict()

def process_documents(data_dir: str) -> List[Dict[str, Any]]:
    """Loads, parses, and chunks documents, assigning metadata."""
    all_chunks = []
    
    print(f"Starting document processing in {data_dir}...")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            base_metadata = extract_metadata_from_filename(file_path)
            print(f"Processing {filename} with metadata: {base_metadata}")

            elements = partition(filename=file_path, strategy="auto", 
                                 chunking_strategy="by_title")
            
            raw_text = "\n\n".join([str(el) for el in elements])
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=200, 
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", " ", ""] # Prioritize section breaks
            )
            
            chunks = text_splitter.create_documents(
                texts=[raw_text],
                metadatas=[base_metadata]
            )
            
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"{filename}_{i}"
                chunk.metadata['text_snippet'] = chunk.page_content[:100].replace('\n', ' ')
                all_chunks.append(chunk)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks

def ingest_data(chunks: List[Dict[str, Any]]):
    """Initializes embeddings and stores data in ChromaDB."""
    
    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating Chroma vector store at {CHROMA_PATH}...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print("Ingestion complete. Vector database built successfully.")
    
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: The '{DATA_DIR}' directory does not exist.")
        print("Please create it and place your FAB PDF reports inside.")
    else:
        document_chunks = process_documents(DATA_DIR)
        
        if document_chunks:
            ingest_data(document_chunks)
        else:
            print("No documents were processed. Check your PDFs and file paths.")