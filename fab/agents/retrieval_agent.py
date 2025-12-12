import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

class RetrievalAgent:
    def __init__(self, persist_dir=None, embed_model=None):
        self.persist_dir = persist_dir or CHROMA_PATH
        self.embed_model = embed_model or EMBED_MODEL
        self._init_db()

    def _init_db(self):
        # initialize embedding function then DB client
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.embed_model)
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)

    def retrieve(self, query: str, k: int = 6, source_filter: list | None = None):
        # perform similarity search, then optionally filter by source substrings
        docs = self.db.similarity_search(query, k=k)
        if source_filter:
            docs = [d for d in docs if any(sf.lower() in (d.metadata.get("source") or "").lower() for sf in source_filter)]
        # return list of dicts
        return [{
            "content": d.page_content,
            "metadata": d.metadata
        } for d in docs]
