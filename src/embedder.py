import gc
import os

import chromadb
import google.generativeai as genai

from src.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_embedding(text):
    """Generate embeddings using Gemini."""
    response = genai.embed_content(
        model="text-embedding-004",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )

    return response["embedding"]


class VectorStoreRepository:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_items(self, ids, documents, embeddings, metadatas):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, n_results=TOP_K_RESULTS):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def count(self):
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


repo = VectorStoreRepository()


def store_chunks(text_chunks, captioned_images):
    all_items = []
    counter = 0
    seen_texts = set()

    for chunk in text_chunks:

        key = hash(chunk["text"].strip().lower())

        if key in seen_texts:
            continue

        seen_texts.add(key)

        all_items.append({
            "id": f"chunk_{counter}",
            "text": chunk["text"],
            "metadata": {
                "content_type": chunk["content_type"],
                "chunk_type": chunk["content_type"],
                "page": chunk["page"],
                "source": chunk["source_file"],
                "visual_content_path": ""
            }
        })

        counter += 1

    for img in captioned_images:

        key = hash(img["text"].strip().lower())

        if key in seen_texts:
            continue

        seen_texts.add(key)

        all_items.append({
            "id": f"chunk_{counter}",
            "text": img["text"],
            "metadata": {
                "content_type": img["content_type"],
                "chunk_type": img["content_type"],
                "page": img["page"],
                "source": img["source_file"],
                "visual_content_path": img["path"]
            }
        })

        counter += 1

    batch_size = 10

    for start in range(0, len(all_items), batch_size):

        batch = all_items[start:start + batch_size]

        ids = [x["id"] for x in batch]
        docs = [x["text"] for x in batch]
        metas = [x["metadata"] for x in batch]

        embeddings = [get_embedding(x) for x in docs]

        repo.add_items(ids, docs, embeddings, metas)

        del embeddings

    gc.collect()

    print(f"Stored {repo.count()} chunks.")


def query_collection(query_text, n_results=TOP_K_RESULTS):
    embedding = get_embedding(query_text)

    return repo.query(embedding, n_results)