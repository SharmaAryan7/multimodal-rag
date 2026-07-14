import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize the LangChain Google embeddings model.
# This automatically reads the GOOGLE_API_KEY environment variable configured in Railway.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

def get_embedding(text: str) -> list[float]:
    """
    Takes a string and returns a vector embedding.
    Replaces the old ollama.embed() implementation and eliminates the 404 error.
    """
    # LangChain's embed_query returns a list of floats directly,
    # which is exactly the format ChromaDB expects.
    return embedding_model.embed_query(text)