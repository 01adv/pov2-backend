from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
load_dotenv()


chroma_client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(anonymized_telemetry=False)
)
openai_client = OpenAI()

# Function to generate embedding for a given text using OpenAI API


def generate_embedding(text: str) -> list:
    try:
        response = openai_client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception:
        raise


def add_document(collection, text: str, metadata: dict):
    try:
        embedding = generate_embedding(text)
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
            ids=[metadata["id"]],
        )
    except Exception:
        raise


def query_collection(collection: str, query_text: str, n_results: int = 10):
    try:
        collection_instance = chroma_client.get_collection(name=collection)
        query_embedding = generate_embedding(query_text)
        results = collection_instance.query(
            query_embeddings=[query_embedding], n_results=n_results
        )
        return results
    except Exception:
        raise
