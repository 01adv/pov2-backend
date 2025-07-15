
# import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

logger.info("Running tests...")
# def test_collection_exists():
#     client = chromadb.PersistentClient(path="./chroma_db")
#     collections = client.list_collections()
#     assert any(col.name == "products" for col in collections), "❌ 'products' collection does not exist."
#     logger.info("✅ 'products' collection exists.")

def test_embedding_and_query():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="products")

    test_query = "summer vibe tops"
    # test_filter = {"category": "pant"}  # static filter

    embedding_function = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    
    # Embed the query
    query_embedding = embedding_function.embed_query(test_query)
    # logger.info(f"🔍 Query embedding generated for: '{test_query}'")

    # Query ChromaDB with metadata included
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        # where= test_filter or {},
        include=["metadatas", "documents"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if documents and metadatas:
        logger.info(f"✅ Found {len(documents)} result(s) for: '{test_query}' with filter ")
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            logger.info(f"{i+1}. Document: {doc[:100]}... | Category: {meta.get('category', 'N/A')} | Product: {meta.get('product_name', 'N/A')}")
    else:
        logger.warning("❌ No results found for the test query with filters.")


def test_all():
    # test_collection_exists()
    test_embedding_and_query()


if __name__ == "__main__":
    test_all()