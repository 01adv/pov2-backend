
# import pandas as pd
# from langchain_community.embeddings import OpenAIEmbeddings
# import chromadb
# import os
# from dotenv import load_dotenv
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
# logger = logging.getLogger(__name__)

# load_dotenv()

# def embed_products():
#     # Initialize ChromaDB client
#     client = chromadb.PersistentClient(path="./chroma_db")
#     # Reset collection to ensure fresh data
#     try:
#         client.delete_collection(name="products")
#         logger.info("Existing 'products' collection deleted.")
#     except Exception as e:
#         logger.info("No existing 'products' collection to delete or error during deletion.")
        
#     collection = client.get_or_create_collection(name="products")

#     # Initialize OpenAI embedding function
#     embedding_function = OpenAIEmbeddings(
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small"
#     )

#     # Read products from CSV
#     df = pd.read_csv("new-cleaned-products.csv")
#     logger.info(f"Loaded {len(df)} products from new-cleaned-products.csv")

#     # Validate required columns
#     required_columns = ["product-name", "category", "price", "product-description"]
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         logger.error(f"Missing columns in new-cleaned-products.csv: {missing_columns}")
#         return

#     # Prepare documents and metadata
#     documents = []
#     metadatas = []
#     ids = []

#     for idx, row in df.iterrows():
#         # Combine relevant fields for embedding, converting to lowercase
#         text_parts = [
#             str(row.get('product-name', '')),
#             str(row.get('product-description', '')),
#             str(row.get('tags', '')),
#             str(row.get('vibe', '')),
#             str(row.get('color', '')),
#             str(row.get('size', '')),
#             str(row.get('by_line', '')),
#             str(row.get('bodyType', '')),
#             str(row.get('bottomlength', '')),
#             str(row.get('bottomshape', '')),
#             str(row.get('fittype', '')),
#             str(row.get('neckline', '')),
#             str(row.get('sleevelength', '')),
#             str(row.get('category', ''))
#         ]
#         text = ' '.join(part.lower() for part in text_parts if pd.notna(part) and str(part).strip() != '')
#         documents.append(text)
        
#         metadatas.append({
#             "product_name": str(row.get("product-name", "")).lower(),
#             "category": str(row.get("category", "")).lower(),
#             "tags": str(row.get("tags", "")).lower(),
#             "price": float(row.get("price", 0.0)),
#             "product_description": str(row.get("product-description", "")).lower(),
#             "seo_title": str(row.get("seo-title", "")).lower(),
#             "seo_description": str(row.get("seo-description", "")).lower(),
#             "vibe": str(row.get("vibe", "")).lower(),
#             "color": str(row.get("color", "")).lower(),
#             "size": str(row.get("size", "")).lower(),
#             "by_line": str(row.get("by_line", "")).lower(),
#             "bodyType": str(row.get("bodyType", "")).lower(),
#             "bottomlength": str(row.get("bottomlength", "")).lower(),
#             "bottomshape": str(row.get("bottomshape", "")).lower(),
#             "fittype": str(row.get("fittype", "")).lower(),
#             "neckline": str(row.get("neckline", "")).lower(),
#             "sleevelength": str(row.get("sleevelength", "")).lower(),
#         })
#         ids.append(str(idx))
#         logger.debug(f"Prepared product {idx}: {row.get('product-name', '')}, category: {row.get('category', '')}")

#     # Generate embeddings and store in ChromaDB
#     embeddings = embedding_function.embed_documents(documents)
#     collection.upsert(
#         documents=documents,
#         metadatas=metadatas,
#         ids=ids,
#         embeddings=embeddings
#     )
#     logger.info(f"Stored {len(documents)} product embeddings in ChromaDB.")

# if __name__ == "__main__":
#     embed_products()



import numpy as np
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv
import logging

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ─── Helper ─────────────────────────────────────────────────────────────────────
def normalize(vec):
    """Return unit‑length vector (for cosine similarity)."""
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)

# ─── Main ───────────────────────────────────────────────────────────────────────
def embed_products():
    client = chromadb.PersistentClient(path="./chroma_db")

    # reset collection
    try:
        client.delete_collection(name="products")
        logger.info("Existing 'products' collection deleted.")
    except Exception:
        logger.info("No existing 'products' collection to delete.")
    collection = client.get_or_create_collection(name="products")

    # OpenAI embeddings
    embedder = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )

    # load CSV
    df = pd.read_csv("new-cleaned-products.csv")
    logger.info(f"Loaded {len(df)} rows.")

    # ensure mandatory columns
    required = {"product-name", "category", "price", "product-description"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return

    documents, metadatas, ids = [], [], []

    for idx, row in df.iterrows():
        text_parts = [
            row.get('product-name', ''),
            row.get('product-description', ''),
            row.get('tags', ''),
            row.get('vibe', ''),
            row.get('color', ''),
            row.get('size', ''),
            row.get('by_line', ''),
            row.get('bodyType', ''),
            row.get('bottomlength', ''),
            row.get('bottomshape', ''),
            row.get('fittype', ''),
            row.get('neckline', ''),
            row.get('sleevelength', ''),
            row.get('category', '')
        ]
        doc = ' '.join(str(p).lower() for p in text_parts if pd.notna(p) and str(p).strip())
        documents.append(doc)

        metadatas.append({
            "product_name": str(row.get("product-name", "")).lower(),
            "category":     str(row.get("category", "")).lower(),
            "tags":         str(row.get("tags", "")).lower(),
            "price":        float(row.get("price", 0.0)),
            "product_description": str(row.get("product-description", "")).lower(),
            "seo_title":    str(row.get("seo-title", "")).lower(),
            "seo_description": str(row.get("seo-description", "")).lower(),
            "vibe":         str(row.get("vibe", "")).lower(),
            "color":        str(row.get("color", "")).lower(),
            "size":         str(row.get("size", "")).lower(),
            "by_line":      str(row.get("by_line", "")).lower(),
            "bodyType":     str(row.get("bodyType", "")).lower(),
            "bottomlength": str(row.get("bottomlength", "")).lower(),
            "bottomshape":  str(row.get("bottomshape", "")).lower(),
            "fittype":      str(row.get("fittype", "")).lower(),
            "neckline":     str(row.get("neckline", "")).lower(),
            "sleevelength": str(row.get("sleevelength", "")).lower()
        })
        ids.append(str(idx))

    # ── Embed & normalize ───────────────────────────────────────────────────────
    raw_embeddings = embedder.embed_documents(documents)
    norm_embeddings = [normalize(e).tolist() for e in raw_embeddings]

    # upsert
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=norm_embeddings
    )
    logger.info(f"Stored {len(documents)} normalized product embeddings in ChromaDB.")

if __name__ == "__main__":
    embed_products()
