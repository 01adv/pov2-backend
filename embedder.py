# import pandas as pd
# # from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
# import chromadb
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def embed_products():
#     # Initialize ChromaDB client
#     client = chromadb.PersistentClient(path="./chroma_db")
#     collection = client.get_or_create_collection(name="products")

#     # Initialize OpenAI embedding function
#     embedding_function = OpenAIEmbeddings(
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small"
#     )

#     # Read products from CSV
#     df = pd.read_csv("products.csv")

#     # Prepare documents and metadata
#     documents = []
#     metadatas = []
#     ids = []

#     for idx, row in df.iterrows():
#         # Combine relevant fields for embedding
#         text = f"{row['Product Name']}  {row['Product Description']} {row['Tags']}  {row['Vibe']} {row['Available Variants']} {row["Product Details"]}"
#         documents.append(text)
#         metadatas.append({
#             "product_name": row["Product Name"],
#             "category": row["Category"],
#             "tags": row["Tags"],
#             "price": float(row["Price"]),
#             "product_description": row["Product Description"],
#             "seo_title": row["SEO Title"],
#             "seo_description": row["SEO Description"],
#             "available_variants": row["Available Variants"],
#             "product_details": row["Product Details"],
#             "vibe": row["Vibe"]
#         })
#         ids.append(str(idx))

#     # Generate embeddings and store in ChromaDB
#     embeddings = embedding_function.embed_documents(documents)
#     collection.upsert(
#         documents=documents,
#         metadatas=metadatas,
#         ids=ids,
#         embeddings=embeddings
#     )
#     print("Product embeddings stored in ChromaDB.")

# if __name__ == "__main__":
#     embed_products()



import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def embed_products():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    # Reset collection to ensure fresh data
    client.delete_collection(name="products")
    collection = client.get_or_create_collection(name="products")

    # Initialize OpenAI embedding function
    embedding_function = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )

    # Read products from CSV
    df = pd.read_csv("products.csv")
    logger.info(f"Loaded {len(df)} products from products.csv")

    # Validate required columns
    required_columns = ["Product Name", "Category", "Price", "Product Description", "Available Variants"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns in products.csv: {missing_columns}")
        return

    # Standardize category values (convert to lowercase)
    df["Category"] = df["Category"].str.lower().str.strip()

    # Prepare documents and metadata
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Combine relevant fields for embedding
        text = f"{row['Product Name']} {row['Product Description']} {row.get('Tags', '')} {row.get('Vibe', '')} {row.get('Available Variants', '')} {row.get('Product Details', '')}"
        documents.append(text)
        metadatas.append({
            "product_name": row["Product Name"],
            "category": row["Category"],
            "tags": row.get("Tags", ""),
            "price": float(row["Price"]),
            "product_description": row["Product Description"],
            "seo_title": row.get("SEO Title", ""),
            "seo_description": row.get("SEO Description", ""),
            "available_variants": row["Available Variants"],
            "product_details": row.get("Product Details", ""),
            "vibe": row.get("Vibe", "")
        })
        ids.append(str(idx))
        logger.debug(f"Prepared product {idx}: {row['Product Name']}, category: {row['Category']}")

    # Generate embeddings and store in ChromaDB
    embeddings = embedding_function.embed_documents(documents)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    logger.info(f"Stored {len(documents)} product embeddings in ChromaDB.")