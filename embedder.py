import pandas as pd
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

def embed_products():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="products")

    # Initialize OpenAI embedding function
    embedding_function = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )

    # Read products from CSV
    df = pd.read_csv("products.csv")

    # Prepare documents and metadata
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Combine relevant fields for embedding
        text = f"{row['Product Name']} {row["Price"]} {row['Product Description']} {row['Tags']} {row["Price"]} {row['Vibe']} {row['Available Variants']}"
        documents.append(text)
        metadatas.append({
            "product_name": row["Product Name"],
            "category": row["Category"],
            "tags": row["Tags"],
            "price": row["Price"],
            "product_description": row["Product Description"],
            "seo_title": row["SEO Title"],
            "seo_description": row["SEO Description"],
            "available_variants": row["Available Variants"],
            "product_details": row["Product Details"],
            "vibe": row["Vibe"]
        })
        ids.append(str(idx))

    # Generate embeddings and store in ChromaDB
    embeddings = embedding_function.embed_documents(documents)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    print("Product embeddings stored in ChromaDB.")

if __name__ == "__main__":
    embed_products()
