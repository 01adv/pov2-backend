import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
from langchain_openai import ChatOpenAI
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


# llm
# llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# module‑level singletons
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="products")
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)


def preprocess_filters(filters: dict) -> dict:
    if not filters:
        return {}

    categories = []
    price_values = []
    other_filters = {}

    for key, value in filters.items():
        if key == "category" and isinstance(value, str):
            categories = [c.strip() for c in value.split(",")]
        elif key == "price":
            if isinstance(value, dict):
                price_val = value.get("$lte")
                if isinstance(price_val, str) and "," in price_val:
                    price_values = [int(x.strip()) for x in price_val.split(",")]
                elif isinstance(price_val, (int, float)):
                    price_values = [price_val]
            elif isinstance(value, (int, float)):
                price_values = [value]
        else:
            other_filters[key] = value

    # Build OR filter blocks
    or_block = []
    if categories:
        for cat in categories:
            if price_values:
                for price in price_values:
                    clause = {"category": cat, "price": {"$lte": price}}
                    clause.update(other_filters)
                    or_block.append(clause)
            else:
                clause = {"category": cat}
                clause.update(other_filters)
                or_block.append(clause)
    elif price_values:
        for price in price_values:
            clause = {"price": {"$lte": price}}
            clause.update(other_filters)
            or_block.append(clause)

    # Simplify return if only one clause
    if len(or_block) == 1:
        return or_block[0]
    elif or_block:
        return {"$or": or_block}

    # Fallback: $and logic for other filters
    if len(filters) > 1:
        return {"$and": [{k: v} for k, v in filters.items()]}

    return filters




def normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def vector_search(query: str, top_k: int = 5, filters: dict | None = None) -> list:

    if isinstance(query, str) and query.lstrip().startswith("{"):
        data = json.loads(query)
        query   = data.get("query", query)
        top_k   = data.get("top_k", top_k)
        filters = data.get("filters", filters)

    if filters:
        filters = preprocess_filters(filters)


    # Normalize query embedding
    query_vec = normalize(embedder.embed_query(query))

    def run(qvec, where):
        query_params = {
            "query_embeddings": [qvec],
            "n_results": top_k,
            "include": ["metadatas", "documents", "distances"]
        }
        if where:
            query_params["where"] = where
        logger.info(f"Querying with params: {query_params}")
        return collection.query(**query_params)

    logger.info(f"Running vector search with query: {query}, filters: {filters}")
    res = run(query_vec, filters)
    # logger.info(f"Raw results (with filters): {res['metadatas'][0]}, distances: {res['distances'][0]}")

    if not res["metadatas"][0]:
        logger.info("No results with filters, trying without filters")
        res = run(query_vec, None)
        # logger.info(f"Raw results (no filters): {res['metadatas'][0]}, distances: {res['distances'][0]}")

    # Use tighter threshold for normalized cosine distance (optional tweak)
    # THRESH = 1.2  # smaller is better similarity, 1 - cosine sim

    hits = [
        {"metadata": m, "document": d, "score": 1 - x}
        for m, d, x in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
        # if x <= THRESH
    ]
    # logger.debug(f"Filtered results (threshold={THRESH}): {hits}")
    return hits



# will use it later 🌄🌄

# def llm_rerank(user_query: str, hits: list, keep: int = 3):
#     docs = [f"{i+1}. {h['metadata']['product_name']}. {h['document'][:120]}" 
#             for i, h in enumerate(hits)]
#     prompt = (
#         "User query: " + user_query + "\n\n"
#         "Products:\n" + "\n".join(docs) + "\n\n"
#         "Return the indexes of the top " + str(keep) + " most relevant products "
#         "in JSON array, e.g. [2,3]. Only output JSON."
#     )
#     logger.debug(f"LLM rerank prompt: {prompt}")
#     indices = json.loads(llm.invoke(prompt).content)
#     logger.debug(f"LLM rerank result: {indices}")
#     return [hits[i-1] for i in indices]
