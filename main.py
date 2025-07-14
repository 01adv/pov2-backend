import re
import ast
import os
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from rapidfuzz import fuzz, process
import chromadb
from embedder import embed_products
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Load system prompt
with open('prompt.txt', 'r', encoding='utf-8') as file:
    system_prompt_text = file.read()

# Load products (for fallback fuzzy matching)
with open('products.txt', 'r', encoding='utf-8') as f:
    product_list = [line.strip() for line in f.readlines() if line.strip()]

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Session management
SESSION_TIMEOUT = timedelta(minutes=30)
sessions = {}  # session_id: {"messages": [...], "last_active": datetime}

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting... Running embedder to generate product embeddings.")
    embed_products()



# def vector_search(query: str, top_k: int = 7, filters: dict = None) -> list:
#     logger.info(f"Raw input query: '{query}', filters: {filters}")
#     # Handle case where query is a JSON string containing query and filters
#     try:
#         if isinstance(query, str) and query.startswith("{"):
#             input_dict = json.loads(query)
#             query = input_dict.get("query", query)
#             filters = input_dict.get("filters", filters)
#     except json.JSONDecodeError:
#         logger.warning("Failed to parse query as JSON, using as-is")
    
    
#     if filters and len(filters) > 1 and "$and" not in filters and "$or" not in filters:
#         filters = {"$and": [{key: value} for key, value in filters.items()]}
    
#     logger.info(f"Running vector search for query: '{query}' with filters: {filters}")
#     client = chromadb.PersistentClient(path="./chroma_db")
#     collection = client.get_collection(name="products")
#     embedding_function = OpenAIEmbeddings(
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small"
#     )
#     query_embedding = embedding_function.embed_query(query)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         where=filters or {},  # Ensure filters are applied
#         include=["metadatas", "documents"]
#     )
#     logger.debug(f"Vector search result: {results}")
#     return [
#         {
#             "metadata": metadata,
#             "document": document
#         }
#         for metadata, document in zip(results["metadatas"][0], results["documents"][0])
#     ]

import json
import logging

def vector_search(query: str, top_k: int = 7, filters: dict = None) -> list:
    logger.info(f"Raw input query: '{query}', filters: {filters}")
    # Handle case where query is a JSON string containing query and filters
    try:
        if isinstance(query, str) and query.startswith("{"):
            input_dict = json.loads(query)
            query = input_dict.get("query", query)
            filters = input_dict.get("filters", filters)
    except json.JSONDecodeError:
        logger.warning("Failed to parse query as JSON, using as-is")

    # Transform filters to use $and if multiple conditions are present
    if filters and len(filters) > 1 and "$and" not in filters and "$or" not in filters:
        filters = {"$and": [{key: value} for key, value in filters.items()]}

    logger.info(f"Running vector search for query: '{query}' with filters: {filters}")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="products")
    embedding_function = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    query_embedding = embedding_function.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters or {},
        include=["metadatas", "documents"]
    )
    logger.debug(f"Vector search result: {results}")

    # Fallback to filter-less search if no results
    if not results["metadatas"] or not results["metadatas"][0]:
        logger.info("No results with filters, retrying without filters")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or {},
            include=["metadatas", "documents"]
        )
        logger.debug(f"Filter-less search result: {results}")

    return [
        {
            "metadata": metadata,
            "document": document
        }
        for metadata, document in zip(results["metadatas"][0], results["documents"][0])
    ] if results["metadatas"] and results["metadatas"][0] else []


# Tool for the agent
tools = [
    Tool(
        name="vector_search",
        func=vector_search,
        description="Search for products matching the query with optional filters for category, price, or available_variants (e.g., {'category': 'dress', 'price': {'$lte': 100}, 'color': 'red'})."    )
]

# Initialize the agent
# Initialize the agent
system_prompt = PromptTemplate(
    input_variables=["input", "chat_history", "agent_scratchpad"],  # Include agent_scratchpad
    template=system_prompt_text  # Load prompt.txt with escaped curly braces
)

agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=system_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms to the format instructions."
)


def is_session_expired(last_active: datetime) -> bool:
    return datetime.utcnow() - last_active > SESSION_TIMEOUT


df = pd.read_csv("products.csv")

def clean_ai_text(ai_content: str) -> str:
    return re.sub(r'products:\s*\[.*?\]', '', ai_content, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_products(text: str, df: pd.DataFrame, threshold: int = 80) -> list:
    product_list = df["Product Name"].tolist()
    matches = process.extract(text, product_list, scorer=fuzz.partial_ratio)
    filtered = [match for match, score, _ in matches if score >= threshold]
    matched_rows = df[df["Product Name"].isin(filtered)][["Product Name", "Category", "Price", "Available Variants"]].to_dict(orient="records")
    logger.info(f"Fuzzy matched products: {matched_rows}")
    return matched_rows

def extract_products_from_ai_response(ai_content: str, df: pd.DataFrame) -> list:
    match = re.search(r'products:\s*(\[[^\]]*\])', ai_content, re.IGNORECASE)
    if match:
        try:
            products = ast.literal_eval(match.group(1))
            logger.info(f"Extracted structured products from AI response: {products}")
            return products
        except Exception as e:
            logger.warning(f"Failed to parse product list from AI response: {e}")
    if any(keyword in ai_content.lower() for keyword in ["dress", "skirt", "top", "pants", "outfit"]):
        return extract_products(ai_content, df)
    return []

def get_chat_history(session_messages):
    return [msg.content for msg in session_messages if isinstance(msg, (HumanMessage, AIMessage))]

# def get_agent_chat_history(session_messages):
#     return [msg for msg in session_messages if isinstance(msg, (HumanMessage, AIMessage))]
def get_agent_chat_history(session_messages, max_messages=8):
    relevant_messages = [msg for msg in session_messages if isinstance(msg, (HumanMessage, AIMessage))]
    return relevant_messages[-max_messages:]  # Keep only the last 5 messages

def generate_title_with_llm(user_input: str, matched_products: list) -> str:
    prompt_messages = [
        SystemMessage(
            content="You are a creative assistant. Generate a short and catchy title summarizing the type of fashion items based on user intent and product names."
        ),
        HumanMessage(
            content=f"User is shopping for: {user_input}\n\nRecommended products:\n{', '.join(matched_products)}\n\nGive me a short catchy title (under 8 words)."
        )
    ]
    logger.debug(f"Generating title with LLM prompt: {prompt_messages}")
    response = llm.invoke(prompt_messages)
    logger.info(f"Generated title: {response.content.strip()}")
    return response.content.strip().strip('"')

# --- Routes ---


@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    try:
        body = await request.json()
        user_input = body.get("message")
        logger.info(f"Received chat message: {user_input} for session {session_id}")

        if not user_input:
            raise HTTPException(status_code=400, detail="No message provided")

        now = datetime.utcnow()

        if session_id not in sessions or is_session_expired(sessions[session_id]["last_active"]):
            logger.info(f"Creating new session: {session_id}")
            sessions[session_id] = {
                "messages": [SystemMessage(content=system_prompt_text)],
                "last_active": now
            }
        else:
            logger.info(f"Updating session activity: {session_id}")
            sessions[session_id]["last_active"] = now

        chat_history = get_agent_chat_history(sessions[session_id]["messages"])
        logger.debug(f"Chat history: {[msg.content for msg in chat_history]}")
        sessions[session_id]["messages"].append(HumanMessage(content=user_input))

        logger.debug(f"Agent inputs: input={user_input}, chat_history={[msg.content for msg in chat_history]}")
        response_payload = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        response = response_payload['output']
        logger.info(f"Agent response: {response}")

        matched_products = extract_products_from_ai_response(response, df)
        logger.debug(f"Matched products: {matched_products}")

        cleaned_text = clean_ai_text(response)
        sessions[session_id]["messages"].append(AIMessage(content=response))

        if matched_products:
            # Handle case where matched_products is a list of strings or list of dicts
            product_names = [p["Product Name"] if isinstance(p, dict) else p for p in matched_products]
            title = generate_title_with_llm(user_input, product_names)
            return {
                "response": {
                    "text": cleaned_text,
                    "products": matched_products,
                    "title": title
                },
                "history": get_chat_history(sessions[session_id]["messages"])
            }

        return {
            "response": {
                "text": cleaned_text
            },
            "history": get_chat_history(sessions[session_id]["messages"])
        }

    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/healthcheck")
async def healthcheck():
    logger.info("Healthcheck called.")
    return {"message": "I'm alive!"}