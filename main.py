import re
import ast
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from rapidfuzz import fuzz, process
import chromadb
from chromadb.utils import embedding_functions
from embedder import embed_products

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Load system prompt
with open('prompt.txt', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

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

# Vector search function
def vector_search(query: str, top_k: int = 3, filters: dict = None) -> list:
    logger.info(f"Running vector search for query: '{query}' with filters: {filters}")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="products")
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    query_embedding = embedding_function([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters,
        include=["metadatas"]
    )
    logger.debug(f"Vector search result: {results}")
    return [metadata for metadata in results["metadatas"][0]]

# Tool for the agent
tools = [
    Tool(
        name="vector_search",
        func=vector_search,
        description="Search for products matching the user's query."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

def is_session_expired(last_active: datetime) -> bool:
    return datetime.utcnow() - last_active > SESSION_TIMEOUT

def extract_products_from_ai_response(ai_content: str) -> list:
    match = re.search(r'products:\s*(\[[^\]]*\])', ai_content, re.IGNORECASE)
    if match:
        try:
            products = ast.literal_eval(match.group(1))
            logger.info(f"Extracted structured products from AI response: {products}")
            return products
        except Exception as e:
            logger.warning(f"Failed to parse product list from AI response: {e}")
    return []

def clean_ai_text(ai_content: str) -> str:
    return re.sub(r'products:\s*\[.*?\]', '', ai_content, flags=re.IGNORECASE | re.DOTALL).strip()

def extract_products(text: str, product_list, threshold: int = 80):
    matches = process.extract(text, product_list, scorer=fuzz.partial_ratio)
    filtered = [match for match, score, _ in matches if score >= threshold]
    logger.info(f"Fuzzy matched products: {filtered}")
    return filtered

def get_chat_history(session_messages):
    return [msg.content for msg in session_messages if isinstance(msg, (HumanMessage, AIMessage))]

def generate_title_with_llm(user_input: str, matched_products: list) -> str:
    prompt = [
        SystemMessage(
            content="You are a creative assistant. Generate a short and catchy title summarizing the type of fashion items based on user intent and product names."
        ),
        HumanMessage(
            content=f"User is shopping for: {user_input}\n\nRecommended products:\n{', '.join(matched_products)}\n\nGive me a short catchy title (under 8 words)."
        )
    ]
    logger.debug(f"Generating title with LLM prompt: {prompt}")
    response = llm(prompt)
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
                "messages": [SystemMessage(content=system_prompt)],
                "last_active": now
            }
        else:
            logger.info(f"Updating session activity: {session_id}")
            sessions[session_id]["last_active"] = now

        sessions[session_id]["messages"].append(HumanMessage(content=user_input))

        logger.debug("Calling agent with user input...")
        response = agent.run(user_input)
        logger.info(f"Agent response: {response}")

        matched_products = extract_products_from_ai_response(response)
        if not matched_products:
            matched_products = extract_products(response, product_list)

        cleaned_text = clean_ai_text(response)
        sessions[session_id]["messages"].append(AIMessage(content=response))

        if matched_products:
            title = generate_title_with_llm(user_input, matched_products)
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

@app.post("/nudge/{session_id}")
async def generate_nudge(session_id: str, request: Request):
    try:
        body = await request.json()
        product_name = body.get("product_name")
        logger.info(f"Received nudge request for product: {product_name}, session: {session_id}")

        if not product_name:
            raise HTTPException(status_code=400, detail="No product name provided")

        now = datetime.utcnow()
        if session_id not in sessions or is_session_expired(sessions[session_id]["last_active"]):
            logger.info(f"Creating new session for nudge: {session_id}")
            sessions[session_id] = {
                "messages": [SystemMessage(content=system_prompt)],
                "last_active": now
            }
        else:
            logger.info(f"Updating session activity for nudge: {session_id}")
            sessions[session_id]["last_active"] = now

        history = [msg.content for msg in sessions[session_id]["messages"] if isinstance(msg, (HumanMessage, AIMessage))]

        prompt = [
            SystemMessage(
                content="You are a persuasive, friendly fashion assistant. Based on the conversation, write a short, encouraging nudge for why this product would be a great choice for the user, incorporating styling tips and making the user feel stylish and confident."
            ),
            HumanMessage(
                content=f"Conversation:\n{chr(10).join(history)}\n\nProduct: {product_name}\n\nProvide a brief, upbeat nudge that includes 1 fun styling tip (with emojis). Ensure that the styling tip has a punchy, engaging vibe, and the nudge should inspire confidence and excitement about the choice. Do not add any fluff words / non-meaningful words. It should be maximum 1 sentence. For the Product - Ambition Crepe & Satin Pencil Skirt, Here is an example nudge for evening look - Pair with a silk blouse & pointed pumps 👠, Here is an example nudge for casual look - Team with a sequin cami & strappy heels, Here is an example nudge for Professional look - Style under a chunky knit & ankle boots ☕. Format: Do not repeat the product name. Correct: Slip into this dress and pair with bold red heels and a metallic clutch for a look that's both daring and sophisticated! 💃✨"
            )
        ]
        logger.debug(f"Sending nudge prompt to LLM: {prompt}")
        nudge_response = llm(prompt)
        logger.info(f"Nudge generated: {nudge_response.content.strip()}")

        sessions[session_id]["messages"].append(AIMessage(content=nudge_response.content.strip()))

        return {"nudge": nudge_response.content.strip()}

    except Exception as e:
        logger.exception("Nudge endpoint failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    try:
        if session_id in sessions:
            del sessions[session_id]
            logger.info(f"Session {session_id} deleted.")
            return {"message": f"Session {session_id} deleted."}
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.exception("Session deletion failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/healthcheck")
async def healthcheck():
    logger.info("Healthcheck called.")
    return {"message": "I'm alive!"}
