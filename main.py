import re
import json
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
from vector_search_and_reranking import vector_search

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

# @app.on_event("startup")
# async def startup_event():
#     logger.info("Server starting... Running embedder to generate product embeddings.")
#     embed_products()


# Tool for the agent
# tools = [
#         Tool(
#             name="vector_search",
#             func=vector_search,
#             description=(
#                 "Semantic product search.\n"
#                 "Args:\n"
#                 "  query (str): user intent.\n"
#                 "  top_k (int, optional): number of products to return (default 5).\n"
#                 "  filters (dict, optional): structured filters, e.g. "
#                 "{'category':'dress', 'price':{'$lte':150}}"
#             )
#         )

# ]

tools = [
Tool(
    name="vector_search",
    func=vector_search,
    description=(
        "Semantic product search.\n"
        "Args:\n"
        "  query (str): user request.\n"
        "  top_k (int): number of hits.\n"
        "  filters (dict): {'category': 'dress', 'price': {'$lte': 150}}"
    ),
    argument_schema={
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5},
        "filters": {"type": "object", "default": {}}
    }
)


]


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


df = pd.read_csv("new-cleaned-products.csv")

def clean_ai_text(ai_content: str) -> str:
    return re.sub(r'products:\s*\[.*?\]', '', ai_content, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_products(text: str, df: pd.DataFrame, threshold: int = 80) -> list:
    product_list = df["product-name"].tolist()
    matches = process.extract(text, product_list, scorer=fuzz.partial_ratio)
    filtered = [match for match, score, _ in matches if score >= threshold]
    matched_rows = df[df["product-name"].isin(filtered)][["product-name", "category", "price", "color"]].to_dict(orient="records")
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
            product_names = [p["product-name"] if isinstance(p, dict) else p for p in matched_products]
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
    
    
    
    
    
    
# nudge endpoint
@app.post("/nudge/{session_id}")
async def generate_nudge(session_id: str, request: Request):
    body = await request.json()
    product_name = body.get("product_name")
    if not product_name:
        return {"error": "No product name provided"}

    now = datetime.utcnow()  # ADDED

    # MODIFIED
    if session_id not in sessions or is_session_expired(sessions[session_id]["last_active"]):
        sessions[session_id] = {
            "messages": [SystemMessage(content=system_prompt)],
            "last_active": now
        }
    else:
        sessions[session_id]["last_active"] = now  # ADDED

    history = [msg.content for msg in sessions[session_id]
               ["messages"] if isinstance(msg, (HumanMessage, AIMessage))]

    prompt = [
        SystemMessage(content="You are a persuasive, friendly fashion assistant. Based on the conversation, write a short, encouraging nudge for why this product would be a great choice for the user, incorporating styling tips, benefits, and making the user feel stylish and confident."),
        HumanMessage(content=f"Conversation:\n{chr(10).join(history)}\n\nProduct: {product_name}\n\nProvide a brief, upbeat nudge that includes 1 fun styling tips (with emojis). Ensure that the styling tip has a punchy, engaging vibe, and the nudge should inspire confidence and excitement about the choice. Do not add any fluff words / non-meaningful words. It should be maximum 1 sentence. For the Product - Ambition Crepe & Satin Pencil Skirt, Here is an example nudges for evening look - Pair with a silk blouse & pointed pumps 👠 , Here is an example nudges for casual look -  Team with a sequin cami & strappy heels, Here is an example nudges for Professional look -  Style under a chunky knit & ankle boots ☕.")
    ]

    logger.debug(f"Nudge history: {history}")
    nudge_response = llm(prompt)
    nudge_text = nudge_response.content.strip()

    sessions[session_id]["messages"].append(AIMessage(content=nudge_text))

    return {"nudge": nudge_text}
    

@app.get("/healthcheck")
async def healthcheck():
    logger.info("Healthcheck called.")
    return {"message": "I'm alive!"}