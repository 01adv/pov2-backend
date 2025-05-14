import os
import re
import ast
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rapidfuzz import fuzz, process

load_dotenv()

app = FastAPI()

# Load system prompt
with open('prompt.txt', 'r') as file:
    system_prompt = file.read()

# Load products
with open('products.txt', 'r') as f:
    product_list = [line.strip() for line in f.readlines() if line.strip()]

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
sessions = {}

# --- Utility Functions ---

def extract_products(text: str, product_list, threshold: int = 80):
    matches = process.extract(text, product_list, scorer=fuzz.partial_ratio)
    found = [match for match, score, _ in matches if score >= threshold]
    return found

def extract_products_from_ai_response(ai_content: str) -> list:
    match = re.search(r'products:\s*(\[[^\]]*\])', ai_content, re.IGNORECASE)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except Exception:
            return []
    return []

def clean_ai_text(ai_content: str) -> str:
    cleaned = re.sub(r'products:\s*\[.*?\]', '', ai_content, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

def get_chat_history(session_messages):
    return [msg.content for msg in session_messages if isinstance(msg, (HumanMessage, AIMessage))]

def generate_title_with_llm(user_input: str, matched_products: list) -> str:
    title_prompt = [
        SystemMessage(content="You are a creative assistant. Generate a short and catchy title summarizing the type of fashion items based on user intent and product names."),
        HumanMessage(content=f"User is shopping for: {user_input}\n\nRecommended products:\n{', '.join(matched_products)}\n\nGive me a short catchy title (under 8 words).")
    ]
    title_response = llm(title_prompt)
    return title_response.content.strip().strip('"')

# --- Routes ---

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    body = await request.json()
    user_input = body.get("message")

    if not user_input:
        return {"error": "No message provided"}

    if session_id not in sessions:
        sessions[session_id] = [SystemMessage(content=system_prompt)]

    sessions[session_id].append(HumanMessage(content=user_input))
    response = llm(sessions[session_id])
    ai_content = response.content

    # Try to extract products from AI response list
    matched_products = extract_products_from_ai_response(ai_content)

    # Fallback to fuzzy match if needed
    if not matched_products:
        matched_products = extract_products(ai_content, product_list)

    cleaned_text = clean_ai_text(ai_content)
    sessions[session_id].append(AIMessage(content=ai_content))

    if matched_products:
        title = generate_title_with_llm(user_input, matched_products)
        return {
            "response": {
                "text": cleaned_text,
                "products": matched_products,
                "title": title
            },
            "history": get_chat_history(sessions[session_id])
        }

    return {
        "response": {
            "text": cleaned_text
        },
        "history": get_chat_history(sessions[session_id])
    }

@app.post("/nudge/{session_id}")
async def generate_nudge(session_id: str, request: Request):
    body = await request.json()
    product_name = body.get("product_name")

    if not product_name:
        return {"error": "No product name provided"}

    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = [SystemMessage(content=system_prompt)]

    history = [msg.content for msg in sessions[session_id] if isinstance(msg, (HumanMessage, AIMessage))]

    prompt = [
        SystemMessage(content="You are a persuasive, friendly fashion assistant. Based on the conversation, write a short, encouraging nudge for why this product would be a great choice for the user, incorporating styling tips, benefits, and making the user feel stylish and confident."),
        HumanMessage(content=f"Conversation:\n{chr(10).join(history)}\n\nProduct: {product_name}\n\nProvide a brief, upbeat nudge that includes 3 fun styling tips (with emojis) and 2 benefits in a friendly tone. Make it sound like the user has picked a great, fashionable item! Ensure each styling tip and benefit has a punchy, engaging vibe, and the nudge should inspire confidence and excitement about the choice. Do not add any fluff words / non-meaningful words. Make sure that it is sent as Tip 1, Tip 2, Tip 3, Benefit 1, Benefit 2. Send it as a structure.")
    ]

    nudge_response = llm(prompt)
    nudge_text = nudge_response.content.strip()

    # Add nudge as AI message to the session context
    sessions[session_id].append(AIMessage(content=nudge_text))

    return {"nudge": nudge_text}


