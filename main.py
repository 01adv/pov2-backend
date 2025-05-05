import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

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

def extract_products(text: str, product_list):
    found = []
    for product in product_list:
        if re.search(rf'\b{re.escape(product)}\b', text, re.IGNORECASE):
            found.append(product)
    return found

def clean_ai_text(ai_content: str) -> str:
    # Remove 'text: "..."' and 'products: [...]'
    cleaned = re.sub(r'text:\s*"[^"]*"', '', ai_content, flags=re.IGNORECASE)
    cleaned = re.sub(r'products:\s*\[.*?\]', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
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
