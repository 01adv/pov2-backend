import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables (for OpenAI API key, etc.)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; use ["http://localhost:3000"] or your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Or ["POST"] for more restrictive settings
    allow_headers=["*"],  # Or ["Content-Type"] etc.
)

# Initialize GPT model (GPT-4o via LangChain)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# In-memory session store (resets on server restart)
sessions = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    body = await request.json()
    user_input = body.get("message")

    if not user_input:
        return {"error": "No message provided"}

    # Create session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content="You are a helpful assistant."),
        ]

    # Add user's message to the session history
    sessions[session_id].append(HumanMessage(content=user_input))

    # Get LLM response
    response = llm(sessions[session_id])

    # Append assistant's response to history
    sessions[session_id].append(AIMessage(content=response.content))

    return {
        "response": response.content,
        "history": [message.content for message in sessions[session_id]]
    }
