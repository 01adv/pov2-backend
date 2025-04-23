import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()
app = FastAPI()

# GPT-4o LLM via LangChain
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Store chat history per session (ephemeral)
sessions = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    body = await request.json()
    user_input = body.get("message")

    if not user_input:
        return {"error": "No message provided"}

    # Get or create chat history
    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content="You are a helpful assistant."),
        ]

    # Add user message
    sessions[session_id].append(HumanMessage(content=user_input))

    # Call OpenAI via LangChain
    response = llm(sessions[session_id])

    # Add AI response to history
    sessions[session_id].append(AIMessage(content=response.content))

    return {"response": response.content, "history": [m.content for m in sessions[session_id]]}
