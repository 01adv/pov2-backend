import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

app = FastAPI()

with open('prompt.txt', 'r') as file:
    system_prompt = file.read()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

sessions = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    body = await request.json()
    user_input = body.get("message")

    if not user_input:
        return {"error": "No message provided"}

    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content=system_prompt),
        ]

    sessions[session_id].append(HumanMessage(content=user_input))

    response = llm(sessions[session_id])

    sessions[session_id].append(AIMessage(content=response.content))

    return {
        "response": response.content,
        "history": [message.content for message in sessions[session_id]]
    }
