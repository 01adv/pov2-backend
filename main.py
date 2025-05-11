from fastapi import FastAPI, Request
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (for OpenAI API key, etc.)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; use ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GPT model (using GPT-4o via LangChain)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Load system prompt for context (adjust as needed)
with open('prompt.txt', 'r') as file:
    system_prompt = file.read()

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
            SystemMessage(content=system_prompt),
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

@app.get("/recommendation/{session_id}")
async def get_recommendation(session_id: str):
    # Check if the session exists
    if session_id not in sessions:
        return {"error": "Session not found"}

    # Extract session history and context
    session_history = sessions[session_id]
    chat_context = " ".join([message.content for message in session_history])

    # Generate the one-liner message based on the context
    one_liner_prompt = f"Given the following chat context, generate a one-liner skincare product recommendation that reflects the user's request and the product(s) discussed:\n{chat_context}\n The output must be under 10 words, directly tying user needs to the product. Example: 'Perfect for dry skin — lightweight, deeply hydrating serum."
        
    one_liner_response = llm([SystemMessage(content=system_prompt), HumanMessage(content=one_liner_prompt)])
    
    return {
        "one_liner": one_liner_response.content
    }
