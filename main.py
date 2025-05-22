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

from langchain.schema import HumanMessage, AIMessage, SystemMessage  # or appropriate import based on your framework

@app.get("/recommendation/{session_id}")
async def get_recommendation(session_id: str):
    # Check if the session exists
    if session_id not in sessions:
        return {"error": "Session not found"}

    # Extract session history
    session_history = sessions[session_id]

    # Filter only assistant messages (usually AIMessage type)
    assistant_messages = [msg for msg in session_history if isinstance(msg, AIMessage)]
    if not assistant_messages:
        return {"error": "No assistant messages to extract recommendation from."}

    # Get the last assistant message
    last_recommendation = assistant_messages[-1].content

    # Remove the last assistant message from the session history to use as context
    trimmed_history = session_history.copy()
    trimmed_history.remove(assistant_messages[-1])

    # Join the rest as chat context
    chat_context = " ".join([msg.content for msg in trimmed_history if hasattr(msg, 'content')])

    # Prompt for one-liner recommendation
    one_liner_prompt = (
        f"Given the following chat context, generate a one-liner skincare product recommendation "
        f"that reflects the user's request and the assistant's final recommendation:\n"
        f"Chat context: {chat_context}\n"
        f"Final assistant recommendation: {last_recommendation}\n"
        f"The output must be under 10 words, directly tying user needs to the product and add the product name too. "
        f"Example: 'For your dry, frizzy hair, Nourishing shampoo with Argan oil and Gentle Hair Mark with Shea Butter"
    )

    # Query the LLM
    one_liner_response = llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=one_liner_prompt)
    ])

    return {
        "one_liner": one_liner_response.content
    }
