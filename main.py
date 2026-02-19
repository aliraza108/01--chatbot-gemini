import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from agents import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

# -------------------------
# CONFIG
# -------------------------
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set")
MODEL = "gemini-2.0-flash"

client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=API_KEY
)

set_default_openai_api("chat_completions")
set_default_openai_client(client)
set_tracing_disabled(True)

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI(title="Chatbot")

# -------------------------
# CORS CONFIGURATION
# -------------------------
origins = [
    "https://01-chatbot.vercel.app",
    *
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# REQUEST / RESPONSE MODELS
# -------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    summary: str

# -------------------------
# GLOBAL MEMORY
# -------------------------
chat_history: List[Dict] = []
chat_summary: str = ""

# -------------------------
# CHAT AGENT
# -------------------------
chat_agent = Agent(
    name="Simple Chatbot",
    instructions="""
You are a friendly and helpful chatbot developed by Ali Raza. 
Respond clearly and concisely. 
Maintain context from the conversation and provide helpful suggestions or answers.
Always summarize the conversation when asked.
""",
    model=MODEL,
)

# -------------------------
# CHAT FUNCTION
# -------------------------
async def run_chat_agent(user_message: str):
    global chat_summary

    chat_history.append({"role": "user", "content": user_message})

    result = Runner.run_streamed(chat_agent, input=chat_history)
    full_response = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            full_response += event.data.delta

    chat_history.append({"role": "assistant", "content": full_response})

    # Update summary
    summary_prompt = f"""
Conversation:
{chat_history}

Current Summary:
{chat_summary}

Update the summary in 2-3 sentences.
"""
    summary_result = await Runner.run(chat_agent, input=summary_prompt)
    chat_summary = summary_result.final_output

    return full_response, chat_summary

# -------------------------
# API ENDPOINTS
# -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response, summary = await run_chat_agent(request.message)
    return ChatResponse(response=response, summary=summary)

@app.post("/reset")
async def reset_chat():
    global chat_history, chat_summary
    chat_history = []
    chat_summary = ""
    return {"status": "reset complete"}
