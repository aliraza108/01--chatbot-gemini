import os
import traceback
from fastapi import FastAPI, HTTPException
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
    print("❌ API_KEY missing")
    raise ValueError("API_KEY environment variable not set")

MODEL = "gemini-2.5-flash"

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
    "http://localhost:3000",
    "*"
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
Maintain context from the conversation.
Always summarize when asked.
""",
    model=MODEL,
)

# -------------------------
# CHAT FUNCTION
# -------------------------
async def run_chat_agent(user_message: str):
    global chat_summary

    chat_history.append({"role": "user", "content": user_message})
    full_response = ""

    try:
        result = Runner.run_streamed(chat_agent, input=chat_history)

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

    except Exception as e:
        print("❌ STREAM ERROR")
        traceback.print_exc()
        raise e

    chat_history.append({"role": "assistant", "content": full_response})

    # update summary
    try:
        summary_prompt = f"""
Conversation:
{chat_history}

Current Summary:
{chat_summary}

Update the summary in 2-3 sentences.
"""
        summary_result = await Runner.run(chat_agent, input=summary_prompt)
        chat_summary = summary_result.final_output

    except Exception as e:
        print("❌ SUMMARY ERROR")
        traceback.print_exc()

    return full_response, chat_summary

# -------------------------
# API ENDPOINTS
# -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response, summary = await run_chat_agent(request.message)
        return ChatResponse(response=response, summary=summary)

    except Exception as e:
        print("❌ CHAT ENDPOINT ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_chat():
    global chat_history, chat_summary
    chat_history = []
    chat_summary = ""
    return {"status": "reset complete"}


# -------------------------
# DEBUG ENDPOINT
# -------------------------
@app.get("/debug")
async def debug():
    return {
        "api_key_present": bool(API_KEY),
        "model": MODEL,
        "history_length": len(chat_history),
        "summary_length": len(chat_summary),
    }


# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
async def health():
    return {"status": "ok"}
