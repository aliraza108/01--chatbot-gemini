import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from agents import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

# -------------------------
# CONFIG
# -------------------------
API_KEY = "AIzaSyDjoJxy-MX9PDxTKgaBKUeKTCPHsodAXF8"
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

    # Append user message
    chat_history.append({"role": "user", "content": user_message})

    # Run agent and stream response
    result = Runner.run_streamed(chat_agent, input=chat_history)
    full_response = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta
            full_response += delta

    # Append assistant response
    chat_history.append({"role": "assistant", "content": full_response})

    # Generate updated summary
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

@app.get("/")
async def getpage():
    return "working fine page"