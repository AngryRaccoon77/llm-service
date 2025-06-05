# api/main.py
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from llmService.app.model.chat import ChatRequest
from llmService.app.application.services import chat_service

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RUT Admissions Chatbot API",
    description="API for the AI assistant of the RUT Admissions Committee",
    version="1.0.0"
)

@app.post("/chat", summary="Process a chat message")
async def chat(request: ChatRequest):
    """
    Receives a user query and chat ID, processes it through the agent,
    and returns the agent's response.
    """
    try:
        logger.info(f"Received query for chat_id {request.chat_id}: {request.query}")
        response = chat_service.process_chat(request.chat_id, request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"An error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8084)