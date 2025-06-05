from pydantic import BaseModel

class ChatRequest(BaseModel):
    chat_id: str
    query: str

class TokenCountRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str