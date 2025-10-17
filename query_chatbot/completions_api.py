import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional

app = FastAPI(title="Completions-Compatible REST API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: float
    model: str
    choices: List[Choice]

# ----------------------
# Endpoints
# ----------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

async def process_completions_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Example stub logic — replace with actual chatbot/RAG
    print(f"Processing request: {request}")
    answer_text = "answer" # Call you chatbot HERE
    return ChatCompletionResponse(
        id="question-1",
        object="chat.completion",
        created=time.time(),
        model="mock-model",
        choices=[
            Choice(
                index=0,
                text=answer_text,
                finish_reason="stop"
            )
        ]
    )

# ----------------------
# Run with:
# uvicorn myapp:app --reload
# ----------------------