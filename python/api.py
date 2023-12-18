from pathlib import Path
import asyncio
import uvicorn,os
import logging
import time
from typing import List, Literal, Optional, Union

import chatglm_cpp
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse



CHAT_SYSTEM_PROMPT = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm3-ggml-q4_0.bin"
if not os.path.exists(DEFAULT_MODEL_PATH):
    print('##### 模型文件不存在：',DEFAULT_MODEL_PATH)

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")

class Settings(BaseSettings):
    # model: str = "chatglm-ggml.bin"
    num_threads: int = 8
    server_name: str = "ChatGLM3 CPP API Server"
    model: str = str(DEFAULT_MODEL_PATH)  # Path to chatglm model in ggml format
    host: str = "127.0.0.1"
    port: int = 8000
    print("####线程数",num_threads)

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "ChatGLM3"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "ChatGLM3"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "ChatGLM3",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


settings = Settings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
pipeline = None
lock = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = chatglm_cpp.Pipeline(settings.model)
    messages =[]
    messages.append({
        "role":"user", "content":"hi"
    })
    messages_with_system=[]
    if CHAT_SYSTEM_PROMPT:
        messages_with_system.append({
            "role":"system", "content":CHAT_SYSTEM_PROMPT
        })
    messages_with_system += messages
    print(messages_with_system)
    res=pipeline.chat(messages_with_system,max_length=2048,
                max_context_length=2048,
                do_sample=0.8 > 0,
                top_k=0,
                top_p=0.8,
                temperature=0.8,
                repetition_penalty=1.0,
                num_threads=0,
                stream=False,)
    
    print(res)
    logging.info("End Loading chatglm model")
    


def stream_chat(messages, body):
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(role="assistant"))],
    )

    for chunk in pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=settings.num_threads,
        stream=True,
    ):
        yield ChatCompletionResponse(
            object="chat.completion.chunk",
            choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=chunk.content))],
        )

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )




@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)

    max_context_length =body.max_context_length | 2048
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )
    logging.info(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[ChatCompletionResponseChoice(message=ChatMessage(role="assistant", content=output.content))],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "owner"
    permission: List = []


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "owner", "permission": []}],
                }
            ]
        }
    }


# @app.get("/v1/models")
# async def list_models() -> ModelList:
#     return ModelList(data=[ModelCard(id="gpt-3.5-turbo")])

def start():
    uvicorn.run(app, host=settings.host, port=settings.port)
    

if __name__ == "__main__":
    start()
