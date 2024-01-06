# Adapted from https://github.com/lloydzhou/rwkv.cpp/blob/master/rwkv/api.py
import asyncio
import os,sys
import logging
import time
from typing import List, Literal, Optional, Union

import chatglm_cpp
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

import socket

from embeddings import DefaultEmbeddingModel


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def find_available_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        if not is_port_in_use(port):
            return port
    raise Exception("No available ports in the range")




# 获取可执行文件的路径
executable_path = sys.argv[0]
# 获取main.py文件所在的目录路径
main_dir = os.path.dirname(os.path.abspath(executable_path))

# 获取相对路径
relative_path = os.path.join(main_dir, "models/chatglm3-ggml-q4_0.bin")

current_path = os.getcwd()
DEFAULT_MODEL_PATH =relative_path
CHAT_SYSTEM_PROMPT = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."

# os.path.join(current_path, "models/chatglm3-ggml-q4_0.bin")
if not os.path.exists(DEFAULT_MODEL_PATH):
    print('##### 模型文件不存在：',DEFAULT_MODEL_PATH)

MAX_LENGTH=4096
MAX_CONTEXT=512



embbeding_tokenizer_path=os.path.join(main_dir, "models/all-MiniLM-L6-v2/tokenizer.json")
embbeding_model_path=os.path.join(main_dir, "models/all-MiniLM-L6-v2/onnx/model_quantized.onnx")


class Settings(BaseSettings):
    # model: str = "chatglm-ggml.bin"
    num_threads: int = 8
    server_name: str = "ChatGLM3 CPP API Server"
    
    host: str = "127.0.0.1"
    port: int = 8000
    print("####线程数",num_threads)



class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    texts: List[str]
    embeddings:List[List[float]]

class ChatCompletionRequest(BaseModel):
    model: str = "ChatGLM3"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=4096, ge=0)
    max_context_length: int = Field(default=4096, ge=0)

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

embbeding_model=None


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = chatglm_cpp.Pipeline(DEFAULT_MODEL_PATH)


def init_chatglm3():
    global pipeline

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
    print("--------")
    print(messages_with_system)
    print("--------")
    res=pipeline.chat(messages_with_system,max_length=4096,
                max_context_length=4096,
                do_sample=0.8 > 0,
                top_k=0,
                top_p=0.8,
                temperature=0.8,
                repetition_penalty=1.0,
                num_threads=settings.num_threads,
                stream=False,)
    
    print(res)
    print("--------")
    print("End Loading chatglm model")


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


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
            for chunk in stream_chat(history, body):
                await asyncio.sleep(0)  # yield control back to event loop for cancellation check
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        print(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        print(f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]
    print('Message length',len(messages))
    print('------')
    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)
    
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=body.max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )
    print(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, body.max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[ChatCompletionResponseChoice(message=ChatMessage(role="assistant", content=output.content))],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@app.get("/")
async def root():
    init_chatglm3()
    return {"message": "Welcome to ChatGLM3 API"}


@app.get("/embedding")
async def init_embedding():
    global embbeding_model
    embbeding_model = DefaultEmbeddingModel(embbeding_tokenizer_path,embbeding_model_path)
    return {"message": "Welcome to Embedding API"}

@app.post("/embedding")
@app.post("/v1/embedding")
async def embbeding_run(body: EmbeddingRequest) -> EmbeddingResponse:
    global embbeding_model
    
    if embbeding_model==None:
        embbeding_model = DefaultEmbeddingModel(embbeding_tokenizer_path,embbeding_model_path)

    texts=body.texts
    embeddings = embbeding_model(texts)
    embeddings=embeddings.tolist()
    print('#embeddings done',texts)
    return {
        "texts":texts,
        "embeddings":embeddings
    }


def start():
    import sys
    import uvicorn
    port: int = 8000
    global DEFAULT_MODEL_PATH
    global MAX_LENGTH
    global MAX_CONTEXT
    global embbeding_tokenizer_path
    global embbeding_model_path

    # chatglm3.exe port=8233 model=xxx max_tokens=2048 max_context_length=2048
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("port="):
            port = int(arg.split("=")[1])
        if arg.startswith("model="):
            DEFAULT_MODEL_PATH=arg.split("=")[1]
            if os.path.exists(DEFAULT_MODEL_PATH):
                print('##### 模型文件存在：',DEFAULT_MODEL_PATH)
        if arg.startswith("max_tokens="): 
            MAX_LENGTH=int(arg.split("=")[1])
            print('##### MAX_LENGTH',MAX_LENGTH) #MAX_LENGTH=2048

        if arg.startswith("max_context_length="):
            MAX_CONTEXT=int(arg.split("=")[1])
            print('##### MAX_CONTEXT',MAX_CONTEXT) #MAX_CONTEXT=512

        if arg.startswith("embbeding_tokenizer_path="):
            embbeding_tokenizer_path= arg.split("=")[1] 
            print('##### embbeding_tokenizer_path',embbeding_tokenizer_path)

        if arg.startswith("embbeding_model_path="):
            embbeding_model_path= arg.split("=")[1] 
            print('##### embbeding_model_path',embbeding_model_path)
            
 
    # 示例用法
    end_port = 9000
    available_port = find_available_port(port, end_port)
    print("##Available port:", available_port)            

    uvicorn.run(app, host=settings.host, port=available_port)
    

if __name__ == "__main__":
    start()
