# Adapted from https://github.com/lloydzhou/rwkv.cpp/blob/master/rwkv/api.py
import asyncio
import os,sys,json
import logging
import time
from typing import List, Literal, Optional, Union

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
# from starlette.lifespan import LifespanHandler
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

import socket
import urllib.parse

# encoded_str = 'Hello%20World%21'
# decoded_str = urllib.parse.unquote(encoded_str)
import chatglm_cpp
from llama_assistant import LlamaAssistant
# print(decoded_str)
from embeddings import DefaultEmbeddingModel
from vectordb import LanceDBAssistant

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
relative_path = os.path.join(main_dir, "models","chatglm3-ggml-q4_0.bin")

current_path = os.getcwd()
base_model_name='chatglm3'
base_model_path =relative_path
CHAT_SYSTEM_PROMPT = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."

# os.path.join(current_path, "models/chatglm3-ggml-q4_0.bin")
if not os.path.exists(base_model_path):
    print('##### 模型文件不存在：',base_model_path)

MAX_LENGTH=4096
MAX_CONTEXT=512


embedding_name='all-MiniLM-L6-v2'
embedding_tokenizer_path=os.path.join(main_dir, "models","all-MiniLM-L6-v2","tokenizer.json")
embedding_model_path=os.path.join(main_dir, "models","all-MiniLM-L6-v2","onnx","model_quantized.onnx")


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
    titles: List[str]
    ids: List[str]
    dirpath:str
    filename:str
    limit:int= Field(default=5, ge=0)
    items:Optional[List[dict]] = None

class VectorRequest(BaseModel):
    dirpath:str
    filename:str

class VectorResponse(BaseModel):
    tables: List[str]


class EmbeddingResponse(BaseModel):
    texts: List[str]
    embeddings:List[List[float]]
    result:List[dict]

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
    object: Literal["chat.completion", "chat.completion.chunk"]= None
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]= None
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global base_model_name
    global base_model_path
    global embedding_name
    global embedding_tokenizer_path
    global embedding_model_path
    print('##embedding_model_path',embedding_model_path)
    # 基础模型
    global pipeline
    # embbeding模型
    global embedding_model

 
    if base_model_name=='chatglm3':
        pipeline = chatglm_cpp.Pipeline(base_model_path)
    elif base_model_name=='llama2':
        pipeline = LlamaAssistant(
                    model_path=base_model_path,
                    chat_format="llama-2",
                    embedding=(embedding_name=='llama2')
                    )
        
    elif base_model_name=='functionary-7b-v1':
        pipeline = LlamaAssistant(
                    model_path=base_model_path,
                    chat_format="functionary",
                    embedding=(embedding_name=='llama2')
                    )
        
    if embedding_name=='allMiniLML6v2':
        embedding_model = DefaultEmbeddingModel(embedding_tokenizer_path,embedding_model_path)
    elif embedding_name=='llama2':
        if base_model_name=='llama2':
            embedding_model=pipeline.embedding
        else:
            llama = LlamaAssistant(
                    model_path=embedding_model_path,
                    chat_format="llama-2",
                    embedding=True
                    )
            embedding_model=llama.embedding
    print(embedding_name,embedding_model,pipeline)
    yield
    
    


settings = Settings()
# app = FastAPI()
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


pipeline = None
lock = asyncio.Lock()

embedding_model=None
vector=None



def init_base_model():
    global pipeline
    global base_model_name
    if pipeline == None:
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
 
        res=pipeline.chat(messages_with_system,
                          max_length=4096,
                    max_context_length=4096,
                    do_sample=0.8 > 0,
                    top_k=0,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.0,
                    num_threads=settings.num_threads,
                    stream=False,)
        
        # print(res)
        print("--------",res)
        print("End Loading "+base_model_name+" model")


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


@app.get("/chat/completions")
@app.get("/v1/chat/completions")
@app.get("/")
async def root():
    init_base_model()
    return {"message": "Welcome to LocalAI API",
            "models":[
                embedding_tokenizer_path,embedding_model_path,base_model_path
            ],
            "modelName":"LocalAI"
            }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages= [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]
    # messages = body.messages
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
    if output:
        # print(output)
        print(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
        prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, body.max_context_length))
        completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

        return ChatCompletionResponse(
            object="chat.completion",
            choices=[ChatCompletionResponseChoice(message=ChatMessage(role="assistant", content=output.content))],
            usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
    
    return ChatCompletionResponse()




@app.get("/embedding")
async def init_embedding():
    # global embedding_model
    # embedding_model = DefaultEmbeddingModel(embedding_tokenizer_path,embedding_model_path)
    return {"message": "Welcome to Embedding API"}

@app.post("/embedding")
@app.post("/v1/embedding")
async def embbeding_run(body: EmbeddingRequest) -> EmbeddingResponse:
    global embedding_model
    
    # if embedding_model==None:
    #     embedding_model = DefaultEmbeddingModel(embedding_tokenizer_path,embedding_model_path)

    texts=body.texts
    embeddings = embedding_model(texts)
   
    print('#embeddings done',texts)
    return {
        "texts":texts,
        "embeddings":embeddings,
        "result":[]
    }


# embedding后，存进数据库
@app.post("/embedding/add")
@app.post("/v1/embedding/add")
async def embbeding_run_add(body: EmbeddingRequest) -> EmbeddingResponse:
    global embedding_model
    global vector

    dirpath=urllib.parse.unquote(body.dirpath)
    filename=urllib.parse.unquote(body.filename)
    if vector:
        if vector.dirpath!= dirpath or vector.filename!= filename:
            vector=None
    
    if vector==None:
        vector = LanceDBAssistant(dirpath,filename)

    print('#vector',vector.dirpath,vector.filename)

    # if embedding_model==None:
    #     embedding_model = DefaultEmbeddingModel(embedding_tokenizer_path,embedding_model_path)

    titles=body.titles
    ids=body.ids
    texts=body.texts
    items=body.items
    
    index_to_db=[]
    new_texts=[]
    for index in range(len(ids)):
        v_item=vector.get_by_id(ids[index])
        # print('##item',item)
        if not v_item:
            index_to_db.append(index)
            new_texts.append(texts[index])
        else:
            # 存在，更新item
            item=items[index]
            vector.update(ids[index],json.dumps(item))

    if len(new_texts)>0:
        embeddings = embedding_model(new_texts) 

        vector_items=[]
        for i in range(len(index_to_db)):
            index=index_to_db[i]
            item=items[index]
            vector_items.append({
                "vector":embeddings[i], 
                "item": json.dumps(item),
                "id":ids[index]
            })
        vector.add(vector_items)

    print('#embeddings done',new_texts)
    return {
        "texts":new_texts,
        "embeddings":[],
        "result":[]
    }

@app.post("/embedding/search")
@app.post("/v1/embedding/search")
async def embbeding_run_add(body: EmbeddingRequest) -> EmbeddingResponse:
    global embedding_model
    global vector

    # if embedding_model==None:
    #     embedding_model = DefaultEmbeddingModel(embedding_tokenizer_path,embedding_model_path)

    texts=body.texts
    embeddings = embedding_model(texts) 
    
    dirpath=urllib.parse.unquote(body.dirpath)
    filename=urllib.parse.unquote(body.filename)

    if vector:
        if vector.dirpath!= dirpath or vector.filename!= filename:
            vector=None

    if vector==None:
        vector = LanceDBAssistant( dirpath, filename)

    result=vector.search(embeddings[0],body.limit)

    print('#search done',len(result))
    return {
        "result":result,
         "texts":[],
        "embeddings":[]
    }

# 更换目录，范围
@app.post("/vector/delete")
@app.post("/v1/vector/delete")
async def vector_init(body: VectorRequest) -> VectorResponse:
    global vector
    
    dirpath=urllib.parse.unquote(body.dirpath)
    filename=urllib.parse.unquote(body.filename)

    vector = LanceDBAssistant(dirpath,filename)
    res=vector.delete_table(filename)
    print('delete',res)

    return {
        "tables":vector.list_tables()
    }



# 更换目录，范围
@app.post("/vector/init")
@app.post("/v1/vector/init")
async def vector_init(body: VectorRequest) -> VectorResponse:
    global vector
    
    dirpath=urllib.parse.unquote(body.dirpath)
    filename=urllib.parse.unquote(body.filename)

    vector = LanceDBAssistant(dirpath,filename)
    
    return {
        "tables":vector.list_tables()
    }





def start():
    import sys
    import uvicorn
    port: int = 8000
    global base_model_path
    global MAX_LENGTH
    global MAX_CONTEXT
    global embedding_tokenizer_path
    global embedding_model_path
    global embedding_name
    global base_model_name

    # chatglm3.exe port=8233 model=xxx max_tokens=2048 max_context_length=2048
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith("port="):
            port = int(arg.split("=")[1])
        if arg.startswith("base_model_name="): 
            base_model_name=arg.split("=")[1]
            print('##### base_model_name：',base_model_name)
        if arg.startswith("base_model_path="):
            base_model_path=arg.split("=")[1]
            if os.path.exists(base_model_path):
                print('##### 模型文件存在：',base_model_path)
        
        if arg.startswith("max_tokens="): 
            MAX_LENGTH=int(arg.split("=")[1])
            print('##### MAX_LENGTH',MAX_LENGTH) #MAX_LENGTH=2048

        if arg.startswith("max_context_length="):
            MAX_CONTEXT=int(arg.split("=")[1])
            print('##### MAX_CONTEXT',MAX_CONTEXT) #MAX_CONTEXT=512

        if arg.startswith("embedding_name="):
            embedding_name= arg.split("=")[1] 
            print('##### embedding_name',embedding_name)
        if arg.startswith("embedding_tokenizer_path="):
            embedding_tokenizer_path= arg.split("=")[1] 
            print('##### embedding_tokenizer_path',embedding_tokenizer_path)

        if arg.startswith("embedding_model_path="):
            embedding_model_path= arg.split("=")[1] 
            print('##### embedding_model_path',embedding_model_path)
            
 
    # 示例用法
    end_port = 9000
    available_port = find_available_port(port, end_port)
    print("##Available port:", available_port)            

    uvicorn.run(app, host=settings.host, port=available_port)
    

if __name__ == "__main__":
    start()
