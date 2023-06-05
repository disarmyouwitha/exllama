import json
import glob
import time
import socket
import asyncio
import uvicorn
import requests
from typing import Union
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from typing import Any, Dict, Optional, List
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
# exllama imports:
from webui.session import prepare_sessions, get_initial_session, Session, load_session, new_session
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from webui.init import init_model
from threading import Timer, Lock
import argparse
import torch
import sys
import os

#meow

# Setup FastAPI:
app = FastAPI()
semaphore = asyncio.Semaphore(1)
templates = Jinja2Templates(directory = str(Path().resolve()))
generate_lock = Lock()
session: Session

# workers
nodes = ["wintermute:7862"]
busy_nodes = set()

# I need open CORS for my setup, you may not!!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#-------


# Chat. Just a wrapper for the HTML page. This way you can hit it from mobile on your network. =]ss
@app.get("/")
async def chat(request: Request, q: Union[str, None] = None):
    return templates.TemplateResponse("fastapi_chat.html", {"request": request, "host": socket.gethostname(), "port": _PORT})

@app.get("/chat")
async def chat(request: Request, q: Union[str, None] = None):
    return templates.TemplateResponse("fastapi_chat.html", {"request": request, "host": socket.gethostname(), "port": _PORT})


# fastapi_chat.html uses this to check what model is being used.
# (My Webserver uses this to check if my LLM is running):
@app.get("/check")
def check():
    # just return name without path or safetensors so we don't expose local paths:
    model = os.path.basename(args.model).replace(".safetensors", "")

    return { model }


# You know.. just in case.
@app.get("/kill")
async def kill():
    import signal

    # Send the kill signal to terminate the script:
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)

    return "Shutdown initiated."


class GenerateRequest(BaseModel):
    message: str
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.65
    min_p: Optional[float] = 0.06
    token_repetition_penalty_max: Optional[float] = 1.15
    token_repetition_penalty_sustain: Optional[int] = 256
    token_repetition_penalty_decay: Optional[int] = None
    stream: Optional[bool] = True
    # [extra options]:
    log: Optional[bool] = True
    benchmark: Optional[bool] = False # change to benchmark in DB
    #break_on_newline:
    #custom_stopping_str


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    """
    # place user message into prompt:
    if req.prompt:
        _MESSAGE = req.prompt.replace("{user_input}", req.message)
    else:
        _MESSAGE = req.message
    print(_MESSAGE)
    """

    with generate_lock:
        return StreamingResponse(session.respond_multi(req.message))
    #return StreamingResponse(generate_simple(_MESSAGE, req.max_new_tokens))

#-------

"""
@app.post("/ask")
async def ask_endpoint(request: Request):
    data = await request.json()

    global nodes, busy_nodes


    for node in nodes:
        if node in busy_nodes:
            continue

        node_url = f"http://{node}/generate"

        try:
            busy_nodes.add(node)

            print(f"Trying on {node_url}")

            #async with aiohttp.ClientSession() as session: (??? async instead?)
            r = requests.post(f"{node_url}", data=json.dumps(data), stream=True)
            return StreamingResponse(r.iter_content())

        except Exception as e:
            print.info(f"Node unreachable: {node_url}")
            pass
        finally:
            busy_nodes.remove(node)

    raise HTTPException(status_code=503, detail="All nodes are currently unavailable")
"""
#-------


if __name__ == "__main__":
    # Load the model
    model, tokenizer, machine = init_model()
    _host, _port = machine.split(":")

    # Get the session ready
    prepare_sessions(model, tokenizer)
    session = get_initial_session()

    # [start fastapi]:
    uvicorn.run(
        app,
        host=_host,
        port=_port
    )