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
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os


# [init torch]:
torch.set_grad_enabled(False)
torch.cuda._lazy_init()
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

# [Parse arguments]:
parser = argparse.ArgumentParser(description = "Simple FastAPI wrapper for ExLlama")

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path",default = None)
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)", default = None)
parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")
parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
# Do we want to bring over any more flags?

args = parser.parse_args()

# Directory check:
if args.directory is not None:
    args.tokenizer = os.path.join(args.directory, "tokenizer.model")
    args.config = os.path.join(args.directory, "config.json")
    st_pattern = os.path.join(args.directory, "*.safetensors")
    st = glob.glob(st_pattern)
    if len(st) == 0:
        print(f" !! No files matching {st_pattern}")
        sys.exit()
    if len(st) > 1:
        print(f" !! Multiple files matching {st_pattern}")
        sys.exit()
    args.model = st[0]
else:
    if args.tokenizer is None or args.config is None or args.model is None:
        print(" !! Please specify -d")
        sys.exit()
#-------


# Setup FastAPI:
app = FastAPI()
semaphore = asyncio.Semaphore(1)
templates = Jinja2Templates(directory = str(Path().resolve()))

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
    while True:
        try:
            # Attempt to acquire the semaphore without waiting, in a loop...
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            print("Server is busy")
            await asyncio.sleep(1)
    
    try:
        # start timer:
        t0 = time.time()

        # place user message into prompt:
        if req.prompt:
            _MESSAGE = req.prompt.replace("{user_input}", req.message)
        else:
            _MESSAGE = req.message
        #print(_MESSAGE)

        # Set these from GenerateRequest:
        generator.settings = ExLlamaGenerator.Settings()
        generator.settings.temperature = req.temperature
        generator.settings.top_k = req.top_k
        generator.settings.top_p = req.top_p
        generator.settings.min_p = req.min_p
        generator.settings.token_repetition_penalty_max = req.token_repetition_penalty_max
        generator.settings.token_repetition_penalty_sustain = req.token_repetition_penalty_sustain
        decay = int(req.token_repetition_penalty_decay if req.token_repetition_penalty_decay else req.token_repetition_penalty_sustain / 2)
        generator.settings.token_repetition_penalty_decay = decay

        if req.stream:
            # copy of generate_simple() so that I could yield each token for streaming without having to change generator.py and make merging updates a nightmare:
            async def generate_simple(prompt, max_new_tokens):
                t0 = time.time()
                new_text = ""
                last_text = ""
                _full_answer = ""

                generator.end_beam_search()

                ids = tokenizer.encode(prompt)
                generator.gen_begin_reuse(ids)

                for i in range(max_new_tokens):
                    token = generator.gen_single_token()
                    text = tokenizer.decode(generator.sequence[0])
                    new_text = text[len(_MESSAGE):]

                    # Get new token by taking difference from last response:
                    new_token = new_text.replace(last_text, "")
                    last_text = new_text

                    #print(new_token, end="", flush=True)
                    yield new_token

                    # [End conditions]:
                    #if break_on_newline and # could add `break_on_newline` as a GenerateRequest option?
                    #if token.item() == tokenizer.newline_token_id:
                    #    print(f"newline_token_id: {tokenizer.newline_token_id}")
                    #    break
                    if token.item() == tokenizer.eos_token_id:
                        #print(f"eos_token_id: {tokenizer.eos_token_id}")
                        break

                # all done:
                generator.end_beam_search() 
                _full_answer = new_text
                _params = ' '.join(sys.argv[1:])
                _benchmark = req.benchmark

                # get num new tokens:
                prompt_tokens = tokenizer.encode(_MESSAGE)
                prompt_tokens = len(prompt_tokens[0])
                new_tokens = tokenizer.encode(_full_answer)
                new_tokens = len(new_tokens[0])

                # calc tokens/sec:
                t1 = time.time()
                _sec = t1-t0
                _tokens_sec = new_tokens/(_sec)

                #print(f"full answer: {_full_answer}")

                print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

                if req.log==True:
                    import pymysql

                    # Establish a connection to the database:
                    db_pw = os.environ.get('DB_PW')
                    connection = pymysql.connect(
                        host='localhost',
                        user='nap',
                        password=f'{db_pw}',
                        database='wntr',
                        charset='utf8mb4',
                        cursorclass=pymysql.cursors.DictCursor
                    )

                    # Execute an insert query
                    try:
                        with connection.cursor() as cursor:
                            model = os.path.basename(args.model).replace(".safetensors", "")
                            sql = "INSERT INTO llm_logs (model, temp, prompt, question, answer, new_tokens, token_sec, bits_loaded, context, run_params, benchmark) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                            values = (model, req.temperature, req.prompt, req.message, _full_answer, new_tokens, _tokens_sec, 4, prompt_tokens, _params, _benchmark)

                            # insert into DB:
                            cursor.execute(sql, values)
                            connection.commit()
                            #print(cursor.rowcount, "record inserted.")
                    finally:
                        # Close the connection
                        connection.close()

            return StreamingResponse(generate_simple(_MESSAGE, req.max_new_tokens))
        else:
            # No streaming, using generate_simple:
            text = generator.generate_simple(_MESSAGE, req.max_new_tokens)
            #print(text)

            # remove prompt from response:
            new_text = text.replace(_MESSAGE,"")
            new_text = new_text.lstrip()
            #print(new_text)

            # get num new tokens:
            prompt_tokens = tokenizer.encode(_MESSAGE)
            prompt_tokens = len(prompt_tokens[0])
            new_tokens = tokenizer.encode(new_text)
            new_tokens = len(new_tokens[0])

            # calc tokens/sec:
            t1 = time.time()
            _sec = t1-t0
            _tokens_sec = new_tokens/(_sec)

            print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

            # return response time here?
            return { new_text }
    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()
#-------


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
#-------


if __name__ == "__main__":
    # Config:
    config = ExLlamaConfig(args.config)
    config.set_auto_map(args.gpu_split)
    config.model_path = args.model
    config.max_seq_len = 2048

    # [Instantiate model and generator]:
    model = ExLlama(config)
    cache = ExLlamaCache(model)
    tokenizer = ExLlamaTokenizer(args.tokenizer)
    generator = ExLlamaGenerator(model, tokenizer, cache)

    # Some feedback
    print(f" -- Loading model")
    print(f" -- Tokenizer: {args.tokenizer}")
    print(f" -- Model config: {args.config}")
    print(f" -- Model: {args.model}")
    print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
    #-------

    # [start fastapi]:
    _PORT = 7862
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=_PORT
    )