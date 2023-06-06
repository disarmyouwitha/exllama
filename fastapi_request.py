import requests
import json

data = {
    "prompt": "You are Samantha, a sentient AI.\nUSER: {user_input}\nASSISTANT:",
    "message": "Tell me about yourself!",
    "max_new_tokens": 256
    # [DEFAULTS]:
    #temperature: 0.7,
    #top_k: 20,
    #top_p: 0.65,
    #min_p: 0.06,
    #token_repetition_penalty_max: 1.15,
    #token_repetition_penalty_sustain: 256,
    #token_repetition_penalty_decay: 128,
    #stream: true,
}

r = requests.post("http://localhost:7862/generate", json=data, stream=True)

if r.status_code==200:
    for chunk in r.iter_content():
        if chunk:
            print(chunk.decode("utf-8"), end="", flush=True)
else:
    print("Request failed with status code:", r.status_code)












