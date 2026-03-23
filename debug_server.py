import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello"}

if __name__ == "__main__":
    print(f"DEBUG: nanochat.engine file: {nanochat.engine.__file__}")
    try:
        import inspect
        print(f"DEBUG: sample_next_token file: {inspect.getfile(nanochat.engine.sample_next_token)}")
    except Exception as e:
        print(f"DEBUG: Error getting sample_next_token file: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8012)
