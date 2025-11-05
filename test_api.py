#!/usr/bin/env python3
"""
Test API Server
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting test API on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
