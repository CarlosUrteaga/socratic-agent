
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from rag_tool import WebRAGTool

app = FastAPI(title="Tiny Web RAG (BM25)")
rag = WebRAGTool()

class IngestBody(BaseModel):
    urls: List[str]

class AskBody(BaseModel):
    question: Optional[str] = None   # ← make optional to match tool
    top_k: Optional[int] = 4

@app.post("/ingest")
def ingest(body: IngestBody):
    return {"added": rag.ingest_urls(body.urls)}

@app.post("/ask")
def ask(body: AskBody):
    answer = rag.forward(question=body.question, urls=[], top_k=body.top_k or 4)
    return {"context": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
