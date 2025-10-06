from smolagents import Tool
import math
import re
import requests
from typing import Optional, List, Dict, Any

class CheckNumericClaim(Tool):
    name = "check_numeric_claim"
    description = ("Given a hypothesis like 'the area of a circle with r=3 is 28', "
                   "compute truth value and return a short finding.")
    inputs = {
        "hypothesis": {"type": "string", "description": "short natural-language numeric claim"}
    }
    output_type = "string"

    def forward(self, hypothesis: str) -> str:
        m = re.search(
            r"area of a circle with r\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*is\s*([0-9]+(?:\.[0-9]+)?)",
            hypothesis.lower(),
        )
        if not m:
            return "Could not parse hypothesis; provide 'area of a circle with r=R is X'."
        r = float(m.group(1)); x = float(m.group(2))
        true_val = math.pi * r * r
        ok = abs(true_val - x) < 0.5  # lenient tolerance
        return f"parsed_r={r}, parsed_x={x}, truth≈{true_val:.2f}, satisfies={ok}"

class HttpRAGTool(Tool):
    name = "web_rag"
    description = ("Query a running RAG microservice. Inputs: question (str), "
                   "urls (list[str], optional), top_k (int, optional). Returns top passages.")
    inputs = {
        "question": {"type": "string", "description": "Question to answer", "nullable": True},
        "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to ingest", "nullable": True},
        "top_k": {"type": "integer", "description": "How many passages", "nullable": True},
    }
    output_type = "string"

    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__()
        self.base_url = base_url.rstrip("/")

    def _ingest_urls(self, urls: Optional[List[str]] = None) -> Dict[str, Any]:
        urls = urls or []
        if not urls:
            return {}
        r = requests.post(f"{self.base_url}/ingest", json={"urls": urls}, timeout=30)
        r.raise_for_status()
        return r.json().get("added", {})

    def forward(self, question: Optional[str] = None, urls: Optional[List[str]] = None, top_k: Optional[int] = 4) -> str:
        if urls:
            try:
                self._ingest_urls(urls)
            except Exception:
                pass
        if not question:
            return "Please provide a 'question'."
        r = requests.post(f"{self.base_url}/ask", json={"question": question, "top_k": top_k or 4}, timeout=30)
        r.raise_for_status()
        return r.json().get("context", "")