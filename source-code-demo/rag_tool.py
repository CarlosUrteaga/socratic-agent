from smolagents import Tool
from typing import List, Optional, Tuple, Dict, Any
import requests, re, math
from bs4 import BeautifulSoup
from collections import Counter, defaultdict

DEFAULT_HEADERS = {
    "User-Agent": "Socratic_agent/1.0 (contact: youremail@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en;q=0.9",
}

def _wikipedia_plain_url(url: str) -> str | None:
    m = re.search(r"wikipedia\.org/wiki/([^?#]+)", url)
    if not m:
        return None
    title = m.group(1)
    return f"https://en.wikipedia.org/api/rest_v1/page/plain/{title}"

def _fetch_text(url: str) -> str:
    # 1) Try regular HTML with headers
    resp = requests.get(url, timeout=20, headers=DEFAULT_HEADERS, allow_redirects=True)
    if resp.status_code == 200 and resp.text.strip():
        return resp.text

    # 2) If blocked and it’s a wikipedia /wiki/ URL, retry via REST plain-text
    if resp.status_code in (403, 429) and "wikipedia.org/wiki/" in url:
        api_url = _wikipedia_plain_url(url)
        if api_url:
            alt = requests.get(api_url, timeout=20, headers=DEFAULT_HEADERS)
            alt.raise_for_status()
            return alt.text

    # 3) As a general fallback, raise
    resp.raise_for_status()
    return resp.text
    
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return _clean(text)

def _chunk(text: str, chunk_words: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        part = words[i:i + chunk_words]
        if not part:
            break
        chunks.append(" ".join(part))
        i += max(1, chunk_words - overlap)
    return chunks

# --------------------------- BM25 (pure Python) ---------------------------

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

class BM25Index:
    """A tiny, dependency-free BM25 index for small corpora."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.texts: List[str] = []
        self.metas: List[dict] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.avgdl = 0.0

    def add_documents(self, chunks: List[str], metadatas: Optional[List[dict]] = None):
        if not chunks:
            return
        metadatas = metadatas or [{} for _ in chunks]
        for text, meta in zip(chunks, metadatas):
            toks = _tokenize(text)
            self.docs.append(toks)
            self.texts.append(text)
            self.metas.append(meta)
            for term in set(toks):
                self.df[term] += 1
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, idx: int) -> float:
        q_toks = _tokenize(query)
        doc = self.docs[idx]
        freq = Counter(doc)
        score = 0.0
        dl = len(doc)
        for t in q_toks:
            if t not in freq:
                continue
            idf = self._idf(t)
            f = freq[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
            score += idf * (f * (self.k1 + 1)) / max(1e-9, denom)
        return score

    def search(self, query: str, top_k: int = 4) -> List[Tuple[float, str, dict]]:
        if not self.docs:
            return []
        scores = [(self.score(query, i), self.texts[i], self.metas[i]) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

class WebRAGTool(Tool):
    name = "web_rag"
    description = (
        "Retrieve relevant passages from given URLs using BM25 (no heavy deps) "
        "and return a compact context for answering a question."
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "Question to answer with retrieved context.",
            "nullable": True,             # ← was False / missing; set to True
        },
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Web URLs to ingest before querying.",
            "nullable": True,             # ← must be True for optional
        },
        "top_k": {
            "type": "integer",
            "description": "Number of passages to return (default 4)",
            "nullable": True,             # ← must be True for optional
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.idx = BM25Index()
        self.seen_urls = set()

    def forward(
        self,
        question: Optional[str] = None,  # ← accept None to match nullable=True
        urls: Optional[List[str]] = None,
        top_k: Optional[int] = 4,
    ) -> str:
        if not question:
            return "Please provide a 'question' string."
        urls = urls or []
        top_k = top_k or 4
        for u in urls:
            try:
                self._ingest_url(u)
            except Exception:
                pass
        hits = self.idx.search(question, top_k=top_k)
        if not hits:
            return "No context available yet; add URLs or content first."
        bullets = []
        for score, passage, meta in hits:
            cite = meta.get("url", "N/A")
            # in rag_tool.py, build bullets:
            snippet = (passage[:220] + "…") if len(passage) > 220 else passage
            bullets.append(f"- score={score:.3f} | {snippet} (source: {cite})")
        return "Top passages for: " + question + "\n" + "\n".join(bullets)

    def _ingest_url(self, url: str) -> int:
        if not url or url in self.seen_urls:
            return 0
        html_or_text = _fetch_text(url)
        # If we got HTML, extract; if we got plain text (Wikipedia REST), _extract will just clean whitespace fine.
        text = _extract_text_from_html(html_or_text)
        chunks = _chunk(text)
        metas = [{"url": url, "chunk": i} for i in range(len(chunks))]
        self.idx.add_documents(chunks, metas)
        self.seen_urls.add(url)
        return len(chunks)

    # NEW: public method (no underscore)
    def ingest_url(self, url: str) -> int:
        return self._ingest_url(url)

    # NEW: convenience for a list of URLs
    def ingest_urls(self, urls: Optional[List[str]] = None) -> dict:
        urls = urls or []
        out = {}
        for u in urls:
            try:
                out[u] = self._ingest_url(u)
            except Exception as e:
                out[u] = f"error: {e}"
        return out
