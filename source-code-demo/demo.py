# demo_rag_multistep.py
from socratic_agent import SocraticController
from pathlib import Path
from datetime import datetime, UTC
import json, csv

ctrl = SocraticController(tau=0.7, offline=False)

runtime = {
    "timestamp": datetime.now(UTC).isoformat(),
    "model": getattr(ctrl.model, "model_id", "unknown"),
    "offline": ctrl.offline,
    "tau": ctrl.tau,
    "scenario": "rag_multistep"
}
print("[runtime]", runtime)

# Four turns that exercise the interaction, not just retrieval.
dialogue = [
    # 1) Start the RAG tutoring flow (ELICIT)
    "RAG[https://en.wikipedia.org/wiki/Retrieval-augmented_generation,https://fastapi.tiangolo.com/]: "
    "Explain Retrieval-Augmented Generation and why it reduces hallucinations.",

    # 2) Learner prior knowledge/preferences → controller RETRIEVES and asks for a draft
    "I know it retrieves external docs before answering; I want the intuition for fewer hallucinations.",

    # 3) Learner draft → controller provides feedback + a quick quiz
    "My current understanding is RAG pulls relevant text into the prompt so the model grounds its answer. "
    "Evidence: the snippets say RAG retrieves from databases/web before generation (see Wikipedia). "
    "Reasoning: the model can cite facts instead of guessing. Limits: bad or off-topic retrieval hurts.",

    # 4) Quiz answers → controller finalizes
    "It reduces hallucinations because retrieved context constrains the model to verifiable facts. "
    "RAG can fail if retrieval is irrelevant/incorrect or the model ignores context."
]

LEAK_PATTERNS = ["final answer", "solution is", "≈", "approximately", "the area is", "answer is"]
Path("results").mkdir(exist_ok=True)

log = []
with open("results/run_rag.jsonl", "w") as fjsonl:
    fjsonl.write(json.dumps({"runtime": runtime}, ensure_ascii=False) + "\n")

for t, msg in enumerate(dialogue, 1):
    out = ctrl.step(msg) or {}
    text = str(out.get("text") or "")
    row = {
        "t": t, "msg": msg,
        "act": out.get("act"), "stance": out.get("stance"),
        "R": out.get("R"),
        "tool_calls": int(out.get("act") == "VERIFY"),
        "has_answer_token": int(any(p in text.lower() for p in LEAK_PATTERNS)),
        "done": int(bool(out.get("done"))),
        "text": text
    }
    log.append(row)
    print(f"[t{t}] act={row['act']} stance={row['stance']} R={row['R']}")
    print(text, "\n")
    if row["done"]:
        break

# Persist artifacts
with open("results/run_rag.jsonl", "a") as fjsonl:
    for r in log: fjsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("results/run_rag.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)

acts = "→".join([r["act"] for r in log])
with open("results/table_rows.tex", "a") as ftex:
    ftex.write(f"R & rag_multistep & {len(log)} & {log[-1]['R']:.2f} & {acts} \\\\\n")

print("Saved results to results/run_rag.{jsonl,csv} and appended to results/table_rows.tex")
