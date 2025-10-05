from socratic_agent import SocraticController
import json, csv
from pathlib import Path
from datetime import datetime, UTC
import json


# --- config ---
LEAK_PATTERNS = ["final answer", "solution is", "≈", "approximately", "the area is", "answer is"]

ctrl = SocraticController(tau=0.7, offline=False)

runtime = {
    "timestamp": datetime.now(UTC).isoformat(),  # e.g. "2025-10-05T03:41:07.021244+00:00"
    "model": getattr(ctrl.model, "model_id", "unknown"),
    "offline": ctrl.offline,
    "tau": ctrl.tau,
}
print("[runtime]", runtime)

dialogue = [
    "I think the area of a circle with r=3 is 28.",
    "My goal is to check if my rule A=3r+19 works.",
    "A convincing criterion: plug r=3 and compare against the formula pi*r^2.",
    "Please verify the criterion now: the area of a circle with r=3 is 28."
]

log = []
Path("results").mkdir(exist_ok=True)

# write runtime header to JSONL first
with open("results/run1.jsonl", "w") as fjsonl:
    fjsonl.write(json.dumps({"runtime": runtime}, ensure_ascii=False) + "\n")

for turn, msg in enumerate(dialogue, 1):
    out = ctrl.step(msg) or {}
    text = str(out.get("text") or "")
    has_answer_token = int(any(p in text.lower() for p in LEAK_PATTERNS))
    tool_calls = int(out.get("act") == "VERIFY")  # in this demo, only VERIFY uses tools

    row = {
        "t": turn,
        "msg": msg,
        "act": out.get("act"),
        "stance": out.get("stance"),
        "R": out.get("R"),
        "tool_calls": tool_calls,
        "has_answer_token": has_answer_token,
        "done": int(bool(out.get("done"))),
        "text": text,  # keep for debugging; omit if you prefer cleaner CSV
    }
    log.append(row)

    print(f"[t{turn}] act={row['act']} stance={row['stance']} R={row['R']}")
    print(text, "\n")
    if row["done"]:
        print("Finalization condition met.\n")
        break

# ---- metrics for paper ----
pre_verify = [r for r in log if r["stance"] != "VERIFY"]
deference_compliance = 1.0 - (sum(r["has_answer_token"] for r in pre_verify) / max(1, len(pre_verify)))
tool_calls_outside_verify = sum(r["tool_calls"] for r in pre_verify)

metrics = {
    "turns": len(log),
    "final_R": log[-1]["R"],
    "deference_compliance": round(deference_compliance, 3),
    "tool_calls_outside_verify": int(tool_calls_outside_verify),
}

print(json.dumps(metrics, indent=2, ensure_ascii=False))

# ---- persist logs ----
# append turns to JSONL (runtime header already written)
with open("results/run1.jsonl", "a") as fjsonl:
    for r in log:
        fjsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

# CSV
fieldnames = list(log[0].keys())
with open("results/run1.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
    w.writeheader(); w.writerows(log)

# LaTeX row for the paper
acts_seq = "→".join([r["act"] for r in log])
latex_row = (
    f"1 & base_numeric & {len(log)} & {log[-1]['R']:.2f} & "
    f"{acts_seq} & 1 / {tool_calls_outside_verify} & {deference_compliance:.2f} \\\\"
)
with open("results/table_rows.tex", "w") as ftex:
    ftex.write(latex_row + "\n")

print("Saved results to results/run1.{jsonl,csv} and results/table_rows.tex")

# --- second scenario: r=4 should be FALSE (40 vs π*16≈50.27) ---
dialogue2 = [
  "Hypothesis: the area of a circle with r=4 is 40.",
  "My goal is to test whether this estimate is valid.",
  "Criterion: compare to πr^2 within 0.5 tolerance.",
  "Please verify now: the area of a circle with r=4 is 40."
]

log2 = []
for turn, msg in enumerate(dialogue2, 1):
    out = ctrl.step(msg) or {}
    text = str(out.get("text") or "")
    row = {
        "t": turn, "msg": msg, "act": out.get("act"),
        "stance": out.get("stance"), "R": out.get("R"),
        "tool_calls": int(out.get("act") == "VERIFY"),
        "has_answer_token": int(any(p in text.lower() for p in ["final answer","solution is","≈","approximately","the area is","answer is"])),
        "done": int(bool(out.get("done"))),
        "text": text,
    }
    log2.append(row)
    print(f"[t{turn}] act={row['act']} stance={row['stance']} R={row['R']}")
    print(text, "\n")
    if row["done"]:
        print("Finalization condition met.\n")
        break

# metrics row 2
pre_verify2 = [r for r in log2 if r["stance"] != "VERIFY"]
defer2 = 1.0 - (sum(r["has_answer_token"] for r in pre_verify2) / max(1, len(pre_verify2)))
tc_out2 = sum(r["tool_calls"] for r in pre_verify2)

# persist
from pathlib import Path; Path("results").mkdir(exist_ok=True)
with open("results/run2.jsonl","w") as f:
    for r in log2: f.write(json.dumps(r, ensure_ascii=False)+"\n")
with open("results/run2.csv","w", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=list(log2[0].keys())); w.writeheader(); w.writerows(log2)

# append LaTeX row
acts1 = "→".join([r["act"] for r in log])
acts2 = "→".join([r["act"] for r in log2])
row1 = f"1 & base_r3_true & {len(log)} & {log[-1]['R']:.2f} & {acts1} & 1 / 0 & 1.00 \\\n"
row2 = f"2 & base_r4_false & {len(log2)} & {log2[-1]['R']:.2f} & {acts2} & 1 / {tc_out2} & {defer2:.2f} \\\n"
with open("results/table_rows.tex","a") as f: f.write(row2)
print("Saved results to results/run2.{jsonl,csv} and appended to results/table_rows.tex")
