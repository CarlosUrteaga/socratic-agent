# aggregate_results.py
import json, csv, re
from pathlib import Path

RESULTS = Path("results")
RUNS = [
    ("run1.jsonl", "Numeric r=3 (True)"),
    ("run2.jsonl", "Numeric r=4 (False)"),
    ("run3.jsonl", "RAG mini-lesson"),
    # add more here if you create additional runs
]

def load_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # skip header lines like {"runtime": {...}}
            if "runtime" in obj and len(obj) == 1:
                continue
            rows.append(obj)
    return rows

def scenario_metrics(rows):
    if not rows: 
        return None
    turns = len(rows)
    final_R = rows[-1].get("R")
    # pre-VERIFY turns
    pre = [r for r in rows if (r.get("stance") or "").upper() != "VERIFY"]
    leaks = sum(int(r.get("has_answer_token", 0)) for r in pre)
    deference = 1.0 - (leaks / max(1, len(pre)))

    # tool discipline: count tool_calls outside VERIFY
    tool_calls_outside = sum(int(r.get("tool_calls", 0)) for r in pre)
    tool_calls_inside = sum(int(r.get("tool_calls", 0)) for r in rows) - tool_calls_outside
    total_tool_calls = tool_calls_inside + tool_calls_outside
    tool_discipline = 1.0 if total_tool_calls == 0 else (tool_calls_inside / total_tool_calls)

    # success criteria
    last_text = (rows[-1].get("text") or "").lower()
    is_numeric = any("hypothesis" in (r.get("msg","").lower()) for r in rows) or "verify the criterion" in (rows[0].get("msg","").lower())
    if is_numeric:
        success = ("satisfies=true" in last_text)
    else:
        success = bool(int(rows[-1].get("done", 0)) == 1)

    # frustration proxies
    turns_to_finalize = turns
    leakage_rate = leaks / max(1, len(pre))
    return {
        "turns": turns,
        "final_R": final_R,
        "deference": round(deference, 3),
        "tool_discipline": round(tool_discipline, 3),
        "success": success,
        "turns_to_finalize": turns_to_finalize,
        "leakage_rate": round(leakage_rate, 3),
    }

def pick_excerpts(rows, max_len=240):
    """Return 2–3 short, readable snippets that evidence the interaction."""
    out = []
    # first ASK/EXPLORE
    for r in rows:
        if (r.get("act") or "").upper() == "ASK":
            out.append(("ELICIT (ASK/EXPLORE)", r.get("text","")))
            break
    # first VERIFY with snippets
    for r in rows:
        if (r.get("act") or "").upper() == "VERIFY" and "Top passages" in (r.get("text","")):
            out.append(("RETRIEVE (VERIFY)", r.get("text","")))
            break
    # first SUMMARIZE (feedback/quiz)
    for r in rows:
        if (r.get("act") or "").upper() == "SUMMARIZE":
            out.append(("REVIEW+QUIZ (SUMMARIZE)", r.get("text","")))
            break

    # trim
    trimmed = []
    for label, txt in out:
        t = " ".join(txt.split())
        if len(t) > max_len:
            t = t[:max_len].rstrip() + "…"
        trimmed.append((label, t))
    return trimmed

def main():
    RESULTS.mkdir(exist_ok=True)
    metrics_rows = []
    tex_lines = []
    ex_lines = []

    for fname, label in RUNS:
        path = RESULTS / fname
        if not path.exists():
            continue
        rows = load_jsonl(path)
        m = scenario_metrics(rows)
        if not m:
            continue
        metrics_rows.append({
            "scenario": label,
            **m
        })

        # LaTeX row for the main results table
        # aggregate_results.py (in main(), when building tex_lines)
        success_symbol = r"\cmark" if m["success"] else r"\xmark"
        tex_lines.append(
            f"{label} & {m['deference']:.2f} & {m['tool_discipline']:.2f} & "
            f"{success_symbol} & {m['turns']} & {m['final_R']:.2f} & "
            f"{m['leakage_rate']:.2f} & " + ("--" if m["quiz_acc"] is None else str(m["quiz_acc"])) + r" \\"
        )


        # Excerpts (keep short)
        ex = pick_excerpts(rows)
        if ex:
            ex_lines.append(r"\paragraph{" + label + "}")
            ex_lines.append(r"\begin{quote}\small")
            for (lab, t) in ex:
                t = t.replace("%","\\%")  # escape %
                ex_lines.append(r"\textbf{" + lab + r"} " + t)
                ex_lines.append(r"\\[0.2em]")
            ex_lines.append(r"\end{quote}")

    # write CSV summary
    # after computing metrics_rows
    if metrics_rows:
        with open(RESULTS / "metrics_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader(); w.writerows(metrics_rows)
    else:
        print("No runs found to summarize.")

    # write LaTeX table
    table = r"""
\begin{table}[t]
\centering
\caption{Empirical results from executable runs. Deference = no premature answers in non-VERIFY turns. Tool discipline = fraction of tool calls inside VERIFY.}
\label{tab:empirical}
\small
\begin{tabular}{lccccccc}
\toprule
Scenario & Deference$\uparrow$ & Tool$\uparrow$ & Success$\uparrow$ & Turns$\downarrow$ & Final $R$ & Leakage$\downarrow$ & Quiz \\
\midrule
""" + "\n".join(tex_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (RESULTS / "main_results.tex").write_text(table.strip() + "\n")

    # write interaction excerpts
    ex_doc = "\n".join(ex_lines) if ex_lines else "% no excerpts found\n"
    (RESULTS / "excerpts.tex").write_text(ex_doc)

    print("Wrote:")
    print(" - results/metrics_summary.csv")
    print(" - results/main_results.tex")
    print(" - results/excerpts.tex")
# --- replace your success block with:
# --- helpers ---
def _is_numeric(rows):
    msgs = " ".join((r.get("msg","") or "").lower() for r in rows)
    return any(k in msgs for k in ["verify the criterion", "hypothesis:", "πr^2", "pi*r^2"])

def _num_truth_intent(rows):
    """Very simple intent detector for your two numeric demos."""
    msgs = " ".join((r.get("msg","") or "").lower() for r in rows)
    if "r=4" in msgs or " r = 4 " in msgs:
        return False
    if "r=3" in msgs or " r = 3 " in msgs:
        return True
    return None  # unknown → fall back to presence of satisfies=True/False

def scenario_metrics(rows):
    if not rows:
        return None

    # --- basic counts
    turns = len(rows)                         # <<< this was missing
    final_R = rows[-1].get("R")

    # --- deference (no premature answers in non-VERIFY turns)
    pre = [r for r in rows if (r.get("stance","").upper() != "VERIFY")]
    leaks = sum(int(r.get("has_answer_token", 0)) for r in pre)
    deference = 1.0 - (leaks / max(1, len(pre)))
    leakage_rate = leaks / max(1, len(pre))

    # --- tool discipline (tool calls inside VERIFY)
    tool_calls_outside = sum(int(r.get("tool_calls", 0)) for r in pre)
    total_tool_calls = sum(int(r.get("tool_calls", 0)) for r in rows)
    tool_calls_inside = total_tool_calls - tool_calls_outside
    tool_discipline = 1.0 if total_tool_calls == 0 else (tool_calls_inside / total_tool_calls)

    # --- success
    last_text = (rows[-1].get("text") or "").lower()
    if _is_numeric(rows):
        truth = _num_truth_intent(rows)
        sat_true  = "satisfies=true"  in last_text
        sat_false = "satisfies=false" in last_text
        if truth is True:
            success = sat_true
        elif truth is False:
            success = sat_false
        else:
            success = (sat_true or sat_false)
    else:
        success = bool(int(rows[-1].get("done", 0)) == 1)

    # --- quiz accuracy proxy (only for RAG runs; optional)
    quiz_acc = None
    for i, r in enumerate(rows):
        if (r.get("act","").upper() == "SUMMARIZE") and ("quick check" in (r.get("text","").lower())):
            # look at the learner's next turn
            if i + 1 < len(rows):
                ans = (rows[i+1].get("msg","") + rows[i+1].get("text","")).lower()
                expected = ["because", "risk", "failure", "limit", "mechanism", "source", "url"]
                quiz_acc = int(any(tok in ans for tok in expected))
            break

    return {
        "turns": turns,
        "final_R": final_R,
        "deference": round(deference, 3),
        "tool_discipline": round(tool_discipline, 3),
        "success": success,
        "turns_to_finalize": turns,
        "leakage_rate": round(leakage_rate, 3),
        "quiz_acc": quiz_acc,
    }

if __name__ == "__main__":
    main()

