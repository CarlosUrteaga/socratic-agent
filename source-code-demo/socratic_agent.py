from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from smolagents import CodeAgent, LiteLLMModel, InferenceClientModel
from smolagents import ToolCallingAgent
from tools import CheckNumericClaim
from tools import HttpRAGTool 

import re

SPEECH_ACTS = ["ASK", "CLARIFY", "PROBE", "CHALLENGE", "SUMMARIZE", "VERIFY"]
@dataclass

# Learner Reasoning node.
class LRTNode:
    #  "claim" | "step" | "evidence" | "counterexample"
    kind: str
    text: str

# Learner Reasoning Trace.
@dataclass
class LRT:
    nodes: List[LRTNode] = field(default_factory=list)
    edges: List[Tuple[int,int,str]] = field(default_factory=list)

@dataclass
class Ledger:
    goal: str = ""
    assumptions: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    criteria: List[str] = field(default_factory=list)
    confidence: float = 0.0
    open_questions: List[str] = field(default_factory=list)

def _extract_hypothesis(msg: str) -> str:
    import re
    m = re.search(r"area of a circle with r\s*=\s*\d+(?:\.\d+)?\s*is\s*\d+(?:\.\d+)?", msg.lower())
    return m.group(0) if m else ""

def _extract_sources_from_ctx(ctx: str) -> list[str]:
    urls = re.findall(r'(https?://[^\s\)\]]+)', ctx)
    # Keep order & unique
    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); ordered.append(u)
    return ordered

def _parse_rag(msg: str):
    """
    Soporta:
      RAG: pregunta...
      RAG[https://u1,https://u2]: pregunta...
    """
    m = re.match(r"^\s*rag\s*:(.*)$", msg, flags=re.I)
    if m:
        return [], m.group(1).strip()
    m = re.match(r"^\s*rag\s*\[([^\]]*)\]\s*:\s*(.*)$", msg, flags=re.I)
    if m:
        urls = [u.strip() for u in m.group(1).split(",") if u.strip()]
        q = m.group(2).strip()
        return urls, q
    return None, None



class SocraticController:
    def __init__(self, model_backend="Qwen/Qwen2.5-7B-Instruct", tau=0.7, offline=True):
        self.last_hypothesis = ""
        self.did_summarize = False
        self.tau = tau
        self.s = "EXPLORE"
        self.R = 0.0
        self.lrt = LRT()
        self.ledger = Ledger()
        self.tools = [CheckNumericClaim()]
        self.tools.append(HttpRAGTool(base_url="http://localhost:8000"))
        self.offline = offline
        self.rag_flow: Optional[Dict[str, Any]] = None

        if not offline:
            self.model = InferenceClientModel(
                model_id=model_backend,
                #. provider="hf-inference",
            )
        else:
            self.model = None

    def _get_tool(self, name: str):
        for t in self.tools:
            if getattr(t, "name", "") == name:
                return t
        return None

    def readiness(self) -> float:
        coverage = 0.0
        coverage += 0.30 if self.ledger.goal else 0.0
        coverage += min(0.40, 0.20 + 0.05 * max(0, len(self.ledger.criteria)-1)) if self.ledger.criteria else 0.0
        evidence_nodes = sum(1 for n in self.lrt.nodes if n.kind == "evidence")
        coverage += min(0.20, 0.10 + 0.05 * max(0, evidence_nodes-1)) if evidence_nodes else 0.0
        counter_nodes = sum(1 for n in self.lrt.nodes if n.kind == "counterexample")
        coverage += min(0.10, 0.05 * counter_nodes)
        return round(min(1.0, coverage), 2)


    # replace current choose_act with:
    def choose_act(self, force_verify: bool = False) -> str:
        self.R = self.readiness()
        have_goal = bool(self.ledger.goal)
        have_criteria = bool(self.ledger.criteria)

        if not have_goal:
            self.s = "EXPLORE"
            return "ASK"

        if have_goal and not have_criteria:
            self.s = "EXPLORE"
            return "PROBE"

        # we have criteria now
        if self.s != "VERIFY":
            if not self.did_summarize and not force_verify:
                self.did_summarize = True
                self.s = "EXPLORE"
                return "SUMMARIZE"
            self.s = "VERIFY"

        # in VERIFY: if forced, verify now; otherwise gate on R
        if force_verify:
            return "VERIFY"
        if self.R < self.tau:
            return "CHALLENGE"
        return "VERIFY"

    def _offline_generate(self, act: str, learner_msg: str) -> str:
        if act == "ASK":
            return "What is your exact goal and what assumptions are you making?"
        if act == "CLARIFY":
            return "When you say that, do you mean the rule holds for all r or a specific case?"
        if act == "PROBE":
            return "State one concrete criterion we can test (e.g., a numeric check you’d accept)."
        if act == "CHALLENGE":
            return "Consider r=10: would your rule still match πr²? If not, how adjust?"
        if act == "SUMMARIZE":
            g = f"Goal: {self.ledger.goal or '(not stated)'}"
            c = f"Criteria: {self.ledger.criteria or '[]'}"
            oq = f"Open questions: {self.ledger.open_questions or '[]'}"
            return f"Here’s your current state.\n{g}\n{c}\n{oq}"
        if act == "VERIFY":
            return "Running the minimal check..."
        return "Let’s continue."


    def step(self, learner_msg: str) -> Dict[str, Any]:
        # ------ state updates (same as before) ------
        msg_low = learner_msg.lower()
        low = msg_low.lower()

        # --- RAG convention: handle early and return ---
        urls, topic = _parse_rag(learner_msg)
        if topic is not None:
            # Initialize a multi-turn RAG tutoring flow
            self.rag_flow = {"phase": "ELICIT", "topic": topic, "urls": urls or [], "ctx": None}
            return {
                "act": "ASK",
                "stance": "EXPLORE",
                "R": self.R,
                "text": (
                    "Before I look things up: what do you already know about this topic, "
                    "and what do you most want to understand "
                    #"‘how to wire it in code’, ‘tradeoffs vs. fine-tuning’)?"
                ),
                "done": False,
            }

        # ------ RAG flow: continue across turns ------
        if self.rag_flow:
            phase = self.rag_flow["phase"]
            topic = self.rag_flow["topic"]
            urls  = self.rag_flow["urls"]

            if phase == "ELICIT":
                # We got the learner's prior. Now retrieve.
                rag = self._get_tool("web_rag")
                # If no URLs provided, give sensible defaults
                if not urls:
                    urls = [
                        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                        "https://fastapi.tiangolo.com/",
                    ]
                    self.rag_flow["urls"] = urls
                ctx = rag.forward(question=topic, urls=urls, top_k=4) if rag else "RAG tool not available."
                self.rag_flow["ctx"] = ctx
                self.rag_flow["phase"] = "SYNTH"

                sources = _extract_sources_from_ctx(ctx)
                sources_block = "" if not sources else "\n\nSources:\n" + "\n".join(f"- {u}" for u in sources)
                scaffold = (
                    "Use this structure:\n"
                    "1) Claim (1–2 lines)\n"
                    "2) Evidence (quote or paraphrase from the snippets + URL)\n"
                    "3) Reasoning (why the evidence supports the claim)\n"
                    "4) Limits/Risks (when RAG might still fail)\n"
                )
                return {
                    "act": "VERIFY",
                    "stance": "VERIFY",
                    "R": max(self.R, 0.6),
                    "text": (
                        "Here are relevant snippets. Read them, then draft your explanation.\n\n"
                        f"{ctx}{sources_block}\n\n"
                        "Start with “My current understanding is…”, then follow the scaffold below.\n\n" + scaffold
                    ),
                    "done": False,
                }

            elif phase == "SYNTH":
                # Provide coach feedback based on learner draft + the retrieved context
                ctx = self.rag_flow.get("ctx") or ""
                system = (
                    "You are a Socratic tutor. Give concise feedback on the student's explanation. "
                    "Do NOT overwrite their answer; instead, evaluate (correctness, evidence use, clarity) "
                    "and suggest one concrete improvement. Keep to 4–6 lines."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context snippets:\n{ctx}\n\nStudent draft:\n{learner_msg}"},
                ]
                try:
                    resp = self.model.generate(messages, max_tokens=180, temperature=0.2)
                    feedback = (resp.content or "").strip()
                except Exception as e:
                    feedback = f"[feedback error: {e}]"

                # Move to a short quiz/check
                self.rag_flow["phase"] = "QUIZ"
                quiz = (
                    "Quick check (1–2 lines each):\n"
                    f"• Summarize the main mechanism discussed.\n"
                     "• Cite one concrete fact from the snippets (include the URL).\n"
                     "• Name one limitation or risk mentioned."
                )
                return {
                    "act": "SUMMARIZE",
                    "stance": "EXPLORE",
                    "R": max(self.R, 0.65),
                    "text": feedback + "\n\n" + quiz,
                    "done": False,
                }

            elif phase == "QUIZ":
                # Close the loop: brief verification + finalize
                self.rag_flow = None
                return {
                    "act": "VERIFY",
                    "stance": "VERIFY",
                    "R": max(self.R, 0.7),
                    "text": (
                        "Thanks. Based on your answer, you’ve identified the key mechanism "
                    ),
                    "done": True,
                }

        hyp = _extract_hypothesis(learner_msg)
        if hyp:
            self.last_hypothesis = hyp

        if ("my goal" in msg_low) or ("goal:" in msg_low) or ("goal is" in msg_low):
            self.ledger.goal = learner_msg
        if ("criterion" in msg_low) or ("criteria" in msg_low) or ("verify the criterion" in msg_low):
            self.ledger.criteria.append(learner_msg)

        force_verify = ("verify" in msg_low) or bool(hyp)
        act = self.choose_act(force_verify=force_verify)

        # ------ generation (always initialize text!) ------
        text: str = ""  # prevent UnboundLocalError

        try:
            if self.offline:
                text = self._offline_generate(act, learner_msg)
            else:
                if act == "VERIFY":
                    # controller-driven tool check (no agent, no tool logs)
                    hyp_to_check = hyp or self.last_hypothesis or learner_msg
                    finding = self.tools[0].forward(hypothesis=hyp_to_check)
                    text = f"Verification finding: {finding}"
                    self.lrt.nodes.append(LRTNode("evidence", text))
                    R_after = self.readiness()
                    done = (self.s == "VERIFY" and R_after >= self.tau and "satisfies=True" in text)
                    return {"act": act, "R": self.R, "stance": self.s, "text": text, "done": done}

                # --- dialogue turns: plain generate (no agents, no tools) ---
                system = (
                    "You are a Socratic tutor. Before readiness, DO NOT provide final answers, "
                    "numeric conclusions, or verdicts. Ask one focused question or reflect the learner's reasoning."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"[ACT={act}] User: {learner_msg}\nTutor:"},
                ]
                try:
                    resp = self.model.generate(messages, max_tokens=128, temperature=0.2)  # <-- here
                    text = (resp.content or "").strip()
                except Exception as e:
                    text = f"[error during {act}: {e}]"
        except Exception as e:
            # Don't crash the demo; surface error text and continue
            text = f"[error during {act}: {e}]"

        # ------ verify path: call curated tool and log evidence ------
        done = False
        if act == "VERIFY":
            hyp_to_check = hyp or self.last_hypothesis or learner_msg
            finding = self.tools[0].forward(hypothesis=hyp_to_check)
            text = f"Verification finding: {finding}"
            self.lrt.nodes.append(LRTNode("evidence", text))

            # Recompute readiness after adding evidence so we can finalize in this turn
            R_after = self.readiness()
            done = (self.s == "VERIFY" and R_after >= self.tau and "satisfies=True" in text)

        # ------ deference filter for non-VERIFY turns (belt & suspenders) ------
        if act != "VERIFY":
            lower = text.lower()
            if any(p in lower for p in ["final answer", "therefore the answer", "the area is", "≈", "approximately"]):
                text = "Let's not finalize yet. What criterion would convince you your idea works?"

        # ------ ALWAYS return a dict ------
        return {"act": act, "R": self.R, "stance": self.s, "text": text, "done": False}

