from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from smolagents import CodeAgent, LiteLLMModel, InferenceClientModel
from smolagents import ToolCallingAgent
from tools import CheckNumericClaim

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
        self.offline = offline
        if not offline:
            self.model = InferenceClientModel(
                model_id=model_backend,
                #. provider="hf-inference",
            )
        else:
            self.model = None

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

