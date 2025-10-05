from smolagents import Tool
import math
import re

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

