import re
from typing import Dict, Optional

SYSTEM_PROMPTS = {
    "solver": (
        "You are a math solver. "
        "Given a problem and any prior solutions from other agents, "
        "provide your best numerical answer. "
        "End your response with: ANSWER: <number>"
    ),
    "critic": (
        "You are a critic. Review the solutions provided and identify errors. "
        "Give a corrected answer. "
        "End your response with: ANSWER: <number>"
    ),
    "aggregator": (
        "You are an aggregator. Given multiple solutions, "
        "pick the most likely correct answer. "
        "End your response with: ANSWER: <number>"
    ),
    "manager": (
        "You are a manager. Read the problem carefully and outline "
        "a clear solution strategy for your team. "
        "End your response with: ANSWER: <number>"
    ),
    "debater": (
        "You are a debater. Solve the problem independently, "
        "without being influenced by other agents. "
        "End your response with: ANSWER: <number>"
    ),
    "judge": (
        "You are a judge. Given several solutions, evaluate each "
        "and pick the most likely correct answer. "
        "End your response with: ANSWER: <number>"
    ),
}


def assign_roles(n_agents: int) -> Dict[str, str]:
    """Assigns roles in order: solver, critic, aggregator (cycling)."""
    roles = ["solver", "critic", "aggregator"]
    return {f"A{i}": roles[i % len(roles)] for i in range(n_agents)}


def parse_answer(text: str) -> Optional[int]:
    m = re.search(r"ANSWER:\s*(-?\d+)", text)
    return int(m.group(1)) if m else None
