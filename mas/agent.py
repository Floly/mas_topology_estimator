import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from mas.prompts import SYSTEM_PROMPTS, parse_answer


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    model: str = "gpt-3.5-turbo"
    stub: bool = False  # if True, returns a deterministic fake answer without API calls
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: Optional[float] = None


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.total_tokens: int = 0
        if not config.stub:
            import openai  # imported lazily so stubs work without the package
            self._client = openai.OpenAI(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_env),
            )

    def run(self, task: str, incoming_messages: List[str]) -> str:
        if self.config.stub:
            return self._stub_response(task, incoming_messages)
        return self._llm_response(task, incoming_messages)

    # ------------------------------------------------------------------
    def _build_user_message(self, task: str, incoming_messages: List[str]) -> str:
        parts = [f"Problem:\n{task}"]
        if incoming_messages:
            parts.append("Previous agent outputs:\n" + "\n---\n".join(incoming_messages))
        return "\n\n".join(parts)

    def _llm_response(self, task: str, incoming_messages: List[str]) -> str:
        system = SYSTEM_PROMPTS[self.config.role]
        user = self._build_user_message(task, incoming_messages)
        if self.config.temperature is not None:
            temperature = self.config.temperature
        else:
            temperature = 1 if 'nano' in self.config.model else 0
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        if response.usage:
            self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
        return response.choices[0].message.content

    def _stub_response(self, task: str, incoming_messages: List[str]) -> str:
        """
        Returns a fake answer for testing pipeline correctness.
        Tries to echo the first ANSWER found in incoming messages,
        otherwise extracts the last number from the task itself.
        """
        for msg in reversed(incoming_messages):
            parsed = parse_answer(msg)
            if parsed is not None:
                return f"[stub:{self.config.agent_id}] Echoing prior answer. ANSWER: {parsed}"

        numbers = re.findall(r"-?\d+", task)
        fake = int(numbers[-1]) if numbers else 42
        return f"[stub:{self.config.agent_id}] No prior answer, guessing last number. ANSWER: {fake}"
