import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from mas.prompts import SYSTEM_PROMPTS, parse_answer


def _trunc(s: str, n: int = 300) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} chars]"


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    model: str = "gpt-3.5-turbo"
    stub: bool = False  # if True, returns a deterministic fake answer without API calls
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: Optional[float] = None
    debug: bool = False  # if True, print prompts/responses/retries/timing to stdout


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
            self._retryable_errors = (json.JSONDecodeError, openai.APIError)

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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self.config.debug:
            print(
                f"  [debug:{self.config.agent_id}] role={self.config.role} "
                f"model={self.config.model} temperature={temperature}"
            )
            print(f"  [debug:{self.config.agent_id}] user_msg: {_trunc(user)}")

        max_attempts = 4
        response = None
        last_error: Optional[Exception] = None
        t0 = time.perf_counter()
        for attempt in range(max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=60,
                )
                break
            except self._retryable_errors as exc:
                last_error = exc
                if self.config.debug:
                    print(f"  [debug:{self.config.agent_id}] attempt {attempt + 1}/{max_attempts} failed: {exc}")
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt
                    if self.config.debug:
                        print(f"  [debug:{self.config.agent_id}] retrying in {delay}s")
                    time.sleep(delay)
        if response is None:
            raise RuntimeError(
                f"LLM call failed after {max_attempts} attempts "
                f"(model={self.config.model}): {last_error}"
            ) from last_error

        if response.usage:
            self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
        content = response.choices[0].message.content or ""

        if self.config.debug:
            elapsed = time.perf_counter() - t0
            usage = response.usage
            tok_str = (
                f"prompt={usage.prompt_tokens} completion={usage.completion_tokens}"
                if usage else "usage=n/a"
            )
            print(f"  [debug:{self.config.agent_id}] {elapsed:.2f}s {tok_str}")
            print(f"  [debug:{self.config.agent_id}] response: {_trunc(content)}")

        return content

    def _stub_response(self, task: str, incoming_messages: List[str]) -> str:
        """
        Returns a fake answer for testing pipeline correctness.
        Tries to echo the first ANSWER found in incoming messages,
        otherwise extracts the last number from the task itself.
        """
        for msg in reversed(incoming_messages):
            parsed = parse_answer(msg)
            if parsed is not None:
                if self.config.debug:
                    print(f"  [debug:{self.config.agent_id}] stub: echoing prior answer {parsed}")
                return f"[stub:{self.config.agent_id}] Echoing prior answer. ANSWER: {parsed}"

        numbers = re.findall(r"-?\d+", task)
        fake = int(numbers[-1]) if numbers else 42
        if self.config.debug:
            print(f"  [debug:{self.config.agent_id}] stub: no prior answer, guessing last number {fake}")
        return f"[stub:{self.config.agent_id}] No prior answer, guessing last number. ANSWER: {fake}"
