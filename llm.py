"""
llm.py — Manzilix LLM module

Reusable wrapper around the NVIDIA NIM API (OpenAI-compatible) using the
nvidia/nemotron-3-ultra-550b-a55b model. Built as the general-purpose
assistant layer for Manzilix (chat, recommendations, summarization).

Usage (as a module):
    from llm import ManzilixLLM

    llm = ManzilixLLM()
    for token in llm.stream_chat("Suggest engineering colleges in Chennai for a CS student."):
        print(token, end="", flush=True)

Setup:
    1. pip install openai
    2. Set your API key as an environment variable (do NOT hardcode it):
         Windows (PowerShell):  setx NVIDIA_API_KEY "your-key-here"
         macOS/Linux:           export NVIDIA_API_KEY="your-key-here"
       Or create a .env file (see .env.example) and use python-dotenv.
"""

import os
from typing import Generator, List, Dict, Optional
from openai import OpenAI


DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_SYSTEM_PROMPT = (
    "You are the Manzilix assistant, a helpful AI guide for students and "
    "parents navigating education and career counselling. Be clear, "
    "supportive, and concise."
)


class ManzilixLLM:
    """Thin wrapper around the NVIDIA NIM chat completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key found. Set the NVIDIA_API_KEY environment variable "
                "or pass api_key= explicitly. Never hardcode keys in source files."
            )

        self.client = OpenAI(base_url=base_url, api_key=resolved_key)
        self.model = model
        self.system_prompt = system_prompt

    def _build_messages(
        self, user_message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def stream_chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        show_reasoning: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        reasoning_budget: int = 16384,
    ) -> Generator[str, None, None]:
        """
        Streams the model's response token-by-token.

        Args:
            user_message: the latest user prompt.
            history: optional list of prior {"role": ..., "content": ...} turns.
            show_reasoning: if True, also yields the model's internal reasoning
                tokens (useful for debugging, not recommended for end-user UI).
            temperature, top_p, max_tokens, reasoning_budget: generation controls.

        Yields:
            str chunks of the response as they arrive.
        """
        messages = self._build_messages(user_message, history)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": reasoning_budget,
            },
            stream=True,
        )

        for chunk in completion:
            if not chunk.choices:
                continue

            if show_reasoning:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    yield reasoning

            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Non-streaming convenience wrapper — collects the full response."""
        return "".join(
            self.stream_chat(user_message, history=history, **kwargs)
        )
