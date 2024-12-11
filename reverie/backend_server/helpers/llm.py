from __future__ import annotations

import llm
from openai import OpenAI
from pydantic import BaseModel

from .config import BackendConfig


def create_llm(config: BackendConfig) -> OpenRouterLLM | LocalLLM:
    return (
        OpenRouterLLM(
            config.llm_name,
            config.openrouter_api_key,
            config.openrouter_base_url,
        )
        if config.openrouter_api_key
        else LocalLLM(config.llm_name)
    )


class LLMParameters(BaseModel):
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: str | list[str] | None = None


class LocalLLM:
    def __init__(self, model_name: str) -> None:
        self.model = llm.get_model(model_name)

    def generate(self, message: str, params: LLMParameters | None = None) -> str:
        params = params or LLMParameters()

        response = self.model.prompt(
            message,
            max_tokens=params.max_tokens,
            temp=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.frequency_penalty,
        )

        return response.text()


class OpenRouterLLM:
    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        self.model_name = model_name
        self.model = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, message: str, params: LLMParameters | None = None) -> str:
        params = params or LLMParameters()

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
            **params.model_dump(),
        )

        return completion.choices[0].message.content
