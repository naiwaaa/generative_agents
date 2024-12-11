from __future__ import annotations

import llm
from openai import OpenAI

from .config import BackendConfig


def create_embedding_model(
    config: BackendConfig,
) -> LocalEmbeddingModel | OpenAIEmbeddingModel:
    return (
        OpenAIEmbeddingModel(config.embedding_model_name, config.openai_api_key)
        if config.openai_api_key
        else LocalEmbeddingModel(config.embedding_model_name)
    )


class LocalEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model = llm.get_embedding_model(model_name)

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        if not text:
            text = "this is blank"

        return self.model.embed(text)


class OpenAIEmbeddingModel:
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.client = OpenAI()

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        if not text:
            text = "this is blank"

        return (
            self.client.embeddings.create(input=text, model=self.model_name)
            .data[0]
            .embedding
        )
