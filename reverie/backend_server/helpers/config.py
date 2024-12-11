from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_name: str
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    embedding_model_name: str
    openai_api_key: str | None = None
