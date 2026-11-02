"""Application settings, loaded from environment variables and a local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
# OpenRouter rejects a longer "models" fallback list with HTTP 400.
MAX_MODELS_IN_CHAIN = 3
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_models(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseSettings):
    # The project's .env works from any folder. A .env in the current folder takes priority.
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Free models, so the project runs with a key that has no credits. README lists a paid chain.
    llm_models: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free,thinkingmachines/inkling:free,"
        "nvidia/nemotron-3.5-lightning:free"
    )
    llm_escalation_models: str = (
        "nvidia/nemotron-3-ultra-550b-a55b:free,thinkingmachines/inkling:free"
    )
    # Free endpoints may log requests, so "deny" leaves few of them. Use "deny" with paid models.
    llm_data_collection: Literal["allow", "deny"] = "allow"
    llm_zdr: bool = False
    # "schema": route only to models with native JSON schema support. "prompt": describe the
    # format in the prompt, which opens up the models without it. Code validates both the same way.
    llm_output_mode: Literal["schema", "prompt"] = "prompt"
    llm_max_attempts: int = Field(default=3, ge=1, le=4)
    llm_max_tokens: int = Field(default=4000, ge=500, le=16000)
    # Reasoning models spend max_tokens on thinking first and may return no text at all.
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    llm_timeout_seconds: float = Field(default=90.0, gt=0, le=300)
    llm_http_retries: int = Field(default=2, ge=0, le=5)
    # Upper limit for one product with all repairs. The web UI waits for 300 seconds.
    generation_deadline_seconds: float = Field(default=280.0, gt=0)
    # Ceiling in USD per million tokens, enforced by OpenRouter at routing time.
    llm_max_price_prompt: float = Field(default=5.0, gt=0)
    llm_max_price_completion: float = Field(default=20.0, gt=0)

    rate_limit_per_minute: int = Field(default=20, ge=1)
    max_concurrent_generations: int = Field(default=4, ge=1, le=32)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    cache_max_entries: int = Field(default=128, ge=1)
    max_body_bytes: int = Field(default=64_000, ge=1_000)
    # Hostnames the web service answers to. Blocks DNS rebinding against the local API key.
    allowed_hosts: str = "localhost,127.0.0.1"

    @field_validator("openrouter_api_key")
    @classmethod
    def _empty_key_is_missing(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value().strip():
            return None
        return value

    @field_validator("openrouter_base_url")
    @classmethod
    def _base_url_must_be_safe(cls, value: str) -> str:
        """The API key is sent to this URL, so plain HTTP is allowed for loopback only."""
        parts = urlsplit(value)
        is_local = parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS
        if not (parts.scheme == "https" or is_local) or not parts.hostname:
            raise ValueError("OPENROUTER_BASE_URL must use https (http is allowed for localhost)")
        return value.rstrip("/")

    @field_validator("llm_models")
    @classmethod
    def _models_not_empty(cls, value: str) -> str:
        if not _split_models(value):
            raise ValueError("LLM_MODELS must list at least one model")
        return value

    @field_validator("llm_models", "llm_escalation_models")
    @classmethod
    def _chain_within_limit(cls, value: str) -> str:
        if len(_split_models(value)) > MAX_MODELS_IN_CHAIN:
            raise ValueError(f"A model chain can hold at most {MAX_MODELS_IN_CHAIN} models")
        return value

    @property
    def models(self) -> list[str]:
        return _split_models(self.llm_models)

    @property
    def escalation_models(self) -> list[str]:
        return _split_models(self.llm_escalation_models) or self.models

    @property
    def hosts(self) -> list[str]:
        return _split_models(self.allowed_hosts)


@lru_cache
def get_settings() -> Settings:
    return Settings()
