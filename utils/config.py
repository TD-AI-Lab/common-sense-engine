"""
Configuration
-------------
Use pydantic settings to load env vars (.env supported).
Keep all service endpoints and runtime flags here.
"""

from __future__ import annotations

import re
from typing import List

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------
    # ElasticSearch
    # ---------------------------
    ES_ENDPOINT: str = Field(default="http://localhost:9200", description="Elasticsearch endpoint (with scheme)")
    ES_INDEX: str = Field(default="facts_index", description="Default ES index name")
    ES_EMBEDDING_DIM: int | None = Field(default=None, description="If set, create 'embedding' dense_vector with this dimension")
    ES_USER: str | None = Field(default=None, description="Elasticsearch username (optional)")
    ES_PASSWORD: str | None = Field(default=None, description="Elasticsearch password (optional)")
    ES_API_KEY: str | None = None
    CAUSAL_ES_INDEX: str = Field(default="causal_relations", description="Elasticsearch index for causal relations (optional)")
    ENABLE_CAUSAL_INDEX: bool = Field(default=False, description="If True, index es_records produced by CausalityMapper")

    # ---------------------------
    # API
    # ---------------------------
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8080)
    # Comma-separated env var is supported by pydantic-settings for lists
    API_CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")

    # ---------------------------
    # Google Programmable Search (Custom Search)
    # ---------------------------
    GOOGLE_API_KEY: str | None = Field(default=None, description="Google API key for Custom Search")
    GOOGLE_CX: str | None = Field(default=None, description="Programmable Search Engine ID (cx)")

    # ---------------------------
    # Gemini
    # ---------------------------
    GEMINI_API_KEY: str | None = Field(default=None, description="Gemini API key (google-generativeai)")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Default Gemini model for summarization/synthesis")
    
    # ---------------------------
    # Vertex AI (placeholders)
    # ---------------------------
    VERTEX_PROJECT: str | None = None
    VERTEX_LOCATION: str | None = None

    # ---------------------------
    # Runtime
    # ---------------------------
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    # pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------
    # Validators / Normalizers
    # ---------------------------

    @field_validator("ES_ENDPOINT")
    @classmethod
    def _normalize_es_endpoint(cls, v: str) -> str:
        """Ensure ES endpoint has an http(s) scheme and no trailing slash."""
        v = (v or "").strip().rstrip("/")
        if v and not re.match(r"^https?://", v, flags=re.I):
            v = f"http://{v}"
        return v

    @field_validator("API_CORS_ORIGINS")
    @classmethod
    def _normalize_cors(cls, v: List[str]) -> List[str]:
        """Trim spaces and drop empties."""
        cleaned = [o.strip() for o in (v or []) if o and o.strip()]
        return cleaned or ["*"]

    # ---------------------------
    # Computed flags
    # ---------------------------

    @computed_field  # type: ignore[misc]
    @property
    def ES_AUTH_ENABLED(self) -> bool:
        """True if ES credentials are set."""
        return bool(self.ES_API_KEY or (self.ES_USER and self.ES_PASSWORD))

    @computed_field  # type: ignore[misc]
    @property
    def GEMINI_ENABLED(self) -> bool:
        """True if a non-empty Gemini key is present."""
        key = (self.GEMINI_API_KEY or "").strip().strip('"').strip("'")
        return bool(key)

    @computed_field  # type: ignore[misc]
    @property
    def GOOGLE_ENABLED(self) -> bool:
        """True if Google Custom Search is configured (used to trigger live ingest by défaut)."""
        return bool((self.GOOGLE_API_KEY or "").strip() and (self.GOOGLE_CX or "").strip())

    # ---------------------------
    # Utilities
    # ---------------------------

    def redacted(self) -> dict:
        """Return a dict safe to log (secrets masked)."""
        data = self.model_dump()
        for k in ("GEMINI_API_KEY", "ES_PASSWORD", "ES_API_KEY", "GOOGLE_API_KEY"):
            if data.get(k):
                data[k] = "********"
        return data