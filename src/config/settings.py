"""
src/config/settings.py
-----------------------
Application settings for PathPilot AI.

All values can be overridden by environment variables or a ``.env`` file
in the project root.  ``pydantic-settings`` handles the binding
automatically — add a field here and it just works.

Usage::

    from src.config import settings

    if settings.vector_store_enabled:
        ...
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.enums import LLMProvider


class AppSettings(BaseSettings):
    """
    Central configuration object.

    Environment variable names map 1-to-1 with field names (case-insensitive).
    Example: ``LLM_PROVIDER=gemini`` sets ``settings.llm_provider``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # LLM configuration
    # ------------------------------------------------------------------

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description=(
            "Which LLM backend to use.  Valid values: openai | gemini | anthropic. "
            "Swap this without touching any node code."
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model name passed to the provider SDK.  "
            "E.g. 'gpt-4o', 'gemini-1.5-pro', 'claude-3-5-sonnet-20241022'."
        ),
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key.  Leave blank when using another provider.",
    )
    gemini_api_key: str = Field(
        default="",
        description="Google AI / Gemini API key.  Leave blank when using another provider.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key.  Leave blank when using another provider.",
    )

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    vector_store_enabled: bool = Field(
        default=False,
        description=(
            "Set to true to activate the Qdrant vector store. "
            "When false the NoopStore adapter is used transparently."
        ),
    )
    resume_upload_enabled: bool = Field(
        default=True,
        description=(
            "When true the UI offers a PDF resume upload widget.  "
            "Disable if you want profile-form-only input."
        ),
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose logging and Streamlit debug helpers.",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    database_path: Path = Field(
        default=Path("storage/pathpilot.db"),
        description="Filesystem path for the local SQLite database file.",
    )
    upload_dir: Path = Field(
        default=Path("storage/uploads"),
        description="Directory where uploaded resumes are temporarily stored.",
    )

    # ------------------------------------------------------------------
    # Qdrant (only read when vector_store_enabled=true)
    # ------------------------------------------------------------------

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL.  Ignored when vector_store_enabled=false.",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant Cloud API key.  Leave blank for local Docker.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("database_path", "upload_dir", mode="after")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        """Convert relative paths to absolute so they work from any cwd."""
        return v.resolve()

    # ------------------------------------------------------------------
    # Derived helpers (not env vars)
    # ------------------------------------------------------------------

    def active_api_key(self) -> str:
        """Return the correct API key for the configured provider (from .env only)."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key
        if self.llm_provider == LLMProvider.GEMINI:
            return self.gemini_api_key
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return ""

    def get_runtime_api_key(self) -> str:
        """
        Return the API key to use at runtime.

        Priority order:
        1. Runtime override (set via ``set_runtime_api_key()`` from the
           Streamlit UI).  This is session-only and never persisted.
        2. The .env / environment variable key (``active_api_key()``).

        This indirection lets the Streamlit app accept an API key via
        a password input without writing it to disk.
        """
        override = getattr(self, "_runtime_api_key_override", None)
        if override:
            return override
        return self.active_api_key()

    def set_runtime_api_key(self, key: str) -> None:
        """
        Inject an API key at runtime (session-only, never persisted).

        SECURITY: This key lives only in process memory.  It is NOT
        written to .env, SQLite, logs, or any file on disk.

        Also sets ``os.environ`` for the active provider so that any
        downstream library that reads the env var directly (e.g.
        LangChain internals) picks it up.
        """
        import os
        # Store on the instance (not a Pydantic field — won't be serialized)
        object.__setattr__(self, "_runtime_api_key_override", key)
        # Also set os.environ so downstream libs work
        if self.llm_provider == LLMProvider.OPENAI:
            os.environ["OPENAI_API_KEY"] = key
        elif self.llm_provider == LLMProvider.GEMINI:
            os.environ["GOOGLE_API_KEY"] = key
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            os.environ["ANTHROPIC_API_KEY"] = key

    def clear_runtime_api_key(self) -> None:
        """Remove the runtime API key override."""
        import os
        object.__setattr__(self, "_runtime_api_key_override", None)
        # Clean up env vars too
        if self.llm_provider == LLMProvider.OPENAI:
            os.environ.pop("OPENAI_API_KEY", None)
        elif self.llm_provider == LLMProvider.GEMINI:
            os.environ.pop("GOOGLE_API_KEY", None)
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def has_valid_api_key(self) -> bool:
        """True if a usable API key exists (runtime override or .env).

        Rejects common placeholder values that ship in .env.example so
        the app correctly shows the API key setup screen.
        """
        key = self.get_runtime_api_key()
        if not key:
            return False
        # Reject obvious placeholder values
        _placeholders = {
            "your-openai-api-key-here",
            "your-gemini-api-key-here",
            "your-anthropic-api-key-here",
            "test-key-not-real",
        }
        return key.lower() not in _placeholders


# Singleton instance — import this everywhere rather than re-instantiating.
settings = AppSettings()
