"""Application settings."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    DEBUG: bool = os.environ.get("DEBUG", "True").lower() in ("1", "true", "yes")
    SECRET_KEY: str = os.environ.get(
        "SECRET_KEY", "change-me-in-production-stock-predictive"
    )
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080",
        ).split(",")
        if o.strip()
    ]
    MODELS_STORE_DIR: str = os.environ.get(
        "MODELS_STORE_DIR", str(BASE_DIR / "models_store")
    )
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "Data")))

    POSTGRES_HOST: str | None = os.environ.get("POSTGRES_HOST")
    POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "stock_predictive")
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "stock")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "stockpass")

    @property
    def database_url(self) -> str:
        if self.POSTGRES_HOST:
            return (
                f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return f"sqlite:///{BASE_DIR / 'db.sqlite3'}"


settings = Settings()
