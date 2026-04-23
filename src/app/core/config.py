"""Centralized configuration loading using Pydantic."""

from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field("HBR Backend")
    app_version: str = Field("0.1.0")
    debug: bool = Field(False)
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    
    # Przykładowe ścieżki - pamiętaj o konwencji west-germany jeśli dotyczy
    data_dir: Path = Field(Path("/app/data/raw"))
    log_dir: Path = Field(Path("/app/data/logs"))

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()