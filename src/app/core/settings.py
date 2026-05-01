"""Application settings for FastAPI app.

Provides a Settings class and a default settings instance used by the app factory,
updated for Pydantic V2 compatibility.
"""
from pathlib import Path

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    app_name: str = Field("HBR Backend")
    app_version: str = Field("0.1.0")
    debug: bool = Field(False)
    host: str = Field("0.0.0.0")
    port: int = Field(8000)

    data_dir: Path = Field(Path("./data/raw"))
    log_dir: Path = Field(Path("./data/logs"))

    # Database configuration
    db_host: str = Field("localhost")
    db_port: int = Field(5432)
    db_user: str = Field("hbr_user")
    db_password: str = Field("hbr_password")
    postgres_db: str = Field("hbr_db")
    database_url: str = Field("postgresql://hbr_user:hbr_password@localhost:5432/hbr_db")


settings = Settings()