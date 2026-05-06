"""Application settings for FastAPI app.

Provides a Settings class and a default settings instance used by the app factory,
updated for Pydantic V2 compatibility.
"""
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field, model_validator
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

    data_dir: Path = Field(Path("/app/data/raw"))
    log_dir: Path = Field(Path("/app/data/logs"))

    # Database configuration
    db_host: str = Field("localhost")
    db_port: int = Field(5432)
    db_user: str = Field("hbr_user")
    db_password: str = Field("hbr_password")
    postgres_db: str = Field("hbr_db")
    database_url: Optional[str] = Field(None)

    @model_validator(mode="after")
    def set_database_url(self) -> "Settings":
        """Build database_url from individual fields if not explicitly set."""
        if self.database_url is None:
            self.database_url = (
                f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.postgres_db}"
            )
        return self


settings = Settings()