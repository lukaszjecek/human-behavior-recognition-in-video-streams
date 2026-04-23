"""Application settings for FastAPI app.

Provides a Settings class and a default settings instance used by the app factory,
updated for Pydantic V2 compatibility.
"""
from pathlib import Path

# Dodajemy import ConfigDict dla Pydantic V2
from pydantic import Field, ConfigDict
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


settings = Settings()