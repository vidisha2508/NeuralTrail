import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    DEVICE: str = "cpu"
    MODEL_TYPE: str = "resnet18"
    MODEL_CHECKPOINT_PATH: str = str(BASE_DIR / "models" / "resnet18.pth")
    NUM_CLASSES: int = 1000

    def get_checkpoint_path(self) -> str:
        p = Path(self.MODEL_CHECKPOINT_PATH)
        if not p.is_absolute():
            return str((BASE_DIR / p).resolve())
        return str(p.resolve())
    CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://neural-trail.vercel.app",
]

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "allow"

settings = Settings()
