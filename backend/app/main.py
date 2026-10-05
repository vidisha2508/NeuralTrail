import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.routes import router
from .services.model_loader import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("neural-trail-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start clean in standby state ready for user to select ResNet18, MobileNet, or Custom Model
    logger.info("Initializing Neural Trail ML Backend in STANDBY state...")
    yield
    # Shutdown
    logger.info("Shutting down Neural Trail ML Backend.")

app = FastAPI(
    title="Neural Trail ML Backend",
    description="Real PyTorch Vision Model Diagnostics & Inference Engine for Neural Trail",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    return {
        "system": "Neural Trail ML Backend",
        "version": "1.0.0",
        "docs_url": "/docs",
        "model_loaded": model_manager.is_loaded,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
