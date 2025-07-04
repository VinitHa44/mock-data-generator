import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add the 'src' directory to the Python path
# This allows for absolute imports from 'app' as if 'src' was the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config.settings import settings
from app.middlewares.rate_limit_middleware import RateLimitingMiddleware
from app.middlewares.request_validation_middleware import (
    RequestValidationMiddleware,
)
from app.routers import generation_router
from app.services.cache_service import cache_service
from app.services.moderation_service import moderation_service
from app.services.llm_service import llm_service
from app.utils.logging_config import setup_logging, get_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        await cache_service.initialize()
        # Initialize LLM service pool
        await llm_service.initialize()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown logic
    try:
        await cache_service.shutdown()
        await moderation_service.close()
        await llm_service.shutdown()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,  # React development server
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add other middlewares
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(RequestValidationMiddleware)


# Include routers
app.include_router(generation_router.router, prefix=settings.API_PREFIX)


@app.get(f"{settings.API_PREFIX}/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "Welcome to the Mock Data Generator API"}


@app.get(f"{settings.API_PREFIX}/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}
