import time
from typing import Any, Dict, List

from app.config.settings import settings
from app.models.schemas.generation import APIResponse
from app.usecases.generate_mock_data import generate_mock_data_usecase
from app.utils.error_handler import api_error_handler
from app.utils.logging_config import get_logger
from app.utils.performance_monitor import performance_monitor
from fastapi import Query

logger = get_logger(__name__)


@api_error_handler
async def generate_mock_data(
    payload: List[Dict[str, Any]],
    count: int = Query(
        10,
        ge=1,
        le=500,
        description="The desired number of mock data entries to generate.",
    ),
    enable_moderation: bool = Query(
        True,
        description="Enable content moderation for generated data.",
    ),
    temperature: float = Query(
        0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation randomness (0.0-2.0).",
    ),
    max_tokens: int = Query(
        2048,
        ge=1,
        le=8192,
        description="Maximum tokens for generation.",
    ),
    top_p: float = Query(
        0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter (0.0-1.0).",
    ),
    cache_expiration: bool = Query(
        False,
        description="Enable cache expiration for generated data.",
    ),
):
    """
    Generate mock data based on a list of JSON object examples.

    - **Request Body**: An array of JSON objects to serve as examples.
    - **count**: The number of mock data entries to generate.
    - **enable_moderation**: Enable content moderation.
    - **temperature**: Generation temperature (0.0-2.0).
    - **max_tokens**: Maximum tokens for generation.
    - **top_p**: Top-p sampling parameter (0.0-1.0).
    - **cache_expiration**: Enable cache expiration.
    """
    start_time = time.time()
    logger.info("Received request for mock data generation", count=count)

    result_data, from_cache, cache_info = await generate_mock_data_usecase.execute(
        input_examples=payload, 
        count=count,
        enable_moderation=enable_moderation,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        cache_expiration=cache_expiration
    )

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\033[92m⏱️  JSON Generation completed in {total_time:.2f} seconds for {count} items\033[0m")
    logger.info(f"JSON Generation completed in {total_time:.2f} seconds for {count} items")

    return APIResponse(
        message="Response received successfully.",
        usedFromCache=from_cache,
        data=result_data,
        cacheInfo=cache_info,
    )


@api_error_handler
async def get_performance_metrics():
    """
    Get performance metrics for the concurrent processing system.
    """
    summary = performance_monitor.get_performance_summary()
    
    return {
        "message": "Performance metrics retrieved successfully.",
        "metrics": summary,
        "concurrent_config": {
            "llm_pool_size": getattr(generate_mock_data_usecase.llm_service, '_pool_size', settings.LLM_POOL_SIZE),
            "batch_size": getattr(generate_mock_data_usecase.llm_service, '_batch_size', settings.LLM_BATCH_SIZE),
            "batch_threshold": settings.LLM_BATCH_THRESHOLD,
            "model_instances": settings.LLM_POOL_SIZE,
            "target_batches_for_500": 500 // settings.LLM_BATCH_SIZE,
            "max_concurrent_batches": settings.LLM_POOL_SIZE
        },
        "cache_config": {
            "enable_expiration": settings.CACHE_ENABLE_EXPIRATION,
            "expiration_seconds": settings.CACHE_EXPIRATION_SECONDS if settings.CACHE_ENABLE_EXPIRATION else None,
            "hash_limit": settings.CACHE_GROUP_HASH_LIMIT
        }
    }
