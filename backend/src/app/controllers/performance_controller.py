from app.config.settings import settings
from app.usecases.generate_mock_data import generate_mock_data_usecase
from app.utils.error_handler import api_error_handler
from app.utils.performance_monitor import performance_monitor


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