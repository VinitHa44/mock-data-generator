from typing import Any, Dict, List

from app.models.schemas.generation import APIResponse
from app.usecases.generate_mock_data import generate_mock_data_usecase
from app.utils.error_handler import api_error_handler
from app.utils.logging_config import get_logger
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
):
    """
    Generate mock data based on a list of JSON object examples.

    - **Request Body**: An array of JSON objects to serve as examples.
    - **count**: The number of mock data entries to generate.
    """
    logger.info("Received request for mock data generation", count=count)

    result_data, from_cache = await generate_mock_data_usecase.execute(
        input_examples=payload, count=count
    )

    return APIResponse(
        message="Response received successfully.",
        usedFromCache=from_cache,
        data=result_data,
    )
