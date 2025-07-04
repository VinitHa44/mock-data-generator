from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateMockDataRequest(BaseModel):
    """
    Request model for the generate-mock-data endpoint.
    """

    examples: List[Dict[str, Any]] = Field(
        ...,
        description="An array of JSON objects to serve as examples for generation.",
    )


class APIResponse(BaseModel):
    """
    Standard API response model.
    """

    code: str = "OK"
    message: str = "Response received"
    usedFromCache: bool = False
    data: Any = None
    error: str = ""
    cacheInfo: Optional[dict] = None  # New field for cache details


class MockDataResponse(APIResponse):
    """
    Response model for successfully generated mock data.
    """

    data: List[Dict[str, Any]] = Field(
        ..., description="The array of generated mock data objects."
    )
