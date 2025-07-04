from app.controllers import generation_controller
from app.models.schemas.generation import APIResponse
from app.utils.performance_monitor import performance_monitor
from fastapi import APIRouter

router = APIRouter()

router.add_api_route(
    "/generate-mock-data",
    generation_controller.generate_mock_data,
    methods=["POST"],
    response_model=APIResponse,
    tags=["Data Generation"],
    summary="Generate Mock Data",
)

router.add_api_route(
    "/performance",
    generation_controller.get_performance_metrics,
    methods=["GET"],
    tags=["Performance"],
    summary="Get Performance Metrics",
)
