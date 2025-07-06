from app.controllers import performance_controller
from fastapi import APIRouter

router = APIRouter()

router.add_api_route(
    "/performance",
    performance_controller.get_performance_metrics,
    methods=["GET"],
    tags=["Performance"],
    summary="Get Performance Metrics",
) 