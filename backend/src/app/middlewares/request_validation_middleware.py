import json
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp
from app.services.moderation_service import moderation_service
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class RequestValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # We only want to validate our specific data generation endpoint
        if "/generate-mock-data" not in request.url.path:
            return await call_next(request)

        # For POST methods, we read the body for validation
        if request.method == "POST":
            try:
                body = await request.json()
                # Store the body in the request state so we don't have to read the stream again
                request.state.json_body = body
            except json.JSONDecodeError:
                logger.warning("Request body is not valid JSON")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid JSON format in request body."},
                )

            # The body itself is the list of examples
            is_valid = await moderation_service.validate_input(body)

            if not is_valid:
                logger.warning("Harmful content detected in request", path=request.url.path)
                return JSONResponse(
                    status_code=403,  # Forbidden
                    content={
                        "code": "HarmfulContentError",
                        "message": "Input validation failed.",
                        "usedFromCache": False,
                        "data": None,
                        "error": "The provided input is considered harmful or inappropriate.",
                    },
                )

        response = await call_next(request)
        return response 