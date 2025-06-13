from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp
import redis.asyncio as redis
from app.config.settings import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        # Allow health checks without rate limiting
        if request.url.path == "/health":
            return await call_next(request)

        # Use a pipeline to execute commands atomically
        pipe = self.redis_client.pipeline()
        pipe.incr(client_ip)
        pipe.expire(client_ip, settings.RATE_LIMIT_TIMEFRAME)
        
        try:
            request_count, _ = await pipe.execute()

            if int(request_count) > settings.RATE_LIMIT_REQUESTS:
                logger.warning("Rate limit exceeded", client_ip=client_ip)
                return JSONResponse(
                    status_code=429,  # Too Many Requests
                    content={
                        "code": "RateLimitExceeded",
                        "message": "Too many requests.",
                        "usedFromCache": False,
                        "data": None,
                        "error": f"You have exceeded the limit of {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_TIMEFRAME} seconds.",
                    },
                )
        except redis.exceptions.RedisError as e:
            logger.error("Could not connect to Redis for rate limiting", error=str(e))
            # Fail open: If Redis is down, we don't want to block all requests.
            return await call_next(request)

        return await call_next(request) 