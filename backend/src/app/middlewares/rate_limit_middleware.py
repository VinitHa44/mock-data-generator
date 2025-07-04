import time
from typing import Dict

from app.config.settings import settings
from app.utils.logging_config import get_logger
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = get_logger(__name__)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.requests: Dict[str, list] = {}
        self.rate_limit_requests = settings.RATE_LIMIT_REQUESTS
        self.rate_limit_timeframe = settings.RATE_LIMIT_TIMEFRAME

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get count parameter for dynamic rate limiting
        count = 1  # Default count
        if request.url.path.endswith("/generate-mock-data"):
            try:
                # Parse query parameters to get count
                query_params = dict(request.query_params)
                count = int(query_params.get("count", 1))
            except (ValueError, KeyError):
                count = 1
        
        # Dynamic rate limiting based on count
        # Higher counts get more lenient rate limits since they're more resource-intensive
        if count > 100:
            # For very large requests (100+ items), allow fewer requests
            effective_rate_limit = max(10, self.rate_limit_requests // (count // 50))
        elif count > 50:
            # For large requests (50-100 items), moderate rate limiting
            effective_rate_limit = max(20, self.rate_limit_requests // (count // 25))
        elif count > 25:
            # For medium requests (25-50 items), slight rate limiting
            effective_rate_limit = max(50, self.rate_limit_requests // 2)
        else:
            # For small requests, use standard rate limiting
            effective_rate_limit = self.rate_limit_requests
        
        current_time = time.time()
        
        # Initialize client tracking if not exists
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests outside the timeframe
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.rate_limit_timeframe
        ]
        
        # Check if rate limit exceeded
        if len(self.requests[client_ip]) >= effective_rate_limit:
            logger.warning(
                f"Rate limit exceeded for {client_ip}",
                count=count,
                effective_limit=effective_rate_limit,
                timeframe=self.rate_limit_timeframe
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {effective_rate_limit} requests per {self.rate_limit_timeframe} seconds for count={count}",
                    "retry_after": self.rate_limit_timeframe
                }
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(effective_rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(effective_rate_limit - len(self.requests[client_ip]))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.rate_limit_timeframe))
        
        return response
