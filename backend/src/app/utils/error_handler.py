import inspect
from functools import wraps

from app.utils.app_exceptions import AppBaseException
from app.utils.logging_config import get_logger
from fastapi import Request
from fastapi.responses import JSONResponse

logger = get_logger(__name__)


def api_error_handler(func):
    """
    Decorator to catch exceptions from API endpoints and return a
    standardized JSON error response.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Inspect the decorated function's signature
        sig = inspect.signature(func)

        # Check if 'request' is a parameter in the decorated function
        if "request" in sig.parameters:
            # The request object is expected to be the first positional arg if present
            request = next(
                (arg for arg in args if isinstance(arg, Request)), None
            )
            if not request:
                # Or it might be in kwargs
                request = kwargs.get("request")
        else:
            request = None

        try:
            # Call the original function without 'request' if it's not in the signature
            # This is a bit of a simplification. A more robust solution might need to
            # selectively pass arguments.
            return await func(*args, **kwargs)

        except AppBaseException as e:
            logger.error(
                "Application error occurred", error=str(e), exc_info=True
            )
            return JSONResponse(
                status_code=400,  # Bad Request for most app-specific errors
                content={
                    "code": e.__class__.__name__,
                    "message": "An application error occurred.",
                    "usedFromCache": False,
                    "data": None,
                    "error": e.detail,
                },
            )
        except Exception as e:
            logger.critical(
                "An unexpected server error occurred",
                error=str(e),
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,  # Internal Server Error
                content={
                    "code": "InternalServerError",
                    "message": "An unexpected error occurred on the server.",
                    "usedFromCache": False,
                    "data": None,
                    "error": "An internal error has occurred.",
                },
            )

    return wrapper
