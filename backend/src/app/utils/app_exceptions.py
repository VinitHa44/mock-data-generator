class AppBaseException(Exception):
    """Base exception for this application."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class HarmfulContentError(AppBaseException):
    """Raised when input content is flagged as harmful."""


class LLMGenerationError(AppBaseException):
    """Raised when the LLM fails to generate data as expected."""


class CacheServiceError(AppBaseException):
    """Raised for issues related to the caching service."""


class InsufficientDataError(AppBaseException):
    """Raised when the LLM generates fewer items than requested."""

    def __init__(self, detail: str, generated_data: list, remaining_count: int):
        super().__init__(detail)
        self.generated_data = generated_data
        self.remaining_count = remaining_count


class ExternalAPIError(AppBaseException):
    """Raised when an external API call fails."""
