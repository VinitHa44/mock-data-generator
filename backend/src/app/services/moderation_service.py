import json
import httpx
from app.config.settings import settings
from app.utils.logging_config import get_logger
from threading import Lock

logger = get_logger(__name__)

class ModerationService:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    logger.info("Initializing ModerationService...")
                    self.client = httpx.AsyncClient(base_url=settings.MODEL_SERVER_URL, timeout=30.0)
                    self.initialized = True
                    logger.info("ModerationService initialized successfully.")

    async def _is_harmful_content(self, text: str) -> bool:
        """Check text with the text moderation model."""
        try:
            response = await self.client.post("/validate_harmful", json={"text": text})
            response.raise_for_status()
            result = response.json()
            if result.get("is_harmful"):
                logger.warning(
                    "Harmful content detected by model server",
                    content=text,
                    details=result.get("details")
                )
                return True
        except httpx.RequestError:
            logger.error("Error calling model server for harmful content check.", exc_info=True)
            return True # Fail safe, assume harmful
        except Exception:
            logger.error("An unexpected error occurred during harmful content check.", exc_info=True)
            return True # Fail safe
        return False

    async def _is_prompt_injection(self, text: str) -> bool:
        """Check text for prompt injection."""
        try:
            response = await self.client.post("/validate_injection", json={"text": text})
            response.raise_for_status()
            result = response.json()
            if result.get("is_injection"):
                logger.warning(
                    "Prompt injection attempt detected by model server",
                    content=text,
                    details=result.get("details")
                )
                return True
            return False  # Return false if no injection is detected
        except httpx.RequestError:
            logger.error("Error calling model server for prompt injection check.", exc_info=True)
            return True # Fail safe, assume injection
        except Exception:
            logger.error("An unexpected error occurred during prompt injection check.", exc_info=True)
            return True # Fail safe
        return False

    async def validate_input(self, payload: list) -> bool:
        """
        Validate the entire input payload for harmful content or prompt injection.

        Args:
            payload: A list of JSON objects (dicts).

        Returns:
            bool: True if the content is valid, False otherwise.
        """

        def extract_string_values(data):
            string_values = []
            if isinstance(data, dict):
                for value in data.values():
                    string_values.extend(extract_string_values(value))
            elif isinstance(data, list):
                for item in data:
                    string_values.extend(extract_string_values(item))
            elif isinstance(data, str):
                string_values.append(data)
            return string_values

        try:
            # Extract and join all string values from the payload for validation
            all_strings = extract_string_values(payload)
            text_to_validate = " ".join(all_strings)

            if not text_to_validate.strip():
                # If there's no text content, no need to validate
                return True

        except Exception:
            logger.error("Failed to extract string values from payload for validation.", exc_info=True)
            return False # Or raise an exception

        if await self._is_harmful_content(text_to_validate):
            return False
        
        if await self._is_prompt_injection(text_to_validate):
            return False

        return True

    async def close(self):
        await self.client.aclose()

# Singleton instance
moderation_service = ModerationService() 