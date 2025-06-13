import httpx
import random
from typing import List, Dict, Any, Set
from urllib.parse import urlparse
from app.config.settings import settings
from app.utils.logging_config import get_logger
from app.utils.app_exceptions import ExternalAPIError

logger = get_logger(__name__)

class ImageEnrichmentService:
    def __init__(self):
        self.api_url = settings.OPENVERSE_API_URL
        self.used_urls = set()

    def _is_url(self, text: str) -> bool:
        """Check if a string is a valid URL."""
        if not isinstance(text, str):
            return False
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _is_keyword_string(self, text: str) -> bool:
        """Check if a string is a '+' delimited keyword string."""
        return isinstance(text, str) and '+' in text and not self._is_url(text)

    async def _fetch_image_url_from_openverse(self, keywords: str) -> str:
        """Fetch a random image URL from Openverse for the given keywords."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}?format=json&q={keywords}")
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    logger.warning("No image results found for keywords", keywords=keywords)
                    return "https://via.placeholder.com/150" # Return a placeholder

                # Try to find a new URL
                possible_urls = [res["url"] for res in results[:20]]
                random.shuffle(possible_urls)
                
                for url in possible_urls:
                    if url not in self.used_urls:
                        self.used_urls.add(url)
                        return url

                # If all are used, just return one of them
                return random.choice(possible_urls) if possible_urls else "https://via.placeholder.com/150"

        except httpx.HTTPStatusError as e:
            logger.error("Openverse API request failed", status_code=e.response.status_code, keywords=keywords)
            raise ExternalAPIError("Failed to fetch image from Openverse API.")
        except Exception as e:
            logger.error("An unexpected error occurred during Openverse API call", error=str(e))
            raise ExternalAPIError("An unexpected error occurred while fetching image data.")

    def find_image_url_keys(self, input_example: Dict[str, Any], output_example: Dict[str, Any]) -> Set[str]:
        """
        Identify keys that should be image URLs by comparing an input object
        with a generated output object.
        A key is considered an image URL key if its value in the input is a URL
        and its value in the output is a keyword string.
        """
        image_keys = set()
        
        # Check top-level keys
        for key in input_example.keys():
            if key in output_example:
                input_value = input_example[key]
                output_value = output_example[key]
                
                if self._is_url(input_value) and self._is_keyword_string(output_value):
                    image_keys.add(key)
        
        # Also check nested dictionaries
        for key, value in input_example.items():
            if isinstance(value, dict) and key in output_example and isinstance(output_example[key], dict):
                nested_keys = self.find_image_url_keys(value, output_example[key])
                for nested_key in nested_keys:
                    # Store nested keys in a format like "parent.child"
                    image_keys.add(f"{key}.{nested_key}")

        if image_keys:
            logger.info("Identified image URL keys for enrichment", keys=list(image_keys))
        return image_keys

    async def enrich_mock_data(self, mock_data: List[Dict[str, Any]], image_keys: Set[str]) -> List[Dict[str, Any]]:
        """
        Iterate through mock data and replace keyword strings with real image URLs.
        """
        if not image_keys:
            return mock_data

        enriched_data = []
        for item in mock_data:
            enriched_item = item.copy()
            for key in image_keys:
                if key in enriched_item and self._is_keyword_string(enriched_item[key]):
                    keywords = enriched_item[key]
                    image_url = await self._fetch_image_url_from_openverse(keywords)
                    enriched_item[key] = image_url
            enriched_data.append(enriched_item)
        
        return enriched_data

# Singleton instance
image_enrichment_service = ImageEnrichmentService() 