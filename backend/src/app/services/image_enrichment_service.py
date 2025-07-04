import httpx
import random
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
from pathlib import Path
from app.config.settings import settings
from app.utils.logging_config import get_logger
from app.utils.app_exceptions import ExternalAPIError

logger = get_logger(__name__)

class ImageEnrichmentService:
    def __init__(self):
        self.api_url = settings.OPENVERSE_API_URL
        self.used_urls = set()
        self.image_keys = [
            'image', 'img', 'picture', 'photo', 'pic', 'avatar', 'thumbnail', 'logo',
            'image_url', 'img_url', 'photo_url', 'avatar_url', 'logo_url', 'profile_pic',
            'gallery'
        ]
        self.image_url_patterns = ['images.unsplash.com', 'pexels.com', 'pixabay.com']
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']

    def _is_potential_image_url(self, key: str, value: Any) -> bool:
        """
        Determines if a value is likely an image URL based on its key,
        value content, or file extension.
        """
        if not isinstance(value, str) or not value.lower().startswith(('http://', 'https://')):
            return False

        key_lower = key.lower()
        if any(k in key_lower for k in self.image_keys):
            return True

        if any(pattern in value for pattern in self.image_url_patterns):
            return True
        
        # Check for image file extensions
        path = urlparse(value).path
        if any(path.lower().endswith(ext) for ext in self.image_extensions):
            return True

        return False

    def _extract_keywords_from_url(self, url: str) -> str:
        """
        Extracts descriptive keywords from a URL path by analyzing its components,
        preferring meaningful directory names over generic filenames.
        """
        try:
            p = Path(urlparse(url).path)
            parts = [part for part in p.parts if part != '/']
            
            if not parts:
                return "random"

            filename = parts[-1]
            file_stem = Path(filename).stem
            
            potential_keywords = parts[:-1]
            
            generic_dirs = {'images', 'assets', 'static', 'uploads', 'content', 'v1', 'v2', 'v3', 'media'}
            meaningful_dirs = [d for d in potential_keywords if d.lower() not in generic_dirs]
            
            if meaningful_dirs:
                keyword = meaningful_dirs[-1]
            elif not file_stem.isdigit() and file_stem:
                keyword = file_stem
            else:
                keyword = "photo" # Fallback for non-descriptive URLs
            
            return re.sub(r'[\-_]', ' ', keyword).strip()
        except Exception:
            return "random"

    async def _fetch_image_url_from_openverse(self, keywords: str) -> str:
        """Fetch a random image URL from Openverse for the given keywords."""
        query = keywords if keywords else "random"
        logger.info("Fetching image from Openverse API", keywords=query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}?format=json&q={query}")
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    logger.warning("No image results found for keywords", keywords=query)
                    return "https://via.placeholder.com/150"

                possible_urls = [res["url"] for res in results if "url" in res]
                random.shuffle(possible_urls)
                
                for url in possible_urls:
                    if url not in self.used_urls:
                        self.used_urls.add(url)
                        return url

                return random.choice(possible_urls) if possible_urls else "https://via.placeholder.com/150"

        except httpx.HTTPStatusError as e:
            logger.error("Openverse API request failed", status_code=e.response.status_code, keywords=query)
            return "https://via.placeholder.com/150" # Return placeholder on error
        except Exception as e:
            logger.error("An unexpected error occurred during Openverse API call", error=str(e))
            return "https://via.placeholder.com/150"

    async def _traverse_and_enrich(self, data: Any, parent_key: str = "") -> Any:
        """
        Recursively traverses the data structure, passing down the parent key
        to correctly identify and enrich image URLs, even within lists.
        """
        if isinstance(data, dict):
            return {key: await self._traverse_and_enrich(value, parent_key=key) for key, value in data.items()}
        elif isinstance(data, list):
            return [await self._traverse_and_enrich(item, parent_key=parent_key) for item in data]
        elif isinstance(data, str):
            if self._is_potential_image_url(parent_key, data):
                keywords = self._extract_keywords_from_url(data)
                return await self._fetch_image_url_from_openverse(keywords)
            return data
        else:
            return data

    async def enrich_mock_data(self, mock_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recursively traverses the mock data, finds fields that look like image URLs,
        and replaces them with real URLs from the Openverse API.
        """
        logger.info("Starting image enrichment process...")
        self.used_urls.clear() # Reset for each new batch
        enriched_data = await self._traverse_and_enrich(mock_data)
        logger.info("Image enrichment process completed.")
        return enriched_data

# Singleton instance
image_enrichment_service = ImageEnrichmentService() 