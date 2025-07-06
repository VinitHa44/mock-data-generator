import asyncio
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
import re

import httpx
from app.config.settings import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageEnrichmentService:
    def __init__(self):
        self.api_url = settings.OPENVERSE_API_URL
        self.used_urls = set()




    def _is_keyword_string(self, key: str, value: Any) -> bool:
        """
        Determines if a value is a keyword string.
        This now includes simple strings, img_ formats, and URLs.
        """
        if not isinstance(value, str):
            return False

        # Any non-empty string is a potential keyword string
        return bool(value.strip())

    def _extract_keywords_from_string(self, keyword_string: str) -> List[str]:
        """
        Extracts keywords from various string formats:
        - "img_tv+smart+led" -> ["tv", "smart", "led"]
        - "img_tv_smart_led" -> ["tv", "smart", "led"]
        - "A nice car" -> ["A", "nice", "car"]
        - "singlekeyword" -> ["singlekeyword"]
        Returns a list of keywords.
        """
        try:
            if keyword_string.startswith("img_"):
                # Handle "img_" prefixed strings
                keywords_part = keyword_string[4:]
                # Split by either '+' or '_'
                keywords = [kw.strip() for kw in re.split('[+_]', keywords_part) if kw.strip()]
                return keywords
            elif keyword_string.lower().startswith(("http://", "https://")):
                 # If it's a URL, return empty list as there are no keywords to search
                return []
            else:
                # Handle simple strings: treat spaces as delimiters
                return [kw.strip() for kw in keyword_string.split() if kw.strip()]
        except Exception as e:
            logger.error(f"Error extracting keywords from string: {keyword_string}", error=str(e))
            return []

    def _extract_keywords_from_url(self, url: str) -> str:
        """
        Extracts descriptive keywords from a URL path by analyzing its components.
        """
        try:
            from urllib.parse import urlparse
            from pathlib import Path
            
            # Parse the URL
            parsed = urlparse(url)
            path = parsed.path
            
            # Split path into components
            path_parts = [part for part in path.split('/') if part]
            
            # Remove common/generic parts
            generic_parts = set(settings.GENERIC_URL_PARTS)
            
            # Filter out generic parts and empty strings
            meaningful_parts = [
                part for part in path_parts 
                if part.lower() not in generic_parts and part.strip()
            ]
            
            # Remove file extensions from the last part
            if meaningful_parts:
                last_part = meaningful_parts[-1]
                # Remove common image extensions
                for ext in settings.IMAGE_EXTENSIONS:
                    if last_part.lower().endswith(ext):
                        last_part = last_part[:-len(ext)]
                        break
                
                # Replace the last part with cleaned version
                meaningful_parts[-1] = last_part
            
            # Convert underscores and hyphens to spaces
            keywords = []
            for part in meaningful_parts:
                # Replace underscores and hyphens with spaces
                clean_part = part.replace('_', ' ').replace('-', ' ')
                # Split by spaces and add non-empty parts
                keywords.extend([kw.strip() for kw in clean_part.split() if kw.strip()])
            
            # Remove duplicates while preserving order
            unique_keywords = []
            for kw in keywords:
                if kw.lower() not in [uk.lower() for uk in unique_keywords]:
                    unique_keywords.append(kw)
            
            # Return the most meaningful keyword (usually the last one)
            if unique_keywords:
                return unique_keywords[-1]
            else:
                return "photo"
                
        except Exception as e:
            logger.error(f"Error extracting keywords from URL: {url}", error=str(e))
            return "photo"

    def _generate_keyword_combinations(self, keywords: List[str]) -> List[str]:
        """
        Generates keyword combinations in sequential fallback order:
        1. All keywords together
        2. First combination of (n-1) keywords
        3. Second combination of (n-1) keywords
        4. ... all (n-1) combinations
        5. First combination of (n-2) keywords
        6. Second combination of (n-2) keywords
        7. ... and so on until single keywords
        """
        combinations_list = []
        n = len(keywords)
        
        if not keywords:
            return combinations_list
        
        # Add all keywords together first, if more than one keyword exists
        if n > 1:
            combinations_list.append(" ".join(keywords))
        
        # Generate combinations from largest to smallest, maintaining order
        for r in range(n - 1, 0, -1):
            # Convert combinations iterator to list to maintain order
            r_combinations = list(combinations(keywords, r))
            for combo in r_combinations:
                combinations_list.append(" ".join(combo))
        
        # Finally, add the single keywords themselves if they weren't the only thing
        if n > 1:
            combinations_list.extend(keywords)

        # If only one keyword, that's the only "combination"
        if n == 1:
            combinations_list.append(keywords[0])

        return combinations_list

    async def _fetch_image_url_from_openverse(self, keywords: str) -> Optional[str]:
        """
        Fetch a random, unused image URL from Openverse for the given keywords
        without performing validation.
        """
        query = keywords if keywords else "random"
        logger.info("Fetching image from Openverse API", keywords=query)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}?format=json&q={query}")
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    logger.warning("No image results found for keywords", keywords=query)
                    return None

                # Find the first URL that has not been used in this batch
                for res in results:
                    url = res.get("url")
                    if url and url not in self.used_urls:
                        self.used_urls.add(url)
                        logger.info(f"Found unused URL: {url}")
                        return url

                # If all URLs are already used, return the first one as a fallback
                if results and results[0].get("url"):
                    first_url = results[0].get("url")
                    logger.warning("All found URLs were already used in this batch. Re-using URL.", url=first_url)
                    return first_url

                logger.warning("No URLs found in Openverse results", keywords=query)
                return None

        except httpx.HTTPStatusError as e:
            logger.error(
                "Openverse API request failed",
                status_code=e.response.status_code,
                keywords=query,
            )
            return None
        except Exception as e:
            logger.error(
                "An unexpected error occurred during Openverse API call",
                error=str(e),
            )
            return None

    async def _fetch_image_with_fallback(self, keyword_string: str) -> str:
        """
        Fetches image using keyword fallback strategy.
        """
        keywords = self._extract_keywords_from_string(keyword_string)
        if not keywords:
            logger.warning("No keywords extracted from string", keyword_string=keyword_string)
            return "https://via.placeholder.com/150"

        keyword_combinations = self._generate_keyword_combinations(keywords)
        
        for combo in keyword_combinations:
            logger.info(f"Trying keyword combination: {combo}")
            try:
                image_url = await self._fetch_image_url_from_openverse(combo)
                if image_url:
                    logger.info(f"Found image for keywords: {combo}")
                    return image_url
            except Exception as e:
                logger.warning(f"Keyword combination failed: {combo}, error: {e}")
                continue

        logger.warning("No images found for any keyword combination", keywords=keywords)
        return "https://via.placeholder.com/150"

    async def _traverse_and_enrich(
        self, data: Any, parent_key: str = "", confirmed_keys: Optional[List[str]] = None
    ) -> Any:
        """
        Recursively traverses the data structure, enriching fields that are confirmed
        to be image keywords.
        """
        confirmed_keys = confirmed_keys or []
            
        if isinstance(data, dict):
            # Create a list of keys to avoid issues with changing dict size during iteration
            keys = list(data.keys())
            for key in keys:
                data[key] = await self._traverse_and_enrich(
                    data[key], 
                    parent_key=key, 
                    confirmed_keys=confirmed_keys
                )
            return data
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = await self._traverse_and_enrich(
                    data[i], 
                    parent_key=parent_key, 
                    confirmed_keys=confirmed_keys
                )
            return data
        elif isinstance(data, str):
            # The key to check is the parent key of the string value.
            current_key = parent_key
            
            # Check if this key should be processed.
            # This logic confirms if the current key (e.g., 'image') is in the list of
            # keys that the analyzer identified as needing enrichment.
            should_process = any(
                current_key in confirmed_key or confirmed_key.endswith(f".{current_key}")
                for confirmed_key in confirmed_keys
            )
            
            if should_process and self._is_keyword_string(current_key, data):
                logger.info(f"Processing keyword string: '{data}' for key: '{current_key}'")
                return await self._fetch_image_with_fallback(data)
            
            return data
        else:
            return data

    async def enrich_mock_data(
        self, mock_data: List[Dict[str, Any]], confirmed_keys: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively traverses the mock data, finds fields that look like image URLs
        or keyword strings, and replaces them with real URLs from the Openverse API.
        Only processes keys that are in the confirmed_keys list.
        """
        logger.info("Starting image enrichment process...")
        if confirmed_keys:
            logger.info(f"Processing only confirmed image keys: {confirmed_keys}")
        else:
            logger.info("Processing all potential image keys (no confirmed keys provided)")
            
        self.used_urls.clear()  # Reset for each new batch
        enriched_data = await self._traverse_and_enrich(mock_data, confirmed_keys=confirmed_keys)
        logger.info("Image enrichment process completed.")
        return enriched_data


# Singleton instance
image_enrichment_service = ImageEnrichmentService()
