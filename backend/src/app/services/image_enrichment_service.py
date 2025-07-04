import asyncio
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

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
        Determines if a value is a keyword string in the format "img_keyword1+keyword2+keyword3"
        or a URL starting with http/https
        """
        if not isinstance(value, str):
            return False

        # Check if it starts with "img_" and contains "+"
        if value.startswith("img_") and "+" in value:
            return True

        # Check if it's a URL starting with http/https
        if value.lower().startswith(("http://", "https://")):
            return True

        return False

    def _extract_keywords_from_string(self, keyword_string: str) -> List[str]:
        """
        Extracts keywords from a string in the format "img_tv+Smart+LED"
        Returns: ["tv", "smart", "led"]
        """
        try:
            # Remove "img_" prefix and split by "+"
            if keyword_string.startswith("img_"):
                keywords_part = keyword_string[4:]  # Remove "img_"
                keywords = [kw.strip() for kw in keywords_part.split("+") if kw.strip()]
                return keywords
            return []
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
        
        # Add all keywords together first
        combinations_list.append(" ".join(keywords))
        
        # Generate combinations from largest to smallest, maintaining order
        for r in range(n - 1, 0, -1):
            # Convert combinations iterator to list to maintain order
            r_combinations = list(combinations(keywords, r))
            for combo in r_combinations:
                combinations_list.append(" ".join(combo))
        
        return combinations_list

    async def _validate_image_url(self, url: str) -> bool:
        """
        Validates if an image URL returns actual image content.
        Returns True if the URL returns a valid image, False otherwise.
        """
        try:
            # Create client with SSL verification disabled for problematic domains
            ssl_verify = True
            if any(domain in url.lower() for domain in settings.DISABLE_SSL_DOMAINS):
                ssl_verify = False
                logger.info(f"Disabling SSL verification for problematic domain: {url}")
            
            async with httpx.AsyncClient(timeout=settings.IMAGE_VALIDATION_TIMEOUT, verify=ssl_verify) as client:
                # First check with HEAD request for content type
                head_response = await client.head(url, follow_redirects=True)
                if head_response.status_code != 200:
                    logger.warning(f"HEAD request failed for {url}, status: {head_response.status_code}")
                    return False
                
                content_type = head_response.headers.get('content-type', '').lower()
                
                # Check if content type indicates an image
                if not any(img_type in content_type for img_type in settings.IMAGE_CONTENT_TYPES):
                    logger.warning(f"URL does not return image content type: {url}, content-type: {content_type}")
                    return False
                
                # Make a GET request to check actual content
                get_response = await client.get(url, follow_redirects=True, timeout=settings.IMAGE_FETCH_TIMEOUT)
                if get_response.status_code != 200:
                    logger.warning(f"GET request failed for {url}, status: {get_response.status_code}")
                    return False
                
                # Check content type again from GET response
                content_type = get_response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in settings.IMAGE_CONTENT_TYPES):
                    logger.warning(f"GET response does not return image content type: {url}, content-type: {content_type}")
                    return False
                
                # Check if response body contains error messages or non-image content
                content = get_response.text[:settings.IMAGE_CONTENT_CHECK_SIZE]  # Check first N characters
                error_indicators = settings.ERROR_INDICATORS
                
                content_lower = content.lower()
                for indicator in error_indicators:
                    if indicator in content_lower:
                        logger.warning(f"URL returns error content: {url}, found: {indicator}")
                        return False
                
                # Check if content is too short (likely not an image)
                if len(get_response.content) < settings.MIN_IMAGE_SIZE_BYTES:  # Less than minimum size is suspicious
                    logger.warning(f"URL returns suspiciously small content: {url}, size: {len(get_response.content)} bytes")
                    return False
                
                logger.info(f"URL validation successful: {url}, content-type: {content_type}, size: {len(get_response.content)} bytes")
                return True
                
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {str(e)}")
            return False

    async def _validate_urls_concurrently(self, urls: List[str]) -> List[Tuple[str, bool]]:
        """
        Validates multiple URLs concurrently and returns list of (url, is_valid) tuples.
        """
        async def validate_single_url(url: str) -> Tuple[str, bool]:
            is_valid = await self._validate_image_url(url)
            return (url, is_valid)

        # Create tasks for concurrent validation
        tasks = [validate_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"URL validation task failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

    async def _fetch_image_url_from_openverse(self, keywords: str) -> Optional[str]:
        """Fetch a random image URL from Openverse for the given keywords."""
        query = keywords if keywords else "random"
        logger.info("Fetching image from Openverse API", keywords=query)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.api_url}?format=json&q={query}"
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    logger.warning(
                        "No image results found for keywords", keywords=query
                    )
                    return None

                possible_urls = [res["url"] for res in results if "url" in res]
                if not possible_urls:
                    return None

                random.shuffle(possible_urls)

                # Separate unused and used URLs
                unused_urls = [url for url in possible_urls if url not in self.used_urls]
                used_urls = [url for url in possible_urls if url in self.used_urls]

                # Validate URLs concurrently
                if unused_urls:
                    # Validate unused URLs first
                    validation_results = await self._validate_urls_concurrently(unused_urls)
                    valid_unused_urls = [url for url, is_valid in validation_results if is_valid]
                    
                    if valid_unused_urls:
                        # Select first valid unused URL (not random)
                        selected_url = valid_unused_urls[0]
                        self.used_urls.add(selected_url)
                        logger.info(f"Found valid unused URL: {selected_url}")
                        return selected_url

                # If no valid unused URLs, try used URLs
                if used_urls:
                    validation_results = await self._validate_urls_concurrently(used_urls)
                    valid_used_urls = [url for url, is_valid in validation_results if is_valid]
                    
                    if valid_used_urls:
                        # Select first valid used URL (not random)
                        selected_url = valid_used_urls[0]
                        logger.info(f"Found valid URL (reused): {selected_url}")
                        return selected_url

                logger.warning("No valid URLs found in Openverse results", keywords=query)
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
        Fetches image using sequential keyword fallback strategy:
        1. Try all keywords together
        2. If no results, try first (n-1) combination
        3. If no results, try next (n-1) combination
        4. Continue until all (n-1) combinations exhausted
        5. Move to (n-2) combinations, and so on
        6. Fallback to placeholder if none work
        """
        keywords = self._extract_keywords_from_string(keyword_string)
        if not keywords:
            logger.warning("No keywords extracted from string", keyword_string=keyword_string)
            return "https://via.placeholder.com/150"

        keyword_combinations = self._generate_keyword_combinations(keywords)
        
        # Try each combination sequentially in order
        for combo in keyword_combinations:
            logger.info(f"Trying keyword combination: {combo}")
            try:
                image_url = await self._fetch_image_url_from_openverse(combo)
                if image_url:
                    logger.info(f"Found image for keywords: {combo}")
                    return image_url
                else:
                    logger.info(f"No valid images found for keywords: {combo}")
            except Exception as e:
                logger.warning(f"Keyword combination failed: {combo}, error: {e}")
                continue

        logger.warning("No images found for any keyword combination", keywords=keywords)
        return "https://via.placeholder.com/150"

    async def _traverse_and_enrich(
        self, data: Any, parent_key: str = "", confirmed_keys: List[str] = None
    ) -> Any:
        """
        Recursively traverses the data structure, passing down the parent key
        to correctly identify and enrich image URLs or keyword strings.
        Only processes keys that are in the confirmed_keys list.
        """
        if confirmed_keys is None:
            confirmed_keys = []
            
        if isinstance(data, dict):
            return {
                key: await self._traverse_and_enrich(
                    value, 
                    parent_key=key, 
                    confirmed_keys=confirmed_keys
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                await self._traverse_and_enrich(
                    item, 
                    parent_key=parent_key, 
                    confirmed_keys=confirmed_keys
                )
                for item in data
            ]
        elif isinstance(data, str):
            # Check if this key should be processed
            should_process = False
            
            # Check exact match first
            if parent_key in confirmed_keys:
                should_process = True
            else:
                # Check if any confirmed key contains this parent_key
                for confirmed_key in confirmed_keys:
                    if parent_key in confirmed_key or confirmed_key.endswith(f".{parent_key}"):
                        should_process = True
                        break
            
            if should_process:
                if self._is_keyword_string(parent_key, data):
                    if data.startswith("img_") and "+" in data:
                        # Handle keyword string format
                        logger.info(f"Processing keyword string: {data} for key: {parent_key}")
                        return await self._fetch_image_with_fallback(data)
                    elif data.lower().startswith(("http://", "https://")):
                        # Handle URL format - extract keywords from URL
                        logger.info(f"Processing URL: {data} for key: {parent_key}")
                        keywords = self._extract_keywords_from_url(data)
                        return await self._fetch_image_url_from_openverse(keywords)
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
