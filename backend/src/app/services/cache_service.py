import asyncio
import json
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from app.config.settings import settings
from app.utils.app_exceptions import CacheServiceError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self.hash_limit = settings.CACHE_GROUP_HASH_LIMIT

    async def find_group_by_hashes(
        self, object_hashes: List[str]
    ) -> Optional[str]:
        """
        Search for a cache group that contains at least one of the given hashes.
        """
        try:
            # This is a simplification. A real implementation would need to scan
            # or use secondary indexes to be efficient. For this example, we'll
            # iterate through keys, which is NOT performant in production.
            all_group_keys = await self.redis_client.keys("group:*")
            for group_key in all_group_keys:
                # Check for intersection between stored hashes and new hashes
                stored_hashes = await self.redis_client.lrange(
                    f"{group_key}:hashes", 0, -1
                )
                if any(h in stored_hashes for h in object_hashes):
                    logger.info(
                        "Found matching cache group", group_id=group_key
                    )
                    return group_key.split(":")[1]
            return None
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while finding group by hashes", error=str(e)
            )
            raise CacheServiceError(
                "Failed to search cache for matching groups."
            )

    async def get_data_if_cache_hit(
        self, object_hashes: List[str], required_count: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Efficiently checks for a cache hit and returns data if the count is sufficient.
        This combines finding the group and getting the data into a more optimal flow.
        """
        try:
            # Note: This key-scanning approach is inefficient for large Redis instances.
            # A production system might use secondary indexing or a different data model.
            all_group_keys = await self.redis_client.keys("group:*:hashes")
            for group_hash_key in all_group_keys:
                stored_hashes = await self.redis_client.lrange(
                    group_hash_key, 0, -1
                )
                if any(h in stored_hashes for h in object_hashes):
                    group_id = group_hash_key.split(":")[1]
                    logger.info("Found matching cache group", group_id=group_id)

                    # Now get the data and check if it's sufficient
                    cached_data = await self.get_cached_mock_data(group_id)
                    if cached_data and len(cached_data) >= required_count:
                        # Non-blocking task to update hashes for this group
                        asyncio.create_task(
                            self.update_group_hashes(group_id, object_hashes)
                        )
                        return cached_data
            return None  # No group found or data was insufficient

        except redis.exceptions.RedisError as e:
            logger.error("Redis error during cache hit check", error=str(e))
            raise CacheServiceError("Failed to check cache for a valid hit.")

    async def get_cached_mock_data(
        self, group_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached mock data for a given group ID.
        """
        try:
            data_str = await self.redis_client.get(f"group:{group_id}:mockData")
            if data_str:
                logger.info("Cache hit", group_id=group_id)
                return json.loads(data_str)
            return None
        except redis.exceptions.RedisError as e:
            logger.error("Redis error while getting cached data", error=str(e))
            raise CacheServiceError("Failed to retrieve data from cache.")

    async def update_group_hashes(self, group_id: str, new_hashes: List[str]):
        """
        Append new, unique hashes to a group's hash list and evict old ones if limit is exceeded.
        """
        group_hash_key = f"group:{group_id}:hashes"
        try:
            for h in new_hashes:
                # LREM 0 h -> remove all occurrences of h
                # RPUSH h -> add h to the right
                # This ensures the hash is at the end (most recent) and unique
                await self.redis_client.lrem(group_hash_key, 0, h)
                await self.redis_client.rpush(group_hash_key, h)

            # Trim the list to the max size
            await self.redis_client.ltrim(group_hash_key, -self.hash_limit, -1)
            logger.info("Updated hashes for group", group_id=group_id)
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while updating group hashes", error=str(e)
            )
            # This is a non-critical error, so we just log it.

    async def create_new_group_cache(
        self, object_hashes: List[str], mock_data: List[Dict[str, Any]]
    ) -> str:
        """
        Create a new cache group with its hashes and mock data.
        """
        try:
            group_id = await self.redis_client.incr("next_group_id")
            group_key = f"group:{group_id}"

            pipe = self.redis_client.pipeline()
            # Store mock data
            pipe.set(f"{group_key}:mockData", json.dumps(mock_data))
            # Store hashes
            pipe.rpush(f"{group_key}:hashes", *object_hashes)

            await pipe.execute()
            logger.info("Created new cache group", group_id=group_id)
            return str(group_id)
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while creating new cache group", error=str(e)
            )
            raise CacheServiceError(
                "Failed to create a new group in the cache."
            )


# Singleton instance
cache_service = CacheService(settings.REDIS_URL)
