import asyncio
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as redis
from pybloom_live import BloomFilter
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
        
        # Bloom filter for quick rejection
        self.bloom_filter = BloomFilter(capacity=1000000, error_rate=0.1)
        
        # Initialize bloom filter synchronously during startup
        # Note: This should be called after the instance is created
        self._initialized = False

    async def initialize(self):
        """Initialize the cache service - must be called after creation"""
        if self._initialized:
            return
            
        try:
            # Load bloom filter state first
            await self.load_bloom_filter_state()
            
            # Then load existing hashes into bloom filter
            await self.initialize_bloom_filter()
            
            self._initialized = True
            logger.info("Cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            raise

    async def initialize_bloom_filter(self):
        """Load all existing hashes into bloom filter"""
        try:
            all_hash_keys = await self.redis_client.keys("hash:*:groups")
            for hash_key in all_hash_keys:
                hash_value = hash_key.split(":")[1]
                self.bloom_filter.add(hash_value)
            logger.info(f"Bloom filter initialized with {len(all_hash_keys)} hash entries")
        except Exception as e:
            logger.error(f"Failed to initialize bloom filter: {e}")
            raise

    async def save_bloom_filter_state(self):
        """Save bloom filter state to disk"""
        try:
            with open(settings.BLOOM_FILTER_FILE, 'wb') as f:
                pickle.dump(self.bloom_filter, f)
            logger.info("Bloom filter state saved")
        except Exception as e:
            logger.error(f"Failed to save bloom filter state: {e}")

    async def load_bloom_filter_state(self):
        """Load bloom filter state from disk"""
        try:
            if os.path.exists(settings.BLOOM_FILTER_FILE):
                with open(settings.BLOOM_FILTER_FILE, 'rb') as f:
                    self.bloom_filter = pickle.load(f)
                logger.info("Bloom filter state loaded from disk")
        except Exception as e:
            logger.error(f"Failed to load bloom filter state: {e}")



    async def get_partial_cache_hit(
        self, object_hashes: List[str], required_count: int
    ) -> Tuple[Optional[List[Dict[str, Any]]], int, Optional[str]]:
        """
        Check for partial cache hits and return cached data + count of items still needed.
        Returns: (cached_data, remaining_count_needed)
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            for hash in object_hashes:
                # STEP 1: Quick rejection using bloom filter (O(1))
                if hash not in self.bloom_filter:
                    logger.debug(f"Hash {hash[:8]}... not in bloom filter, skipping")
                    continue
                
                # STEP 2: Direct lookup using reverse index (O(1))
                group_ids = await self.redis_client.smembers(f"hash:{hash}:groups")
                
                if not group_ids:
                    continue
                    
                logger.info(f"Found {len(group_ids)} groups for hash {hash[:8]}...")
                
                # STEP 3: Check each group for available data
                for group_id in group_ids:
                    cached_data = await self.get_cached_mock_data(group_id)
                    if cached_data:
                        cached_count = len(cached_data)
                        logger.info(f"Partial cache hit for group {group_id}: {cached_count} items available")
                        
                        if cached_count >= required_count:
                            # Full cache hit - return all needed data
                            asyncio.create_task(self.update_group_hashes(group_id, object_hashes))
                            return cached_data[:required_count], 0, group_id
                        else:
                            # Partial cache hit - return what we have and indicate how many more needed
                            remaining_needed = required_count - cached_count
                            logger.info(f"Partial cache hit: returning {cached_count} items, need {remaining_needed} more")
                            asyncio.create_task(self.update_group_hashes(group_id, object_hashes))
                            return cached_data, remaining_needed, group_id
            
            logger.info("No cache hit found")
            return None, required_count, None
            
        except redis.exceptions.RedisError as e:
            logger.error("Redis error during partial cache hit check", error=str(e))
            raise CacheServiceError("Failed to check cache for partial hits.")

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
        Also updates reverse index and bloom filter.
        """
        group_hash_key = f"group:{group_id}:hashes"
        try:
            # Get current hashes to determine what to remove from reverse index
            current_hashes = await self.redis_client.lrange(group_hash_key, 0, -1)
            
            # Add new hashes to group's hash list
            for h in new_hashes:
                await self.redis_client.lrem(group_hash_key, 0, h)
                await self.redis_client.rpush(group_hash_key, h)
                
                # Update reverse index
                await self.redis_client.sadd(f"hash:{h}:groups", group_id)
                
                # Update bloom filter
                self.bloom_filter.add(h)

            # Trim the list to the max size and get removed hashes
            removed_count = await self.redis_client.llen(group_hash_key) - self.hash_limit
            if removed_count > 0:
                removed_hashes = await self.redis_client.lrange(group_hash_key, 0, removed_count - 1)
                await self.redis_client.ltrim(group_hash_key, removed_count, -1)
                
                # Remove old hashes from reverse index
                for old_hash in removed_hashes:
                    await self.redis_client.srem(f"hash:{old_hash}:groups", group_id)
                    
            logger.info("Updated hashes and reverse index for group", group_id=group_id)
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while updating group hashes", error=str(e)
            )
            # This is a non-critical error, so we just log it.

    async def update_existing_group_cache(
        self, group_id: str, object_hashes: List[str], mock_data: List[Dict[str, Any]]
    ) -> None:
        """
        Update an existing cache group with new data.
        This is used for partial cache hits where we add more data to an existing group.
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            group_key = f"group:{group_id}"

            pipe = self.redis_client.pipeline()
            
            # Update mock data
            pipe.set(f"{group_key}:mockData", json.dumps(mock_data))
            
            # Update hashes in group (limit to hash_limit)
            hashes_to_store = object_hashes[-self.hash_limit:] if len(object_hashes) > self.hash_limit else object_hashes
            pipe.delete(f"{group_key}:hashes")  # Clear existing hashes
            pipe.rpush(f"{group_key}:hashes", *hashes_to_store)
            
            # Update reverse index: hash → groups
            for hash in hashes_to_store:
                pipe.sadd(f"hash:{hash}:groups", group_id)
                # Add to bloom filter
                self.bloom_filter.add(hash)

            await pipe.execute()
            logger.info("Updated existing cache group with new data", group_id=group_id)
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while updating existing cache group", error=str(e)
            )
            raise CacheServiceError(
                "Failed to update existing group in the cache."
            )

    async def create_new_group_cache(
        self, object_hashes: List[str], mock_data: List[Dict[str, Any]]
    ) -> str:
        """
        Create a new cache group with its hashes and mock data.
        Also creates reverse index and updates bloom filter.
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            group_id = await self.redis_client.incr("next_group_id")
            group_key = f"group:{group_id}"

            pipe = self.redis_client.pipeline()
            
            # Store mock data
            pipe.set(f"{group_key}:mockData", json.dumps(mock_data))
            
            # Store hashes in group (limit to hash_limit)
            hashes_to_store = object_hashes[-self.hash_limit:] if len(object_hashes) > self.hash_limit else object_hashes
            pipe.rpush(f"{group_key}:hashes", *hashes_to_store)
            
            # Create reverse index: hash → groups
            for hash in hashes_to_store:
                pipe.sadd(f"hash:{hash}:groups", group_id)
                # Add to bloom filter
                self.bloom_filter.add(hash)

            await pipe.execute()
            logger.info("Created new cache group with reverse index", group_id=group_id)
            return str(group_id)
        except redis.exceptions.RedisError as e:
            logger.error(
                "Redis error while creating new cache group", error=str(e)
            )
            raise CacheServiceError(
                "Failed to create a new group in the cache."
            )

    async def shutdown(self):
        """Cleanup method to save bloom filter state"""
        try:
            await self.save_bloom_filter_state()
            logger.info("Cache service shutdown complete")
        except Exception as e:
            logger.error(f"Error during cache service shutdown: {e}")


# Singleton instance
cache_service = CacheService(settings.REDIS_URL)
