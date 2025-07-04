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
        self.expiration_seconds = settings.CACHE_EXPIRATION_SECONDS
        
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
                # Refresh TTL for accessed data
                await self.refresh_cache_ttl(group_id)
                return json.loads(data_str)
            return None
        except redis.exceptions.RedisError as e:
            logger.error("Redis error while getting cached data", error=str(e))
            raise CacheServiceError("Failed to retrieve data from cache.")

    async def refresh_cache_ttl(self, group_id: str):
        """
        Refresh the TTL for a cache group when it's accessed.
        """
        try:
            group_key = f"group:{group_id}"
            mock_data_key = f"{group_key}:mockData"
            hashes_key = f"{group_key}:hashes"
            
            # Refresh TTL for both keys
            await self.redis_client.expire(mock_data_key, self.expiration_seconds)
            await self.redis_client.expire(hashes_key, self.expiration_seconds)
            
            # Refresh TTL for all hash groups associated with this group
            hashes = await self.redis_client.lrange(hashes_key, 0, -1)
            for hash_value in hashes:
                await self.redis_client.expire(f"hash:{hash_value}:groups", self.expiration_seconds)
                
            logger.debug(f"Refreshed TTL for cache group {group_id}")
        except Exception as e:
            logger.error(f"Error refreshing TTL for group {group_id}: {e}")

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
                await self.redis_client.expire(f"hash:{h}:groups", self.expiration_seconds)
                
                # Update bloom filter
                self.bloom_filter.add(h)

            # Set expiration for the group hash list
            await self.redis_client.expire(group_hash_key, self.expiration_seconds)

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
            
            # Update mock data with expiration
            pipe.set(f"{group_key}:mockData", json.dumps(mock_data), ex=self.expiration_seconds)
            
            # Update hashes in group (limit to hash_limit)
            hashes_to_store = object_hashes[-self.hash_limit:] if len(object_hashes) > self.hash_limit else object_hashes
            pipe.delete(f"{group_key}:hashes")  # Clear existing hashes
            pipe.rpush(f"{group_key}:hashes", *hashes_to_store)
            pipe.expire(f"{group_key}:hashes", self.expiration_seconds)
            
            # Update reverse index: hash → groups
            for hash in hashes_to_store:
                pipe.sadd(f"hash:{hash}:groups", group_id)
                pipe.expire(f"hash:{hash}:groups", self.expiration_seconds)
                # Add to bloom filter
                self.bloom_filter.add(hash)

            await pipe.execute()
            logger.info("Updated existing cache group with new data", group_id=group_id, expiration_hours=self.expiration_seconds/3600)
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
            
            # Store mock data with expiration
            pipe.set(f"{group_key}:mockData", json.dumps(mock_data), ex=self.expiration_seconds)
            
            # Store hashes in group (limit to hash_limit)
            hashes_to_store = object_hashes[-self.hash_limit:] if len(object_hashes) > self.hash_limit else object_hashes
            pipe.rpush(f"{group_key}:hashes", *hashes_to_store)
            pipe.expire(f"{group_key}:hashes", self.expiration_seconds)
            
            # Create reverse index: hash → groups
            for hash in hashes_to_store:
                pipe.sadd(f"hash:{hash}:groups", group_id)
                pipe.expire(f"hash:{hash}:groups", self.expiration_seconds)
                # Add to bloom filter
                self.bloom_filter.add(hash)

            await pipe.execute()
            logger.info("Created new cache group with reverse index", group_id=group_id, expiration_hours=self.expiration_seconds/3600)
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

    async def cleanup_expired_entries(self):
        """
        Clean up expired entries from bloom filter and reverse index.
        This should be called periodically to prevent bloom filter false positives.
        """
        try:
            # Get all hash keys and check their TTL
            all_hash_keys = await self.redis_client.keys("hash:*:groups")
            expired_hashes = []
            
            for hash_key in all_hash_keys:
                ttl = await self.redis_client.ttl(hash_key)
                if ttl == -2:  # Key doesn't exist (already expired)
                    hash_value = hash_key.split(":")[1]
                    expired_hashes.append(hash_value)
                    logger.debug(f"Found expired hash: {hash_value[:8]}...")
            
            # Remove expired hashes from bloom filter
            for hash_value in expired_hashes:
                # Note: BloomFilter doesn't support removal, so we'll just log it
                logger.info(f"Hash {hash_value[:8]}... has expired and will be ignored by bloom filter")
            
            logger.info(f"Cleanup completed: {len(expired_hashes)} expired hashes found")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    async def get_cache_info(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cache information including TTL for a group.
        """
        try:
            group_key = f"group:{group_id}"
            mock_data_key = f"{group_key}:mockData"
            hashes_key = f"{group_key}:hashes"
            
            # Get TTL for both keys
            mock_data_ttl = await self.redis_client.ttl(mock_data_key)
            hashes_ttl = await self.redis_client.ttl(hashes_key)
            
            # Get data size
            data_str = await self.redis_client.get(mock_data_key)
            data_size = len(data_str) if data_str else 0
            
            return {
                "group_id": group_id,
                "mock_data_ttl": mock_data_ttl,
                "hashes_ttl": hashes_ttl,
                "data_size_bytes": data_size,
                "expires_in_hours": max(mock_data_ttl, hashes_ttl) / 3600 if max(mock_data_ttl, hashes_ttl) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting cache info for group {group_id}: {e}")
            return None


# Singleton instance
cache_service = CacheService(settings.REDIS_URL)
