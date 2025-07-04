"""
Example demonstrating the new caching system with partial cache hits.

This shows how the system handles:
1. First request: Generate 10 items, cache them
2. Second request: Same input, request 20 items
   - Returns 10 from cache
   - Generates only 10 more (not 20 from scratch)
"""

import asyncio
import json
from typing import List, Dict, Any

# Mock the cache service for demonstration
class MockCacheService:
    def __init__(self):
        self.cache = {}
        self.hash_to_groups = {}
        self.bloom_filter = set()  # Simplified bloom filter
    
    def compute_hash(self, data: Dict) -> str:
        """Simple hash computation for demo"""
        serialized = json.dumps(data, sort_keys=True)
        return str(hash(serialized))
    
    async def get_partial_cache_hit(self, object_hashes: List[str], required_count: int):
        """Simulate partial cache hit logic"""
        for hash in object_hashes:
            if hash in self.bloom_filter:
                group_ids = self.hash_to_groups.get(hash, set())
                for group_id in group_ids:
                    cached_data = self.cache.get(group_id, [])
                    if cached_data:
                        cached_count = len(cached_data)
                        print(f"Found {cached_count} items in cache for hash {hash[:8]}...")
                        
                        if cached_count >= required_count:
                            # Full cache hit
                            return cached_data[:required_count], 0, group_id
                        else:
                            # Partial cache hit
                            remaining_needed = required_count - cached_count
                            print(f"Partial cache hit: returning {cached_count} items, need {remaining_needed} more")
                            return cached_data, remaining_needed, group_id
        
        print("No cache hit found")
        return None, required_count, None
    
    async def create_new_group_cache(self, object_hashes: List[str], mock_data: List[Dict]):
        """Simulate creating new cache entry"""
        group_id = len(self.cache) + 1
        
        # Store the data
        self.cache[group_id] = mock_data
        
        # Create reverse index
        for hash in object_hashes:
            if hash not in self.hash_to_groups:
                self.hash_to_groups[hash] = set()
            self.hash_to_groups[hash].add(group_id)
            self.bloom_filter.add(hash)
        
        print(f"Created cache group {group_id} with {len(mock_data)} items")
        return str(group_id)
    
    async def update_existing_group_cache(self, group_id: str, object_hashes: List[str], mock_data: List[Dict]):
        """Simulate updating existing cache entry"""
        # Store the updated data
        self.cache[int(group_id)] = mock_data
        
        # Update reverse index
        for hash in object_hashes:
            if hash not in self.hash_to_groups:
                self.hash_to_groups[hash] = set()
            self.hash_to_groups[hash].add(int(group_id))
            self.bloom_filter.add(hash)
        
        print(f"Updated cache group {group_id} with {len(mock_data)} items")

# Mock the generation service
class MockGenerationService:
    async def generate_mock_data(self, input_examples: List[Dict], count: int) -> List[Dict]:
        """Simulate generating mock data"""
        print(f"Generating {count} new items...")
        
        # Simulate generation time
        await asyncio.sleep(0.1)
        
        # Generate mock data based on input structure
        result = []
        for i in range(count):
            item = {}
            for key, value in input_examples[0].items():
                if isinstance(value, str):
                    item[key] = f"{value}_generated_{i+1}"
                elif isinstance(value, int):
                    item[key] = value + i + 1
                else:
                    item[key] = value
            result.append(item)
        
        return result

# Main demonstration
async def demonstrate_caching():
    print("=== Caching System Demonstration ===\n")
    
    # Initialize services
    cache_service = MockCacheService()
    generation_service = MockGenerationService()
    
    # Sample input data
    input_examples = [
        {"name": "John Doe", "email": "john@example.com", "age": 30},
        {"name": "Jane Smith", "email": "jane@example.com", "age": 25}
    ]
    
    print("Input examples:")
    for example in input_examples:
        print(f"  {example}")
    print()
    
    # Step 1: First request - Generate 10 items
    print("=== STEP 1: First Request (count=10) ===")
    count1 = 10
    
    # Check cache
    object_hashes = [cache_service.compute_hash(example) for example in input_examples]
    cached_data, remaining_count, group_id = await cache_service.get_partial_cache_hit(object_hashes, count1)
    
    if cached_data and remaining_count == 0:
        print("✅ Full cache hit - returning all data from cache")
        final_data = cached_data
    elif cached_data and remaining_count > 0:
        print(f"🔄 Partial cache hit - returning {len(cached_data)} from cache, generating {remaining_count} more")
        additional_data = await generation_service.generate_mock_data(input_examples, remaining_count)
        final_data = cached_data + additional_data
    else:
        print("❌ No cache hit - generating all data from scratch")
        final_data = await generation_service.generate_mock_data(input_examples, count1)
    
    # Cache the result
    if group_id:
        await cache_service.update_existing_group_cache(group_id, object_hashes, final_data)
    else:
        await cache_service.create_new_group_cache(object_hashes, final_data)
    
    print(f"✅ Generated {len(final_data)} items total")
    print(f"First 3 items: {final_data[:3]}")
    print()
    
    # Step 2: Second request - Same input, but request 20 items
    print("=== STEP 2: Second Request (count=20) ===")
    count2 = 20
    
    # Check cache again (same hashes)
    cached_data, remaining_count, group_id = await cache_service.get_partial_cache_hit(object_hashes, count2)
    
    if cached_data and remaining_count == 0:
        print("✅ Full cache hit - returning all data from cache")
        final_data = cached_data
    elif cached_data and remaining_count > 0:
        print(f"🔄 Partial cache hit - returning {len(cached_data)} from cache, generating {remaining_count} more")
        additional_data = await generation_service.generate_mock_data(input_examples, remaining_count)
        final_data = cached_data + additional_data
    else:
        print("❌ No cache hit - generating all data from scratch")
        final_data = await generation_service.generate_mock_data(input_examples, count2)
    
    # Update cache with new combined data
    if group_id:
        await cache_service.update_existing_group_cache(group_id, object_hashes, final_data)
    else:
        await cache_service.create_new_group_cache(object_hashes, final_data)
    
    print(f"✅ Generated {len(final_data)} items total")
    print(f"First 3 items: {final_data[:3]}")
    print()
    
    # Step 3: Third request - Same input, request 5 items (should be full cache hit)
    print("=== STEP 3: Third Request (count=5) ===")
    count3 = 5
    
    # Check cache again (same hashes)
    cached_data, remaining_count, group_id = await cache_service.get_partial_cache_hit(object_hashes, count3)
    
    if cached_data and remaining_count == 0:
        print("✅ Full cache hit - returning all data from cache")
        final_data = cached_data
    elif cached_data and remaining_count > 0:
        print(f"🔄 Partial cache hit - returning {len(cached_data)} from cache, generating {remaining_count} more")
        additional_data = await generation_service.generate_mock_data(input_examples, remaining_count)
        final_data = cached_data + additional_data
    else:
        print("❌ No cache hit - generating all data from scratch")
        final_data = await generation_service.generate_mock_data(input_examples, count3)
    
    print(f"✅ Generated {len(final_data)} items total")
    print(f"All items: {final_data}")
    print()
    
    print("=== Summary ===")
    print("✅ First request: Generated 10 items from scratch")
    print("🔄 Second request: Retrieved 10 from cache + generated 10 more = 20 total")
    print("✅ Third request: Retrieved 5 from cache (no generation needed)")
    print("\n🎯 Key Benefits:")
    print("  - Avoids regenerating data that's already available")
    print("  - Only generates the additional items needed")
    print("  - Significantly reduces LLM calls and processing time")
    print("  - Maintains data consistency across requests")

if __name__ == "__main__":
    asyncio.run(demonstrate_caching()) 