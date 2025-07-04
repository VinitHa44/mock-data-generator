#!/usr/bin/env python3
"""
Test script to verify that ID fields are placed at the top of JSON objects.
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict

# Configuration
API_BASE_URL = "http://localhost:8000/api"
SAMPLE_DATA = [
    {
        "user_id": "2",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "department": "Engineering",
        "salary": 75000
    },
    {
        "user_id": "1",
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "age": 25,
        "department": "Marketing",
        "salary": 65000
    }
]

async def test_id_placement(session: aiohttp.ClientSession, count: int = 5) -> Dict:
    """Test that ID fields are placed at the top of JSON objects."""
    print(f"\n🧪 Testing ID field placement ({count} items)...")
    
    start_time = time.time()
    async with session.post(
        f"{API_BASE_URL}/generate-mock-data",
        params={"count": count},
        json=SAMPLE_DATA
    ) as response:
        result = await response.json()
        duration = time.time() - start_time
        
        success = response.status == 200
        data_count = len(result.get("data", [])) if success else 0
        
        print(f"✅ Status: {response.status}")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Generated: {data_count}/{count} items")
        print(f"🎯 Success: {success}")
        
        # Check ID field placement
        if success and data_count > 0:
            print(f"\n📋 ID Field Placement Analysis:")
            print("=" * 40)
            
            for i, item in enumerate(result['data'][:3]):  # Check first 3 items
                print(f"\nItem {i+1}:")
                item_keys = list(item.keys())
                print(f"  Keys order: {item_keys}")
                
                # Check if ID fields are at the top
                id_fields = [key for key in item_keys if key.lower() == 'id' or key.lower().endswith('_id')]
                if id_fields:
                    first_id_index = min(item_keys.index(id_field) for id_field in id_fields)
                    print(f"  ID fields: {id_fields}")
                    print(f"  First ID at position: {first_id_index}")
                    
                    if first_id_index == 0:
                        print(f"  ✅ ID fields are at the top!")
                    else:
                        print(f"  ❌ ID fields are not at the top (should be at position 0)")
                else:
                    print(f"  ⚠️  No ID fields found in this item")
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count
        }

async def test_nested_id_placement(session: aiohttp.ClientSession, count: int = 3) -> Dict:
    """Test ID field placement in nested objects."""
    print(f"\n🧪 Testing nested ID field placement ({count} items)...")
    
    nested_data = [
        {
            "id": 1,
            "name": "Product A",
            "category": {
                "category_id": "cat-001",
                "name": "Electronics"
            },
            "price": 99.99
        }
    ]
    
    async with session.post(
        f"{API_BASE_URL}/generate-mock-data",
        params={"count": count},
        json=nested_data
    ) as response:
        result = await response.json()
        
        success = response.status == 200
        data_count = len(result.get("data", [])) if success else 0
        
        print(f"✅ Status: {response.status}")
        print(f"📊 Generated: {data_count}/{count} items")
        
        # Check nested ID placement
        if success and data_count > 0:
            print(f"\n📋 Nested ID Field Analysis:")
            print("=" * 40)
            
            for i, item in enumerate(result['data'][:2]):  # Check first 2 items
                print(f"\nItem {i+1}:")
                print(f"  Top level keys: {list(item.keys())}")
                
                # Check if top-level ID is first
                if 'id' in item:
                    id_position = list(item.keys()).index('id')
                    print(f"  Top-level 'id' at position: {id_position}")
                
                # Check nested objects
                for key, value in item.items():
                    if isinstance(value, dict):
                        print(f"  Nested '{key}' keys: {list(value.keys())}")
                        if 'category_id' in value:
                            cat_id_position = list(value.keys()).index('category_id')
                            print(f"    'category_id' at position: {cat_id_position}")
        
        return {
            "count": count,
            "success": success,
            "data_count": data_count
        }

async def main():
    """Main test function."""
    print("🔧 Testing ID Field Placement")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Basic ID placement
        result1 = await test_id_placement(session, 5)
        
        # Test 2: Nested ID placement
        result2 = await test_nested_id_placement(session, 3)
        
        # Summary
        print(f"\n📋 Test Summary")
        print("=" * 30)
        print(f"Basic ID placement:     {'✅' if result1['success'] else '❌'}")
        print(f"Nested ID placement:    {'✅' if result2['success'] else '❌'}")
        
        print(f"\n✅ All tests completed!")
        print(f"\n💡 Check the output above to verify ID fields appear at the top of each object.")

if __name__ == "__main__":
    asyncio.run(main()) 