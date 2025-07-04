#!/usr/bin/env python3
"""
Test script to verify the improved JSON parsing fixes.
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
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "department": "Engineering",
        "salary": 75000
    }
]

async def test_small_generation(session: aiohttp.ClientSession, count: int = 5) -> Dict:
    """Test very small generation to verify basic functionality."""
    print(f"\n🧪 Testing small generation ({count} items)...")
    
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
        
        if success and data_count > 0:
            print(f"📝 Sample item: {result['data'][0]}")
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count
        }

async def test_medium_generation(session: aiohttp.ClientSession, count: int = 15) -> Dict:
    """Test medium generation to verify batch processing."""
    print(f"\n🧪 Testing medium generation ({count} items)...")
    
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
        print(f"🚀 Throughput: {data_count/duration:.1f} items/sec")
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count
        }

async def test_large_generation(session: aiohttp.ClientSession, count: int = 30) -> Dict:
    """Test large generation to verify concurrent processing."""
    print(f"\n🧪 Testing large generation ({count} items)...")
    
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
        print(f"🚀 Throughput: {data_count/duration:.1f} items/sec")
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count
        }

async def test_error_recovery(session: aiohttp.ClientSession) -> Dict:
    """Test error recovery with problematic input."""
    print(f"\n🧪 Testing error recovery...")
    
    problematic_data = [
        {
            "name": "Test User",
            "description": "This has\nnewlines\tand\r\ncontrol characters",
            "tags": ["test", "problematic", "input"]
        }
    ]
    
    async with session.post(
        f"{API_BASE_URL}/generate-mock-data",
        params={"count": 10},
        json=problematic_data
    ) as response:
        result = await response.json()
        
        success = response.status == 200
        data_count = len(result.get("data", [])) if success else 0
        
        print(f"✅ Status: {response.status}")
        print(f"📊 Generated: {data_count}/10 items")
        print(f"🎯 Handled gracefully: {success}")
        
        return {
            "status": response.status,
            "success": success,
            "data_count": data_count
        }

async def main():
    """Main test function."""
    print("🔧 Testing Improved JSON Parsing")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Very small generation
        result1 = await test_small_generation(session, 5)
        
        # Test 2: Medium generation
        result2 = await test_medium_generation(session, 15)
        
        # Test 3: Large generation
        result3 = await test_large_generation(session, 30)
        
        # Test 4: Error recovery
        error_test = await test_error_recovery(session)
        
        # Summary
        print(f"\n📋 Test Summary")
        print("=" * 30)
        print(f"Small generation (5):   {'✅' if result1['success'] else '❌'}")
        print(f"Medium generation (15): {'✅' if result2['success'] else '❌'}")
        print(f"Large generation (30):  {'✅' if result3['success'] else '❌'}")
        print(f"Error recovery:         {'✅' if error_test['success'] else '❌'}")
        
        # Success rate analysis
        total_tests = 4
        successful_tests = sum([
            result1['success'],
            result2['success'], 
            result3['success'],
            error_test['success']
        ])
        
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        if success_rate >= 75:
            print("🎉 JSON parsing improvements appear to be working!")
        elif success_rate >= 50:
            print("⚠️  Partial success - some improvements working")
        else:
            print("❌ Significant issues remain with JSON parsing")
        
        print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 