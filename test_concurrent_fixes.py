#!/usr/bin/env python3
"""
Test script to verify the concurrent architecture fixes for JSON parsing issues.
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

async def test_small_generation(session: aiohttp.ClientSession, count: int = 10) -> Dict:
    """Test small generation to verify basic functionality."""
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
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count
        }

async def test_medium_generation(session: aiohttp.ClientSession, count: int = 50) -> Dict:
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

async def test_large_generation(session: aiohttp.ClientSession, count: int = 100) -> Dict:
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

async def test_performance_metrics(session: aiohttp.ClientSession) -> Dict:
    """Test performance metrics endpoint."""
    print(f"\n📊 Testing performance metrics...")
    
    async with session.get(f"{API_BASE_URL}/performance") as response:
        if response.status == 200:
            result = await response.json()
            print("✅ Performance metrics retrieved successfully")
            print(f"📈 Metrics available: {list(result.get('metrics', {}).keys())}")
            return result
        else:
            print(f"❌ Failed to get performance metrics: {response.status}")
            return {"error": f"Status {response.status}"}

async def test_error_handling(session: aiohttp.ClientSession) -> Dict:
    """Test error handling with invalid input."""
    print(f"\n🧪 Testing error handling...")
    
    invalid_data = [
        {
            "name": "Test",
            "description": "This contains\ncontrol\ncharacters\tand\r\nnewlines"
        }
    ]
    
    async with session.post(
        f"{API_BASE_URL}/generate-mock-data",
        params={"count": 5},
        json=invalid_data
    ) as response:
        result = await response.json()
        
        print(f"✅ Status: {response.status}")
        print(f"🎯 Handled gracefully: {response.status != 500}")
        
        return {
            "status": response.status,
            "handled_gracefully": response.status != 500
        }

async def main():
    """Main test function."""
    print("🔧 Testing Concurrent Architecture Fixes")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Small generation
        result1 = await test_small_generation(session, 10)
        
        # Test 2: Medium generation
        result2 = await test_medium_generation(session, 50)
        
        # Test 3: Large generation
        result3 = await test_large_generation(session, 100)
        
        # Test 4: Performance metrics
        metrics = await test_performance_metrics(session)
        
        # Test 5: Error handling
        error_test = await test_error_handling(session)
        
        # Summary
        print(f"\n📋 Test Summary")
        print("=" * 30)
        print(f"Small generation (10):  {'✅' if result1['success'] else '❌'}")
        print(f"Medium generation (50): {'✅' if result2['success'] else '❌'}")
        print(f"Large generation (100): {'✅' if result3['success'] else '❌'}")
        print(f"Performance metrics:    {'✅' if 'metrics' in metrics else '❌'}")
        print(f"Error handling:         {'✅' if error_test['handled_gracefully'] else '❌'}")
        
        # Performance analysis
        if result1['success'] and result2['success'] and result3['success']:
            print(f"\n📈 Performance Analysis")
            print("=" * 30)
            print(f"Small (10):   {result1['duration']:6.2f}s ({result1['data_count']/result1['duration']:5.1f} items/sec)")
            print(f"Medium (50):  {result2['duration']:6.2f}s ({result2['data_count']/result2['duration']:5.1f} items/sec)")
            print(f"Large (100):  {result3['duration']:6.2f}s ({result3['data_count']/result3['duration']:5.1f} items/sec)")
            
            # Check if concurrent processing is working
            if result3['duration'] < result2['duration'] * 1.5:  # Should be roughly linear
                print("🚀 Concurrent processing appears to be working!")
            else:
                print("⚠️  Concurrent processing may not be optimal")
        
        print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 