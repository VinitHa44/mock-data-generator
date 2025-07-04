#!/usr/bin/env python3
"""
Test script to verify the truncation fixes for large JSON responses.
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

async def test_truncation_fix(session: aiohttp.ClientSession, count: int = 20) -> Dict:
    """Test the truncation fix with a moderate generation size."""
    print(f"\n🧪 Testing truncation fix ({count} items)...")
    
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
        
        # Check for truncation issues
        if success and data_count > 0:
            success_rate = data_count / count
            if success_rate >= 0.8:
                print("🎉 No truncation issues detected!")
            elif success_rate >= 0.5:
                print("⚠️  Some truncation issues, but partial success")
            else:
                print("❌ Significant truncation issues detected")
        
        return {
            "count": count,
            "duration": duration,
            "success": success,
            "data_count": data_count,
            "success_rate": data_count / count if count > 0 else 0
        }

async def test_adaptive_generation(session: aiohttp.ClientSession, count: int = 30) -> Dict:
    """Test the adaptive generation strategy."""
    print(f"\n🧪 Testing adaptive generation ({count} items)...")
    
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
            "data_count": data_count,
            "success_rate": data_count / count if count > 0 else 0
        }

async def test_large_generation(session: aiohttp.ClientSession, count: int = 50) -> Dict:
    """Test large generation to verify the fixes work for bigger requests."""
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
            "data_count": data_count,
            "success_rate": data_count / count if count > 0 else 0
        }

async def main():
    """Main test function."""
    print("🔧 Testing Truncation Fixes")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Truncation fix
        result1 = await test_truncation_fix(session, 20)
        
        # Test 2: Adaptive generation
        result2 = await test_adaptive_generation(session, 30)
        
        # Test 3: Large generation
        result3 = await test_large_generation(session, 50)
        
        # Summary
        print(f"\n📋 Test Summary")
        print("=" * 30)
        print(f"Truncation fix (20):    {'✅' if result1['success_rate'] >= 0.8 else '⚠️' if result1['success_rate'] >= 0.5 else '❌'}")
        print(f"Adaptive generation (30): {'✅' if result2['success_rate'] >= 0.8 else '⚠️' if result2['success_rate'] >= 0.5 else '❌'}")
        print(f"Large generation (50):   {'✅' if result3['success_rate'] >= 0.8 else '⚠️' if result3['success_rate'] >= 0.5 else '❌'}")
        
        # Success rate analysis
        total_tests = 3
        successful_tests = sum([
            result1['success_rate'] >= 0.8,
            result2['success_rate'] >= 0.8, 
            result3['success_rate'] >= 0.8
        ])
        
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n📈 High Success Rate Tests: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        # Performance analysis
        if result1['success'] and result2['success'] and result3['success']:
            print(f"\n📈 Performance Analysis")
            print("=" * 30)
            print(f"Small (20):  {result1['duration']:6.2f}s ({result1['data_count']/result1['duration']:5.1f} items/sec)")
            print(f"Medium (30): {result2['duration']:6.2f}s ({result2['data_count']/result2['duration']:5.1f} items/sec)")
            print(f"Large (50):  {result3['duration']:6.2f}s ({result3['data_count']/result3['duration']:5.1f} items/sec)")
        
        if success_rate >= 66:
            print("🎉 Truncation fixes appear to be working!")
        elif success_rate >= 33:
            print("⚠️  Partial success - some fixes working")
        else:
            print("❌ Significant issues remain with truncation")
        
        print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 