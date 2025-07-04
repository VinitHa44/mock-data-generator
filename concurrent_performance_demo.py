#!/usr/bin/env python3
"""
Concurrent Performance Demo for Mock Data Generator

This script demonstrates the performance improvements achieved through
the concurrent/parallel model architecture for handling 500 samples per user.
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

async def test_single_generation(session: aiohttp.ClientSession, count: int) -> Dict:
    """Test single generation with specified count."""
    start_time = time.time()
    
    async with session.post(
        f"{API_BASE_URL}/generate-mock-data",
        params={"count": count},
        json=SAMPLE_DATA
    ) as response:
        result = await response.json()
        duration = time.time() - start_time
        
        return {
            "count": count,
            "duration": duration,
            "success": response.status == 200,
            "method": "single"
        }

async def test_concurrent_generation(session: aiohttp.ClientSession, total_count: int, batch_size: int = 50) -> Dict:
    """Test concurrent generation with batching."""
    start_time = time.time()
    
    # Calculate batches
    num_batches = (total_count + batch_size - 1) // batch_size
    tasks = []
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_count - i * batch_size)
        task = test_single_generation(session, current_batch_size)
        tasks.append(task)
    
    # Execute all batches concurrently
    results = await asyncio.gather(*tasks)
    
    total_duration = time.time() - start_time
    total_generated = sum(r["count"] for r in results if r["success"])
    
    return {
        "count": total_count,
        "duration": total_duration,
        "success": all(r["success"] for r in results),
        "method": f"concurrent_{batch_size}",
        "batches": num_batches,
        "items_per_second": total_generated / total_duration if total_duration > 0 else 0
    }

async def get_performance_metrics(session: aiohttp.ClientSession) -> Dict:
    """Get current performance metrics from the API."""
    async with session.get(f"{API_BASE_URL}/performance") as response:
        if response.status == 200:
            return await response.json()
        return {"error": f"Failed to get metrics: {response.status}"}

async def main():
    """Main demonstration function."""
    print("🚀 Concurrent Performance Demo for Mock Data Generator")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Single generation of 500 items
        print("\n📊 Test 1: Single Generation (500 items)")
        print("-" * 40)
        result1 = await test_single_generation(session, 500)
        print(f"Duration: {result1['duration']:.2f}s")
        print(f"Success: {result1['success']}")
        print(f"Throughput: {500/result1['duration']:.1f} items/sec")
        
        # Test 2: Concurrent generation of 500 items
        print("\n📊 Test 2: Concurrent Generation (500 items, batch size 50)")
        print("-" * 40)
        result2 = await test_concurrent_generation(session, 500, 50)
        print(f"Duration: {result2['duration']:.2f}s")
        print(f"Success: {result2['success']}")
        print(f"Batches: {result2['batches']}")
        print(f"Throughput: {result2['items_per_second']:.1f} items/sec")
        
        # Test 3: Different batch sizes
        print("\n📊 Test 3: Different Batch Sizes (500 items)")
        print("-" * 40)
        batch_sizes = [25, 50, 100]
        for batch_size in batch_sizes:
            result = await test_concurrent_generation(session, 500, batch_size)
            print(f"Batch size {batch_size:3d}: {result['duration']:6.2f}s "
                  f"({result['items_per_second']:.1f} items/sec)")
        
        # Performance comparison
        print("\n📈 Performance Comparison")
        print("-" * 40)
        if result1['success'] and result2['success']:
            improvement = (result1['duration'] - result2['duration']) / result1['duration'] * 100
            speedup = result1['duration'] / result2['duration']
            print(f"Single generation:    {result1['duration']:6.2f}s")
            print(f"Concurrent generation: {result2['duration']:6.2f}s")
            print(f"Improvement:          {improvement:6.1f}%")
            print(f"Speedup:              {speedup:6.1f}x")
        
        # Get API performance metrics
        print("\n📊 API Performance Metrics")
        print("-" * 40)
        metrics = await get_performance_metrics(session)
        if "metrics" in metrics:
            for metric_name, stats in metrics["metrics"].items():
                print(f"{metric_name}: avg={stats['avg']:.2f}s, "
                      f"min={stats['min']:.2f}s, max={stats['max']:.2f}s")
        
        print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main()) 