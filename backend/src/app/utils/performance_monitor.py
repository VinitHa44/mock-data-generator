import time
import asyncio
from typing import Dict, List, Optional
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "generation_times": [],
            "batch_times": [],
            "cache_hit_times": [],
            "total_request_times": []
        }
    
    def record_generation_time(self, count: int, duration: float, method: str = "single"):
        """Record the time taken for data generation."""
        self.metrics["generation_times"].append(duration)
        logger.info(
            f"Generation completed",
            count=count,
            duration=f"{duration:.2f}s",
            method=method,
            avg_time_per_item=f"{duration/count:.3f}s"
        )
    
    def record_batch_time(self, batch_size: int, duration: float):
        """Record the time taken for batch processing."""
        self.metrics["batch_times"].append(duration)
        logger.info(
            f"Batch processing completed",
            batch_size=batch_size,
            duration=f"{duration:.2f}s",
            throughput=f"{batch_size/duration:.1f} items/sec"
        )
    
    def record_cache_hit_time(self, duration: float):
        """Record the time taken for cache hits."""
        self.metrics["cache_hit_times"].append(duration)
        logger.debug(f"Cache hit served in {duration:.3f}s")
    
    def record_total_request_time(self, duration: float):
        """Record the total request processing time."""
        self.metrics["total_request_times"].append(duration)
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of performance metrics."""
        summary = {}
        
        for metric_name, times in self.metrics.items():
            if times:
                summary[metric_name] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        
        return summary
    
    def log_performance_summary(self):
        """Log a performance summary."""
        summary = self.get_performance_summary()
        
        logger.info("Performance Summary:")
        for metric_name, stats in summary.items():
            logger.info(
                f"  {metric_name}: "
                f"avg={stats['avg']:.2f}s, "
                f"min={stats['min']:.2f}s, "
                f"max={stats['max']:.2f}s, "
                f"total_requests={stats['count']}"
            )


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class PerformanceTracker:
    """Context manager for tracking performance of operations."""
    
    def __init__(self, operation_name: str, **kwargs):
        self.operation_name = operation_name
        self.kwargs = kwargs
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.operation_name == "generation":
            count = self.kwargs.get("count", 1)
            method = self.kwargs.get("method", "single")
            performance_monitor.record_generation_time(count, duration, method)
        elif self.operation_name == "batch":
            batch_size = self.kwargs.get("batch_size", 1)
            performance_monitor.record_batch_time(batch_size, duration)
        elif self.operation_name == "cache_hit":
            performance_monitor.record_cache_hit_time(duration)
        elif self.operation_name == "total_request":
            performance_monitor.record_total_request_time(duration)
        
        logger.debug(f"{self.operation_name} completed in {duration:.3f}s") 