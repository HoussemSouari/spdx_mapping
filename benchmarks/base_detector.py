"""
Base class for license detection tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import time
import psutil
import os
from tqdm import tqdm


class BaseLicenseDetector(ABC):
    """Abstract base class for all license detection tools."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_available = False
        self.setup_error = None
        
    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the detector (load models, check dependencies, etc.).
        Returns True if setup successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def detect(self, text: str) -> str:
        """
        Detect license from text.
        Returns SPDX license identifier.
        """
        pass
    
    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Detect licenses for multiple texts.
        Default implementation calls detect() for each text.
        Override for optimized batch processing.
        """
        if show_progress:
            results = []
            for text in tqdm(texts, desc=f"{self.name}", unit="sample"):
                results.append(self.detect(text))
            return results
        else:
            return [self.detect(text) for text in texts]
    
    def benchmark_single(self, text: str) -> Dict:
        """
        Benchmark single detection with timing and memory metrics.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = self.detect(text)
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
    def benchmark_batch(self, texts: List[str], show_progress: bool = True) -> Dict:
        """
        Benchmark batch detection with timing and memory metrics.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        results = self.detect_batch(texts, show_progress=show_progress)
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "predictions": results,
            "execution_time": end_time - start_time,
            "memory_delta": mem_after - mem_before,
            "avg_time_per_sample": (end_time - start_time) / len(texts)
        }
        return {
            "predictions": results,
            "execution_time": end_time - start_time,
            "memory_delta": mem_after - mem_before,
            "avg_time_per_sample": (end_time - start_time) / len(texts)
        }
    
    def __str__(self):
        return f"{self.name} (available={self.is_available})"
