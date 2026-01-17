"""
Random baseline detector (for comparison floor).
"""

from benchmarks.base_detector import BaseLicenseDetector
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RandomDetector(BaseLicenseDetector):
    """Random guess baseline (worst case scenario)."""
    
    def __init__(self):
        super().__init__("Random Guess")
        self.license_ids = []
        
    def setup(self) -> bool:
        """Load list of possible licenses."""
        try:
            from src.data_loader import load_dataset
            
            df = load_dataset('data/scancode_licenses', min_samples_per_class=10)
            self.license_ids = df['spdx_id'].unique().tolist()
            
            self.is_available = True
            return True
            
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Randomly guess a license."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        return random.choice(self.license_ids)
