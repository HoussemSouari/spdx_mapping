"""
ML-based license detector (this project).
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.base_detector import BaseLicenseDetector
from src.train import load_model
from src.evaluate import predict_license


class MLDetector(BaseLicenseDetector):
    """ML classifier from this project."""
    
    def __init__(self, model_path: Path):
        super().__init__("ML Classifier (This Project)")
        self.model_path = model_path
        self.model = None
        
    def setup(self) -> bool:
        """Load the trained model."""
        try:
            if not self.model_path.exists():
                self.setup_error = f"Model not found at {self.model_path}"
                return False
            
            self.model = load_model(str(self.model_path))
            self.is_available = True
            return True
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license using ML model."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        return predict_license(self.model, text)
    
    def detect_batch(self, texts: list, show_progress: bool = True) -> list:
        """Optimized batch prediction."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        # Use model's batch prediction for efficiency
        from src.preprocessor import preprocess_text
        
        if show_progress:
            print(f"  Preprocessing {len(texts)} samples...")
            preprocessed = [preprocess_text(text) for text in tqdm(texts, desc="  Preprocessing", unit="sample")]
            print(f"  Running ML predictions...")
            predictions = self.model.predict(preprocessed)
        else:
            preprocessed = [preprocess_text(text) for text in texts]
            predictions = self.model.predict(preprocessed)
        
        return predictions.tolist()
