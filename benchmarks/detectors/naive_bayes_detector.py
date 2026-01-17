"""
Naive Bayes baseline detector.
"""

from benchmarks.base_detector import BaseLicenseDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class NaiveBayesDetector(BaseLicenseDetector):
    """Simple Naive Bayes classifier baseline."""
    
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = None
        
    def setup(self) -> bool:
        """Train a simple Naive Bayes model."""
        try:
            from src.data_loader import load_dataset
            from src.preprocessor import preprocess_text
            
            # Load and preprocess data
            df = load_dataset('data/scancode_licenses', min_samples_per_class=10)
            texts = [preprocess_text(text) for text in df['text']]
            labels = df['spdx_id']
            
            # Create simple pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', MultinomialNB(alpha=0.1))
            ])
            
            # Train on full dataset (for baseline comparison)
            self.model.fit(texts, labels)
            
            self.is_available = True
            return True
            
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license using Naive Bayes."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        from src.preprocessor import preprocess_text
        
        try:
            preprocessed = preprocess_text(text)
            prediction = self.model.predict([preprocessed])[0]
            return prediction
        except Exception as e:
            return "ERROR"
    
    def detect_batch(self, texts: list, show_progress: bool = True) -> list:
        """Optimized batch prediction."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        from src.preprocessor import preprocess_text
        from tqdm import tqdm
        
        if show_progress:
            print(f"  Preprocessing {len(texts)} samples...")
            preprocessed = [preprocess_text(text) for text in tqdm(texts, desc="  Preprocessing", unit="sample")]
            print(f"  Running Naive Bayes predictions...")
            predictions = self.model.predict(preprocessed)
        else:
            preprocessed = [preprocess_text(text) for text in texts]
            predictions = self.model.predict(preprocessed)
        
        return predictions.tolist()
