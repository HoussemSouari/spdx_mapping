"""
TF-IDF similarity-based license detector (cosine similarity baseline).
"""

from benchmarks.base_detector import BaseLicenseDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TFIDFSimilarityDetector(BaseLicenseDetector):
    """Detect licenses using TF-IDF cosine similarity against templates."""
    
    def __init__(self):
        super().__init__("TF-IDF Similarity")
        self.vectorizer = None
        self.templates = {}
        self.template_vectors = None
        self.license_ids = []
        
    def setup(self) -> bool:
        """Load license templates and create TF-IDF vectors."""
        try:
            from src.data_loader import load_dataset
            
            # Load dataset to get license templates
            df = load_dataset('data/scancode_licenses', min_samples_per_class=10)
            
            # For each license, use the first sample as template
            templates = {}
            for spdx_id in df['spdx_id'].unique():
                license_df = df[df['spdx_id'] == spdx_id]
                # Use the longest text as template (likely most complete)
                idx = license_df['text'].str.len().idxmax()
                templates[spdx_id] = license_df.loc[idx, 'text']
            
            self.templates = templates
            self.license_ids = list(templates.keys())
            template_texts = [templates[lid] for lid in self.license_ids]
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1
            )
            
            # Fit and transform templates
            self.template_vectors = self.vectorizer.fit_transform(template_texts)
            
            self.is_available = True
            return True
            
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license using cosine similarity."""
        if not self.is_available:
            raise RuntimeError(f"Detector not available: {self.setup_error}")
        
        try:
            # Transform input text
            text_vector = self.vectorizer.transform([text])
            
            # Calculate cosine similarity with all templates
            similarities = cosine_similarity(text_vector, self.template_vectors)[0]
            
            # Return license with highest similarity
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # Threshold: require at least 10% similarity
            if best_similarity < 0.1:
                return "UNKNOWN"
            
            return self.license_ids[best_idx]
            
        except Exception as e:
            return "ERROR"
