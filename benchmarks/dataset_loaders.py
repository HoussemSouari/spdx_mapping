"""
Dataset loaders for benchmarking.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.is_available = False
        
    @abstractmethod
    def load(self) -> Tuple[List[str], List[str]]:
        """
        Load dataset.
        Returns (texts, labels) where labels are SPDX identifiers.
        """
        pass
    
    def get_info(self) -> Dict:
        """Get dataset statistics."""
        if not self.is_available:
            return {"error": "Dataset not loaded"}
        
        texts, labels = self.load()
        df = pd.DataFrame({"label": labels})
        
        return {
            "name": self.name,
            "total_samples": len(texts),
            "num_classes": df["label"].nunique(),
            "class_distribution": df["label"].value_counts().to_dict(),
            "avg_text_length": sum(len(t) for t in texts) / len(texts)
        }


class ScanCodeDatasetLoader(BaseDatasetLoader):
    """Loader for ScanCode license dataset."""
    
    def __init__(self, path: Path):
        super().__init__("ScanCode License Dataset", path)
        
    def load(self) -> Tuple[List[str], List[str]]:
        """Load ScanCode dataset using existing data loader."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from src.data_loader import load_dataset
        
        # load_dataset returns a DataFrame, not (texts, labels)
        df = load_dataset(
            str(self.path),
            min_samples_per_class=10,
            min_text_length=150
        )
        
        # Extract texts and labels
        texts = df['text'].tolist()
        labels = df['spdx_id'].tolist()
        
        self.is_available = True
        return texts, labels


class SPDXSamplesLoader(BaseDatasetLoader):
    """Loader for SPDX official license samples."""
    
    def __init__(self, path: Path):
        super().__init__("SPDX License Samples", path)
        
    def load(self) -> Tuple[List[str], List[str]]:
        """Load SPDX samples from JSON files."""
        import json
        
        texts = []
        labels = []
        
        if not self.path.exists():
            raise FileNotFoundError(f"SPDX samples not found at {self.path}")
        
        # Look for JSON files
        for json_file in self.path.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract license text and ID
                    if isinstance(data, dict):
                        license_id = data.get("licenseId") or data.get("spdxId")
                        license_text = data.get("licenseText") or data.get("text")
                        
                        if license_id and license_text:
                            texts.append(license_text)
                            labels.append(license_id)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        self.is_available = True
        return texts, labels


class RealWorldDatasetLoader(BaseDatasetLoader):
    """Loader for real-world license files."""
    
    def __init__(self, path: Path):
        super().__init__("Real-World Projects", path)
        
    def load(self) -> Tuple[List[str], List[str]]:
        """Load real-world license files with manual labels."""
        import csv
        
        texts = []
        labels = []
        
        if not self.path.exists():
            raise FileNotFoundError(f"Real-world dataset not found at {self.path}")
        
        # Expect a CSV with columns: file_path, license_id
        labels_file = self.path / "labels.csv"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_file}")
        
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = self.path / row["file_path"]
                license_id = row["license_id"]
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as lf:
                        text = lf.read()
                        texts.append(text)
                        labels.append(license_id)
        
        self.is_available = True
        return texts, labels
