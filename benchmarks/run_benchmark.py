"""
Main benchmarking runner.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json
import time
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import (
    TOOLS,
    DATASETS,
    METRICS,
    TEST_SIZE,
    RANDOM_SEED,
    BENCHMARK_OUTPUT_DIR
)
from benchmarks.detectors import (
    MLDetector,
    ScanCodeDetector,
    ScanCodeAPIDetector,
    KeywordDetector,
    TFIDFSimilarityDetector,
    NaiveBayesDetector,
    RandomDetector
)
from benchmarks.dataset_loaders import (
    ScanCodeDatasetLoader,
    SPDXSamplesLoader,
    RealWorldDatasetLoader
)


class BenchmarkRunner:
    """Main class for running benchmarks."""
    
    def __init__(self):
        self.detectors = {}
        self.datasets = {}
        self.results = []
        
    def setup_detectors(self):
        """Initialize all enabled detectors."""
        print("Setting up detectors...")
        
        for tool_id, tool_config in TOOLS.items():
            if not tool_config["enabled"]:
                print(f"  ⊘ Skipping {tool_config['name']} (disabled)")
                continue
            
            print(f"  → Setting up {tool_config['name']}...")
            
            try:
                if tool_id == "ml_classifier":
                    detector = MLDetector(tool_config["model_path"])
                elif tool_id == "scancode":
                    detector = ScanCodeDetector()
                elif tool_id == "scancode_api":
                    detector = ScanCodeAPIDetector()
                elif tool_id == "keyword_matching":
                    detector = KeywordDetector()
                elif tool_id == "tfidf_similarity":
                    detector = TFIDFSimilarityDetector()
                elif tool_id == "naive_bayes":
                    detector = NaiveBayesDetector()
                elif tool_id == "random_guess":
                    detector = RandomDetector()
                else:
                    print(f"    ⚠ Detector '{tool_id}' not implemented yet")
                    continue
                
                if detector.setup():
                    self.detectors[tool_id] = detector
                    print(f"    ✓ {detector.name} ready")
                else:
                    print(f"    ✗ Failed: {detector.setup_error}")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print(f"\n{len(self.detectors)} detector(s) ready\n")
        
    def setup_datasets(self):
        """Initialize all enabled datasets."""
        print("Setting up datasets...")
        
        for dataset_id, dataset_config in DATASETS.items():
            if not dataset_config["enabled"]:
                print(f"  ⊘ Skipping {dataset_config['name']} (disabled)")
                continue
            
            print(f"  → Loading {dataset_config['name']}...")
            
            try:
                if dataset_id == "scancode":
                    loader = ScanCodeDatasetLoader(dataset_config["path"])
                elif dataset_id == "spdx_samples":
                    loader = SPDXSamplesLoader(dataset_config["path"])
                elif dataset_id == "real_world":
                    loader = RealWorldDatasetLoader(dataset_config["path"])
                else:
                    print(f"    ⚠ Dataset loader '{dataset_id}' not implemented yet")
                    continue
                
                # Try to load dataset
                texts, labels = loader.load()
                self.datasets[dataset_id] = {
                    "loader": loader,
                    "texts": texts,
                    "labels": labels
                }
                print(f"    ✓ Loaded {len(texts)} samples, {len(set(labels))} classes")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print(f"\n{len(self.datasets)} dataset(s) ready\n")
    
    def run_benchmark(self, detector_id: str, dataset_id: str):
        """Run benchmark for a specific detector-dataset pair."""
        print(f"Benchmarking {detector_id} on {dataset_id}...")
        
        detector = self.detectors[detector_id]
        dataset = self.datasets[dataset_id]
        
        texts = dataset["texts"]
        y_true = dataset["labels"]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_true,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_true
        )
        
        print(f"  Test set: {len(X_test)} samples")
        
        # Run predictions with timing
        start_time = time.time()
        try:
            benchmark_result = detector.benchmark_batch(X_test)
            y_pred = benchmark_result["predictions"]
            execution_time = benchmark_result["execution_time"]
            memory_delta = benchmark_result["memory_delta"]
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            return None
        
        # Calculate metrics
        try:
            metrics = {
                "detector": detector_id,
                "dataset": dataset_id,
                "test_samples": len(X_test),
                "num_classes": len(set(y_test)),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "execution_time": execution_time,
                "avg_time_per_sample": execution_time / len(X_test),
                "memory_delta_mb": memory_delta
            }
            
            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ✓ F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  ✓ Time: {execution_time:.2f}s ({metrics['avg_time_per_sample']*1000:.1f}ms/sample)")
            
            return metrics
            
        except Exception as e:
            print(f"  ✗ Metric calculation failed: {e}")
            return None
    
    def run_all_benchmarks(self):
        """Run all enabled detector-dataset combinations."""
        print("="*80)
        print("STARTING BENCHMARKS")
        print("="*80 + "\n")
        
        self.setup_detectors()
        self.setup_datasets()
        
        if not self.detectors:
            print("⚠ No detectors available. Exiting.")
            return
        
        if not self.datasets:
            print("⚠ No datasets available. Exiting.")
            return
        
        print("="*80)
        print("RUNNING BENCHMARKS")
        print("="*80 + "\n")
        
        for detector_id in self.detectors.keys():
            for dataset_id in self.datasets.keys():
                result = self.run_benchmark(detector_id, dataset_id)
                if result:
                    self.results.append(result)
                print()
        
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save benchmark results to files."""
        if not self.results:
            print("No results to save.")
            return
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = BENCHMARK_OUTPUT_DIR / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved results to {csv_path}")
        
        # Save as JSON
        json_path = BENCHMARK_OUTPUT_DIR / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved results to {json_path}")
    
    def generate_report(self):
        """Generate human-readable report."""
        if not self.results:
            print("No results to report.")
            return
        
        df = pd.DataFrame(self.results)
        
        report_path = BENCHMARK_OUTPUT_DIR / "benchmark_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LICENSE DETECTION BENCHMARK REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary table
            f.write("RESULTS SUMMARY\n")
            f.write("-"*80 + "\n")
            
            summary_cols = [
                "detector", "dataset", "accuracy", "f1_macro", 
                "execution_time", "avg_time_per_sample"
            ]
            f.write(df[summary_cols].to_string(index=False))
            f.write("\n\n")
            
            # Best performance
            f.write("BEST PERFORMANCE BY METRIC\n")
            f.write("-"*80 + "\n")
            
            for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
                best_idx = df[metric].idxmax()
                best_row = df.loc[best_idx]
                f.write(f"{metric.upper()}: {best_row['detector']} on {best_row['dataset']} = {best_row[metric]:.4f}\n")
            
            f.write("\n")
            
            # Speed comparison
            f.write("SPEED COMPARISON\n")
            f.write("-"*80 + "\n")
            speed_sorted = df.sort_values("avg_time_per_sample")
            for _, row in speed_sorted.iterrows():
                f.write(f"{row['detector']:30s} | {row['avg_time_per_sample']*1000:6.1f} ms/sample\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved report to {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(df[["detector", "dataset", "accuracy", "f1_macro"]].to_string(index=False))
        print("="*80 + "\n")


def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
