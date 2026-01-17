# Benchmarking Guide

## Complete Guide to Benchmarking License Detection Tools

This guide explains how to use the benchmarking framework to compare different license detection tools and datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Available Tools](#available-tools)
4. [Available Datasets](#available-datasets)
5. [Running Benchmarks](#running-benchmarks)
6. [Interpreting Results](#interpreting-results)
7. [Adding New Tools](#adding-new-tools)
8. [Adding New Datasets](#adding-new-datasets)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The benchmarking framework allows you to:

- **Compare multiple tools**: ML classifier vs rule-based tools (ScanCode, Askalono, etc.)
- **Test on multiple datasets**: ScanCode, SPDX samples, GitHub licenses, real-world files
- **Measure performance**: Accuracy, precision, recall, F1-score, speed, memory usage
- **Visualize results**: Generate comparison charts and reports

### Architecture

```
benchmarks/
├── config.py              # Configuration (enable/disable tools & datasets)
├── base_detector.py       # Abstract base class for detectors
├── detectors/             # Tool implementations
│   ├── ml_detector.py     # This project's ML classifier
│   └── scancode_detector.py  # ScanCode wrapper
├── dataset_loaders.py     # Dataset loaders
├── run_benchmark.py       # Main benchmark runner
├── visualize.py           # Visualization utilities
└── download_datasets.py   # Dataset download helper
```

---

## Quick Start

### Step 1: Install Dependencies

```bash
# Install additional dependencies
pip install scancode-toolkit psutil requests

# Or add to requirements.txt
echo "scancode-toolkit>=32.0.0" >> requirements.txt
echo "psutil>=5.9.0" >> requirements.txt
pip install -r requirements.txt
```

### Step 2: Download Additional Datasets (Optional)

```bash
python benchmarks/download_datasets.py
```

This downloads:
- SPDX official license samples
- GitHub common licenses
- Creates template for real-world dataset

### Step 3: Configure Benchmarks

Edit `benchmarks/config.py` to enable/disable tools and datasets:

```python
TOOLS = {
    "ml_classifier": {
        "enabled": True,  # Your ML model
        # ...
    },
    "scancode": {
        "enabled": True,  # ScanCode Toolkit
        # ...
    }
}

DATASETS = {
    "scancode": {
        "enabled": True,  # Your training dataset
        # ...
    },
    "spdx_samples": {
        "enabled": True,  # Enable after downloading
        # ...
    }
}
```

### Step 4: Run Benchmarks

```bash
python benchmarks/run_benchmark.py
```

### Step 5: Generate Visualizations

```bash
python benchmarks/visualize.py
```

Results will be saved in `benchmark_results/`.

---

## Available Tools

### 1. ML Classifier (This Project)

**Type**: Machine Learning  
**Model**: TF-IDF + LinearSVC  
**Status**: ✅ Enabled by default

**Pros:**
- Fast inference
- Handles text variations well
- 82.69% accuracy on training data

**Cons:**
- Requires trained model
- May not generalize to very different text formats

**Configuration:**
```python
"ml_classifier": {
    "enabled": True,
    "model_path": "outputs/license_classifier.pkl"
}
```

### 2. ScanCode Toolkit

**Type**: Rule-based  
**Vendor**: nexB  
**Status**: ✅ Implemented

**Pros:**
- Industry standard
- High precision on known licenses
- Handles template matching

**Cons:**
- Slower than ML
- May fail on heavily modified texts

**Installation:**
```bash
pip install scancode-toolkit
```

**Configuration:**
```python
"scancode": {
    "enabled": True,
    "command": "scancode"
}
```

### 3. Askalono (Optional)

**Type**: Rule-based  
**Vendor**: Amazon  
**Status**: ⚠️ Requires Rust

**Installation:**
```bash
# Install Rust first
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Askalono
cargo install askalono-cli
```

**Configuration:**
```python
"askalono": {
    "enabled": True,
    "command": "askalono"
}
```

### 4. Licensee (Optional)

**Type**: Rule-based  
**Vendor**: GitHub  
**Status**: ⚠️ Requires Ruby

**Installation:**
```bash
gem install licensee
```

---

## Available Datasets

### 1. ScanCode License Dataset

**Source**: nexB/scancode-toolkit  
**Size**: 8,058 samples, 110 classes  
**Status**: ✅ Already downloaded

**Description**: Comprehensive dataset with .LICENSE and .RULE files.

**Path**: `data/scancode_licenses/`

### 2. SPDX License Samples

**Source**: spdx/license-list-data  
**Size**: ~500 licenses  
**Status**: ⚠️ Download required

**Description**: Official SPDX license texts.

**Download:**
```bash
python benchmarks/download_datasets.py
```

**Configuration:**
```python
"spdx_samples": {
    "enabled": True,
    "path": "data/spdx_samples"
}
```

### 3. GitHub Common Licenses

**Source**: GitHub API  
**Size**: ~20 popular licenses  
**Status**: ⚠️ Download required

**Description**: Most commonly used licenses on GitHub.

**Download:**
```bash
python benchmarks/download_datasets.py
```

### 4. Real-World Projects

**Source**: Manual collection  
**Size**: Custom  
**Status**: 🔧 Requires manual setup

**Setup:**

1. Collect LICENSE files from real projects:
```bash
mkdir -p data/real_world_licenses/projects

# Example: Clone and extract LICENSE
git clone https://github.com/tensorflow/tensorflow.git temp_tf
cp temp_tf/LICENSE data/real_world_licenses/projects/tensorflow_LICENSE
rm -rf temp_tf
```

2. Create `data/real_world_licenses/labels.csv`:
```csv
file_path,license_id
projects/tensorflow_LICENSE,Apache-2.0
projects/react_LICENSE,MIT
projects/linux_COPYING,GPL-2.0-only
```

3. Enable in config:
```python
"real_world": {
    "enabled": True,
    "path": "data/real_world_licenses"
}
```

---

## Running Benchmarks

### Basic Usage

```bash
python benchmarks/run_benchmark.py
```

### Output

The script will:

1. **Setup Phase**: Load models and datasets
```
Setting up detectors...
  → Setting up ML Classifier...
    ✓ ML Classifier (This Project) ready
  → Setting up ScanCode Toolkit...
    ✓ ScanCode Toolkit ready

2 detector(s) ready
```

2. **Benchmark Phase**: Test each tool on each dataset
```
Benchmarking ml_classifier on scancode...
  Test set: 1,612 samples
  ✓ Accuracy: 0.8269
  ✓ F1 (macro): 0.7959
  ✓ Time: 2.45s (1.5ms/sample)
```

3. **Save Results**: Generate CSV, JSON, and text reports
```
✓ Saved results to benchmark_results/benchmark_results.csv
✓ Saved results to benchmark_results/benchmark_results.json
✓ Saved report to benchmark_results/benchmark_report.txt
```

### Generated Files

```
benchmark_results/
├── benchmark_results.csv       # Raw data
├── benchmark_results.json      # Raw data (JSON)
├── benchmark_report.txt        # Human-readable report
├── accuracy_comparison.png     # Accuracy bar chart
├── f1_comparison.png           # F1-score bar chart
├── speed_comparison.png        # Speed bar chart
├── precision_recall.png        # Precision vs Recall scatter
├── metrics_heatmap.png         # Heatmap of all metrics
└── radar_chart_*.png           # Radar charts per dataset
```

---

## Interpreting Results

### Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Overall correctness | % of correct predictions |
| **Precision (macro)** | Average per-class precision | How many predictions are correct |
| **Recall (macro)** | Average per-class recall | How many true licenses are found |
| **F1 (macro)** | Harmonic mean of P&R | Balanced performance measure |
| **Execution Time** | Total time for test set | Overall speed |
| **Avg Time/Sample** | Time per license | Efficiency per file |
| **Memory Delta** | Memory increase | Resource usage |

### Example Report

```
RESULTS SUMMARY
--------------------------------------------------------------------------------
detector          dataset    accuracy  f1_macro  execution_time  avg_time_per_sample
ML Classifier     scancode   0.8269    0.7959    2.45           0.0015
ScanCode          scancode   0.8543    0.8201    45.32          0.0281

BEST PERFORMANCE BY METRIC
--------------------------------------------------------------------------------
ACCURACY: ScanCode on scancode = 0.8543
F1_MACRO: ScanCode on scancode = 0.8201
PRECISION_MACRO: ScanCode on scancode = 0.8512
RECALL_MACRO: ScanCode on scancode = 0.8103

SPEED COMPARISON
--------------------------------------------------------------------------------
ML Classifier                  | 1.5 ms/sample
ScanCode                       | 28.1 ms/sample
```

### Key Insights

1. **Accuracy**: ScanCode slightly better (85.43% vs 82.69%)
2. **Speed**: ML ~19x faster (1.5ms vs 28.1ms per sample)
3. **Trade-off**: ML offers good accuracy with much better speed

---

## Adding New Tools

### Step 1: Create Detector Class

Create `benchmarks/detectors/mytool_detector.py`:

```python
from benchmarks.base_detector import BaseLicenseDetector

class MyToolDetector(BaseLicenseDetector):
    def __init__(self):
        super().__init__("My Tool Name")
        
    def setup(self) -> bool:
        """Check if tool is available."""
        try:
            # Check if tool is installed
            # Load models, etc.
            self.is_available = True
            return True
        except Exception as e:
            self.setup_error = str(e)
            return False
    
    def detect(self, text: str) -> str:
        """Detect license from text."""
        # Your implementation
        # Return SPDX license ID
        return "MIT"
```

### Step 2: Register in Config

Edit `benchmarks/config.py`:

```python
TOOLS = {
    # ... existing tools ...
    "mytool": {
        "name": "My Tool Name",
        "type": "rule_based",
        "enabled": True,
        "command": "mytool"
    }
}
```

### Step 3: Update Runner

Edit `benchmarks/run_benchmark.py`:

```python
from benchmarks.detectors import MLDetector, ScanCodeDetector, MyToolDetector

# In setup_detectors():
elif tool_id == "mytool":
    detector = MyToolDetector()
```

---

## Adding New Datasets

### Step 1: Create Loader Class

Edit `benchmarks/dataset_loaders.py`:

```python
class MyDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: Path):
        super().__init__("My Dataset Name", path)
        
    def load(self) -> Tuple[List[str], List[str]]:
        """Load dataset."""
        texts = []
        labels = []
        
        # Your loading logic
        # Read files, parse data, etc.
        
        self.is_available = True
        return texts, labels
```

### Step 2: Register in Config

```python
DATASETS = {
    # ... existing datasets ...
    "mydataset": {
        "name": "My Dataset Name",
        "path": "data/mydataset",
        "enabled": True,
        "type": "custom"
    }
}
```

### Step 3: Update Runner

```python
# In setup_datasets():
elif dataset_id == "mydataset":
    loader = MyDatasetLoader(dataset_config["path"])
```

---

## Troubleshooting

### Issue: "ScanCode not installed"

**Solution:**
```bash
pip install scancode-toolkit
```

### Issue: "Dataset directory not found"

**Solution:**
```bash
# Download datasets
python benchmarks/download_datasets.py

# Or check path in config.py
```

### Issue: "Model file not found"

**Solution:**
```bash
# Train model first
python main.py

# Or check path in config.py
TOOLS["ml_classifier"]["model_path"] = Path("outputs/license_classifier.pkl")
```

### Issue: Benchmarks are slow

**Solution:**
```bash
# Reduce test set size in config.py
TEST_SIZE = 0.1  # Use only 10% for testing

# Or disable slow tools
TOOLS["scancode"]["enabled"] = False
```

### Issue: Out of memory

**Solution:**
```python
# Process in smaller batches
# Override detect_batch() in your detector:

def detect_batch(self, texts: list) -> list:
    batch_size = 100
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results.extend([self.detect(t) for t in batch])
    return results
```

---

## Advanced Usage

### Custom Metrics

Add custom metrics in `run_benchmark.py`:

```python
# In run_benchmark():
metrics = {
    # ... existing metrics ...
    "top1_accuracy": top_k_accuracy_score(y_test, y_pred_proba, k=1),
    "top3_accuracy": top_k_accuracy_score(y_test, y_pred_proba, k=3)
}
```

### Cross-Dataset Evaluation

Train on one dataset, test on another:

```python
# Train ML model on scancode
# Test on spdx_samples
# Measures generalization
```

### Statistical Significance

Add significance testing:

```python
from scipy.stats import ttest_ind

# Compare two tools
tool1_scores = [...]
tool2_scores = [...]
t_stat, p_value = ttest_ind(tool1_scores, tool2_scores)
print(f"p-value: {p_value}")
```

---

## Best Practices

1. **Run multiple times**: Use different random seeds for robustness
2. **Check data leakage**: Ensure test set is completely separate
3. **Monitor resources**: Track memory and CPU usage
4. **Document assumptions**: Note any preprocessing or filtering
5. **Version control**: Track dataset versions and model checkpoints

---

## Example Workflows

### Workflow 1: Quick Comparison

```bash
# 1. Enable only ML and ScanCode
vim benchmarks/config.py  # Set enabled: True for both

# 2. Run benchmark
python benchmarks/run_benchmark.py

# 3. View results
cat benchmark_results/benchmark_report.txt
```

### Workflow 2: Full Evaluation

```bash
# 1. Download all datasets
python benchmarks/download_datasets.py

# 2. Enable all in config
vim benchmarks/config.py

# 3. Run benchmarks
python benchmarks/run_benchmark.py

# 4. Generate visualizations
python benchmarks/visualize.py

# 5. Open results
xdg-open benchmark_results/accuracy_comparison.png
```

### Workflow 3: Custom Dataset

```bash
# 1. Collect licenses
mkdir -p data/my_licenses
# Copy LICENSE files...

# 2. Create labels.csv
cat > data/my_licenses/labels.csv << EOF
file_path,license_id
project1_LICENSE,MIT
project2_LICENSE,Apache-2.0
EOF

# 3. Create loader (see "Adding New Datasets")

# 4. Run benchmark
python benchmarks/run_benchmark.py
```

---

## References

- [ScanCode Toolkit](https://github.com/nexB/scancode-toolkit)
- [SPDX License List](https://spdx.org/licenses/)
- [Askalono](https://github.com/jpeddicord/askalono)
- [Licensee](https://github.com/licensee/licensee)

---

*Last updated: January 2026*
