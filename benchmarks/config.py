"""
Benchmark configuration for license detection tools and datasets.
"""

from pathlib import Path
from typing import Dict, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Benchmark output directory
BENCHMARK_OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
BENCHMARK_OUTPUT_DIR.mkdir(exist_ok=True)

# Tools to benchmark
TOOLS = {
    "ml_classifier": {
        "name": "ML Classifier (This Project)",
        "type": "ml",
        "enabled": True,
        "model_path": PROJECT_ROOT / "outputs" / "license_classifier.pkl"
    },
    "scancode_api": {
        "name": "ScanCode API",
        "type": "rule_based",
        "enabled": True,
        "description": "ScanCode license detection via Python API"
    },
    "naive_bayes": {
        "name": "Naive Bayes Baseline",
        "type": "ml",
        "enabled": True,
        "description": "Simple Multinomial Naive Bayes classifier"
    },
    "tfidf_similarity": {
        "name": "TF-IDF Cosine Similarity",
        "type": "similarity",
        "enabled": True,
        "description": "Nearest neighbor using cosine similarity"
    },
    "keyword_matching": {
        "name": "Keyword Matching",
        "type": "rule_based",
        "enabled": True,
        "description": "Simple regex-based keyword detection"
    },
    "random_guess": {
        "name": "Random Guess",
        "type": "baseline",
        "enabled": False,  # Disable for cleaner comparisons
        "description": "Random selection (worst case baseline)"
    },
    "scancode": {
        "name": "ScanCode Toolkit",
        "type": "rule_based",
        "enabled": False,  # Not applicable for license text classification
        "command": "scancode",
        "install_cmd": "pip install scancode-toolkit"
    },
    "askalono": {
        "name": "Askalono (Amazon)",
        "type": "rule_based",
        "enabled": False,  # Requires Rust installation
        "command": "askalono",
        "install_cmd": "cargo install askalono-cli"
    },
    "licensee": {
        "name": "Licensee (GitHub)",
        "type": "rule_based",
        "enabled": False,  # Requires Ruby
        "command": "licensee",
        "install_cmd": "gem install licensee"
    },
    "licensecheck": {
        "name": "LicenseCheck",
        "type": "rule_based",
        "enabled": False,  # Requires Perl
        "command": "licensecheck",
        "install_cmd": "apt-get install licensecheck"
    }
}

# Datasets to benchmark
DATASETS = {
    "scancode": {
        "name": "ScanCode License Dataset",
        "path": PROJECT_ROOT / "data" / "scancode_licenses",
        "enabled": True,
        "type": "full_dataset"
    },
    "spdx_samples": {
        "name": "SPDX License Samples",
        "path": PROJECT_ROOT / "data" / "spdx_samples",
        "enabled": False,
        "type": "samples",
        "url": "https://github.com/spdx/license-list-data"
    },
    "github_licenses": {
        "name": "GitHub Common Licenses",
        "path": PROJECT_ROOT / "data" / "github_licenses",
        "enabled": False,
        "type": "samples",
        "url": "https://api.github.com/licenses"
    },
    "real_world": {
        "name": "Real-World Projects",
        "path": PROJECT_ROOT / "data" / "real_world_licenses",
        "enabled": False,
        "type": "real_world",
        "description": "License files collected from real open-source projects"
    }
}

# Benchmark metrics
METRICS = [
    "accuracy",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "execution_time",
    "memory_usage"
]

# Top N licenses to focus on
TOP_N_LICENSES = 20

# Random seed for reproducibility
RANDOM_SEED = 42

# Test set size
TEST_SIZE = 0.05  # Use 5% for faster benchmarking (~400 samples)

# Minimum samples per class
MIN_SAMPLES_PER_CLASS = 10
