# FOSS License Classification System

## Complete Documentation & User Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Installation Guide](#4-installation-guide)
5. [Dataset](#5-dataset)
6. [Model Pipeline](#6-model-pipeline)
7. [Usage Guide](#7-usage-guide)
8. [Evaluation Results](#8-evaluation-results)
9. [Project Structure](#9-project-structure)
10. [Technical Details](#10-technical-details)
11. [Baseline Comparison](#11-baseline-comparison)
12. [Troubleshooting](#12-troubleshooting)
13. [Future Improvements](#13-future-improvements)

---

## 1. Project Overview

This project implements an **automated license classification system** that predicts the SPDX license identifier from raw license text. Given any license file (e.g., `LICENSE`, `LICENSE.txt`, `COPYING`), the system identifies which open-source license it represents.

### Key Features

- ✅ **82.69% accuracy** on 110 license classes
- ✅ Uses **TF-IDF vectorization** with word and character n-grams
- ✅ **LinearSVC classifier** with calibration for probability estimates
- ✅ Trained on **8,058 samples** from the ScanCode dataset
- ✅ No deep learning required - runs on any machine
- ✅ Fully reproducible pipeline

---

## 2. Problem Statement

### The Challenge

When organizations use open-source software, they must:
1. **Identify all licenses** in their codebase
2. **Ensure compliance** with license terms
3. **Avoid legal risks** from incompatible licenses

### Why It's Difficult

| Problem | Description |
|---------|-------------|
| **Volume** | Modern projects have hundreds of dependencies |
| **Variations** | Same license may have different formatting, typos, or modifications |
| **Legal Precision** | Small differences can have legal implications |
| **Manual Effort** | Human review is slow and error-prone |

### Our Solution

An ML-based classifier that:
- Takes raw license text as input
- Outputs the standardized **SPDX identifier** (e.g., `MIT`, `Apache-2.0`, `GPL-3.0-only`)
- Handles variations in formatting and wording
- Processes thousands of files in seconds

---

## 3. Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│              Raw License Text File                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                 │
│  • Lowercase conversion                                          │
│  • Punctuation removal                                           │
│  • Whitespace normalization                                      │
│  • NO stemming (preserves legal precision)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION                              │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │  Word TF-IDF        │    │  Char TF-IDF        │             │
│  │  (1-2 grams)        │ +  │  (3-5 grams)        │             │
│  │  12,000 features    │    │  5,000 features     │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                              │                                   │
│                    Feature Union                                 │
│                              │                                   │
│                    SelectKBest (χ²)                              │
│                    → 10,000 features                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION                                │
│                                                                  │
│              Calibrated LinearSVC                                │
│              • C = 0.5 (regularization)                          │
│              • Balanced class weights                            │
│              • 3-fold CV calibration                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│              SPDX License Identifier                             │
│              (e.g., "MIT", "Apache-2.0")                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Installation Guide

### Prerequisites

- Python 3.10+
- pip or conda
- Git
- ~500 MB disk space for dataset

### Step 1: Clone or Navigate to Project

```bash
cd /path/to/projet_ml
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- PyYAML >= 6.0
- matplotlib >= 3.6.0

### Step 4: Download Dataset

```bash
# Clone ScanCode toolkit (shallow clone)
git clone --depth 1 https://github.com/nexB/scancode-toolkit.git temp_scancode

# Copy license data
mkdir -p data/scancode_licenses
cp -r temp_scancode/src/licensedcode/data/licenses data/scancode_licenses/
cp -r temp_scancode/src/licensedcode/data/rules data/scancode_licenses/

# Clean up
rm -rf temp_scancode
```

### Step 5: Verify Installation

```bash
python -c "from src.data_loader import load_dataset; print('✓ Installation successful')"
```

---

## 5. Dataset

### Source

**ScanCode License Dataset**  
https://github.com/nexB/scancode-toolkit/tree/develop/src/licensedcode/data

### Structure

```
data/scancode_licenses/
├── licenses/           # 2,615 .LICENSE files with YAML front matter
│   ├── mit.LICENSE
│   ├── apache-2.0.LICENSE
│   ├── gpl-3.0.LICENSE
│   └── ...
└── rules/              # 36,472 .RULE files with license text variants
    ├── mit_1.RULE
    ├── mit_2.RULE
    ├── apache-2.0_1.RULE
    └── ...
```

### File Format

**.LICENSE files** contain YAML front matter with metadata:
```yaml
---
key: mit
short_name: MIT License
spdx_license_key: MIT
category: Permissive
---

Permission is hereby granted, free of charge...
```

**.RULE files** contain text variants:
```yaml
---
license_expression: mit
relevance: 100
---

Permission is hereby granted, free of charge, to any person...
```

### Data Filtering

| Filter | Threshold | Reason |
|--------|-----------|--------|
| Min text length | 150 chars | Remove short snippets |
| Min samples/class | 10 | Ensure reliable training |

### Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 8,058 |
| License classes | 110 |
| Train set | 6,446 (80%) |
| Test set | 1,612 (20%) |

---

## 6. Model Pipeline

### 6.1 Text Preprocessing

```python
def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation (replace with space)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

**Why no stemming/lemmatization?**

Legal texts require precise terminology:
- "licensor" ≠ "licensee" (different legal entities)
- "derivative" vs "derived" may have legal implications
- "AS IS" is a standard legal phrase that must be preserved

### 6.2 Feature Extraction

**Word TF-IDF:**
- N-gram range: (1, 2) - unigrams and bigrams
- Max features: 12,000
- Captures semantic meaning

**Character TF-IDF:**
- N-gram range: (3, 5) - character sequences
- Max features: 5,000
- Captures morphological patterns

**Feature Selection:**
- SelectKBest with chi-squared test
- Keeps top 10,000 features
- Removes noisy/irrelevant features

### 6.3 Classifier

**Calibrated LinearSVC:**
- Linear SVM with squared hinge loss
- C = 0.5 (regularization strength)
- Balanced class weights
- Calibrated with 3-fold CV for probability estimates

---

## 7. Usage Guide

### 7.1 Train the Model

```bash
python main.py
```

**Options:**
```bash
python main.py --help

Options:
  --data-dir PATH      Dataset directory (default: data/scancode_licenses)
  --output-dir PATH    Output directory (default: outputs)
  --test-size FLOAT    Test set fraction (default: 0.2)
  --top-n INT          Top N licenses for confusion matrix (default: 10)
  --no-plot            Skip plotting confusion matrix
```

### 7.2 Predict a License

**Python API:**
```python
from src.train import load_model
from src.evaluate import predict_license

# Load trained model
model = load_model('outputs/license_classifier.pkl')

# Read license file
with open('path/to/LICENSE', 'r') as f:
    license_text = f.read()

# Predict
spdx_id = predict_license(model, license_text)
print(f"Detected license: {spdx_id}")
```

**Example:**
```python
license_text = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED...
"""

result = predict_license(model, license_text)
print(result)  # Output: MIT
```

### 7.3 Batch Processing

```python
import os
from src.train import load_model
from src.evaluate import predict_license

model = load_model('outputs/license_classifier.pkl')

# Process all LICENSE files in a directory
results = {}
for root, dirs, files in os.walk('/path/to/project'):
    for file in files:
        if file in ['LICENSE', 'LICENSE.txt', 'COPYING']:
            filepath = os.path.join(root, file)
            with open(filepath, 'r', errors='replace') as f:
                text = f.read()
            results[filepath] = predict_license(model, text)

# Print results
for path, license_id in results.items():
    print(f"{path}: {license_id}")
```

---

## 8. Evaluation Results

### Final Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **82.69%** |
| Precision (macro) | 82.92% |
| Precision (weighted) | 82.37% |
| Recall (macro) | 78.83% |
| Recall (weighted) | 82.69% |
| F1-Score (macro) | 79.59% |
| F1-Score (weighted) | 81.86% |

### Performance Improvement Journey

| Version | Accuracy | Key Changes |
|---------|----------|-------------|
| Baseline | 63.5% | Basic TF-IDF + LinearSVC |
| v2 | 72.9% | Filtered short texts, min 5 samples/class |
| v3 | 76.0% | Tuned TF-IDF and SVC parameters |
| v4 | 79.1% | Optimized C=0.5, squared_hinge loss |
| **Final** | **82.7%** | Added char n-grams, feature selection |

### Top 10 Most Frequent Licenses

| Rank | License | Samples |
|------|---------|---------|
| 1 | GPL-2.0-or-later | 761 |
| 2 | GPL-2.0-only | 734 |
| 3 | proprietary-license | 722 |
| 4 | BSD-3-Clause | 713 |
| 5 | Apache-2.0 | 413 |
| 6 | other-permissive | 334 |
| 7 | GPL-3.0-or-later | 330 |
| 8 | LGPL-2.1-or-later | 290 |
| 9 | LGPL-2.0-or-later | 221 |
| 10 | MIT | 203 |

---

## 9. Project Structure

```
projet_ml/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Quick start guide
├── DOCUMENTATION.md           # This file
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading & preprocessing
│   ├── preprocessor.py        # Text preprocessing functions
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation & visualization
│   ├── tune_model.py          # Hyperparameter tuning (optional)
│   └── ensemble.py            # Ensemble models (experimental)
│
├── data/                      # Dataset (not in git)
│   └── scancode_licenses/
│       ├── licenses/          # .LICENSE files
│       └── rules/             # .RULE files
│
└── outputs/                   # Model & results
    ├── license_classifier.pkl # Trained model
    ├── metrics.txt            # Evaluation metrics
    └── confusion_matrix.png   # Visualization
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `data_loader.py` | Loads ScanCode dataset, parses YAML front matter, filters data |
| `preprocessor.py` | Text normalization (lowercase, punctuation, whitespace) |
| `train.py` | Creates ML pipeline, trains model, saves/loads models |
| `evaluate.py` | Computes metrics, plots confusion matrix, prediction API |
| `tune_model.py` | Grid search cross-validation for hyperparameters |

---

## 10. Technical Details

### 10.1 TF-IDF Vectorization

**Term Frequency (TF):**
$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}$$

**Inverse Document Frequency (IDF):**
$$IDF(t) = \log\frac{N}{1 + \text{docs containing } t}$$

**TF-IDF:**
$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

With `sublinear_tf=True`, we use:
$$TF(t, d) = 1 + \log(count)$$

### 10.2 LinearSVC

Solves the optimization problem:
$$\min_{w, b} \frac{1}{2}||w||^2 + C \sum_i \max(0, 1 - y_i(w^T x_i + b))^2$$

Where:
- $w$ = weight vector
- $b$ = bias
- $C$ = regularization parameter (0.5)
- Squared hinge loss for smoother optimization

### 10.3 Calibration

CalibratedClassifierCV applies Platt scaling:
$$P(y=1|x) = \frac{1}{1 + \exp(Af(x) + B)}$$

Where $f(x)$ is the SVM decision function, and $A$, $B$ are learned from held-out data.

### 10.4 Chi-Squared Feature Selection

Selects features with highest chi-squared statistics:
$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where $O$ = observed frequency, $E$ = expected frequency under independence.

---

## 11. Baseline Comparison

### Rule-Based Tools

**ScanCode Toolkit:**
- Maintains database of license templates
- Uses exact text matching and regex patterns
- Handles variable regions (copyright holders, dates)

**FOSSology:**
- Multiple detection agents (Nomos, Monk)
- Regular expression-based detection
- Compares against reference texts

### Comparison

| Aspect | Rule-Based | ML (This Project) |
|--------|------------|-------------------|
| **Known licenses** | ✅ High precision | ✅ High accuracy |
| **Format variations** | ❌ May fail | ✅ Robust |
| **Typos/errors** | ❌ Fails | ✅ Tolerant |
| **Novel licenses** | ❌ No match | ⚠️ Closest match |
| **Maintenance** | Manual rules | Retrain with data |
| **Interpretability** | ✅ Exact matches | ⚠️ Feature weights |
| **Speed** | ✅ Very fast | ✅ Fast |

### Best Practice

Combine both approaches:
1. Use ML for initial classification
2. Verify with rule-based matching for high-confidence cases
3. Flag uncertain predictions for human review

---

## 12. Troubleshooting

### Common Issues

**Issue: "Dataset directory not found"**
```
Solution: Download the dataset (see Installation Guide Step 4)
```

**Issue: "No valid licenses found"**
```
Solution: Ensure data/scancode_licenses/licenses/ contains .LICENSE files
```

**Issue: Model training is slow**
```
Solution: 
- Reduce max_features in TfidfVectorizer
- Use create_simple_pipeline() instead of create_pipeline()
- Reduce dataset size for testing
```

**Issue: "Can't pickle local object"**
```
Solution: Ensure preprocess_text is defined at module level, not nested
```

**Issue: Low accuracy on specific licenses**
```
Solution:
- Check if the license has enough training samples
- Look at confusion matrix for commonly confused pairs
- Consider adding more training data
```

### Memory Issues

For large datasets, reduce memory usage:
```python
# Use sparse matrices (already default)
# Reduce max_features
vectorizer = TfidfVectorizer(max_features=5000)

# Use incremental learning (SGDClassifier)
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge')
```

---

## 13. Future Improvements

### Potential Enhancements

| Improvement | Expected Impact | Effort |
|-------------|-----------------|--------|
| Ensemble methods | +2-5% accuracy | Medium |
| Data augmentation | +2-4% accuracy | Medium |
| Hierarchical classification | Better for similar licenses | High |
| License family grouping | Reduced confusion | Medium |
| Active learning | Better with less data | High |

### Recommended Next Steps

1. **Add more training data** - Collect real-world license variants
2. **License family features** - Add metadata about license categories
3. **Confidence thresholds** - Flag low-confidence predictions
4. **Integration** - Build CLI tool or REST API

### Example: Adding Confidence Threshold

```python
from src.train import load_model
import numpy as np

model = load_model('outputs/license_classifier.pkl')

def predict_with_confidence(model, text, threshold=0.7):
    """Predict license with confidence check."""
    proba = model.predict_proba([text])[0]
    max_proba = np.max(proba)
    predicted_idx = np.argmax(proba)
    predicted_label = model.classes_[predicted_idx]
    
    if max_proba < threshold:
        return predicted_label, max_proba, "LOW_CONFIDENCE"
    return predicted_label, max_proba, "OK"

# Usage
license, confidence, status = predict_with_confidence(model, text)
print(f"{license} ({confidence:.1%}) - {status}")
```

---

## References

1. **ScanCode Toolkit**: https://github.com/nexB/scancode-toolkit
2. **SPDX License List**: https://spdx.org/licenses/
3. **Scikit-learn Documentation**: https://scikit-learn.org/
4. **TF-IDF Explanation**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

---

## License

This project is for educational purposes. The ScanCode license dataset is from the [ScanCode Toolkit](https://github.com/nexB/scancode-toolkit) project (Apache-2.0 licensed).

---

*Documentation generated for FOSS License Classification Project*  
*Last updated: January 2026*
