# FOSS License Classification System

An automated system for classifying Free and Open Source Software (FOSS) licenses using machine learning. Given a license text file, the system predicts its SPDX license identifier.

## Project Overview

This project implements a text classification pipeline using:
- **TF-IDF Vectorization** (word-level with unigrams and bigrams)
- **Linear Support Vector Machine (LinearSVC)** classifier

The model is trained on the [ScanCode License Dataset](https://github.com/nexB/scancode-toolkit/tree/develop/src/licensedcode/data/licenses), which contains hundreds of open source license texts with their corresponding SPDX identifiers.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
# Clone ScanCode toolkit (shallow clone for speed)
git clone --depth 1 https://github.com/nexB/scancode-toolkit.git temp_scancode

# Copy only the licenses data
mkdir -p data/scancode_licenses
cp -r temp_scancode/src/licensedcode/data/licenses data/scancode_licenses/

# Clean up
rm -rf temp_scancode
```

### 3. Run the Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the license dataset
2. Split data into train/test sets (80/20, stratified)
3. Train the TF-IDF + LinearSVC model
4. Evaluate and print metrics (accuracy, precision, recall, F1)
5. Generate a confusion matrix for the top 10 licenses
6. Save the trained model to `outputs/`

## Project Structure

```
projet_ml/
├── data/
│   └── scancode_licenses/
│       └── licenses/           # License dataset (.LICENSE + .yml files)
├── outputs/
│   ├── license_classifier.pkl  # Trained model
│   ├── metrics.txt             # Evaluation metrics
│   └── confusion_matrix.png    # Visualization
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── preprocessor.py         # Text preprocessing
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Evaluation and visualization
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
└── README.md
```

## Usage Examples

### Train with custom settings

```bash
# Use different test split ratio
python main.py --test-size 0.3

# Show confusion matrix for top 15 licenses
python main.py --top-n 15

# Skip plotting (headless server)
python main.py --no-plot
```

### Use the trained model

```python
from src.train import load_model
from src.evaluate import predict_license

# Load the model
model = load_model('outputs/license_classifier.pkl')

# Predict a license
license_text = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
"""

predicted_spdx = predict_license(model, license_text)
print(f"Predicted license: {predicted_spdx}")  # Output: MIT
```

## Text Preprocessing

The preprocessing pipeline:
1. Converts text to lowercase
2. Removes punctuation
3. Normalizes whitespace

### Why No Stemming/Lemmatization?

We intentionally avoid stemming and lemmatization for legal texts because:

1. **Legal Precision**: Terms like "licensor" vs "licensee" have distinct legal meanings that would be lost if both were reduced to "licens"

2. **Defined Terms**: Licenses define specific terms (e.g., "Derivative Works", "Contribution") that must remain unchanged

3. **Standard Phrases**: Exact wording matters for identification (e.g., "AS IS" appears in many licenses)

4. **Version Sensitivity**: Minor word differences distinguish license versions (GPL-2.0 vs GPL-3.0)

5. **Copyright Holders**: Names and entity references should remain intact

TF-IDF with full words provides sufficient discrimination while preserving legal precision.

---

## Baseline Comparison: Rule-Based vs ML Approaches

### Rule-Based Tools (ScanCode, FOSSology)

Traditional license detection tools like **ScanCode** and **FOSSology** use rule-based approaches:

#### ScanCode Approach
- Maintains a database of license templates and rules
- Uses exact text matching and pattern matching
- Identifies licenses by matching against known license text patterns
- Handles license detection through:
  - Full text matching
  - Key phrase detection
  - Template matching with variable regions (e.g., copyright holder names)

#### FOSSology Approach
- Uses multiple detection agents (Nomos, Monk, etc.)
- Nomos: Regular expression-based detection
- Monk: Compares file content against license reference texts
- Combines results from multiple agents for higher accuracy

### Advantages of Rule-Based Tools
- **High precision** for known, standard license texts
- **Interpretable**: Can point to exact matching phrases
- **No training required**: Works out-of-the-box
- **Deterministic**: Same input always produces same output

### Limitations of Rule-Based Tools
- **Brittle to variations**: Minor text changes may cause misses
- **Manual maintenance**: New licenses require manual rule creation
- **Limited generalization**: Cannot handle novel paraphrasing
- **Complex rule management**: Rules can conflict or overlap

### Why ML Models Handle Variants Better

Machine learning approaches like our TF-IDF + LinearSVC model offer complementary strengths:

1. **Robust to Variations**
   - TF-IDF captures semantic similarity, not just exact matches
   - Can recognize licenses even with formatting changes, typos, or minor modifications

2. **Automatic Feature Learning**
   - The model learns discriminative features from data
   - No need to manually specify matching rules

3. **Handles Noise**
   - Training on diverse examples builds tolerance to:
     - Extra whitespace or line breaks
     - Missing sections
     - Added copyright notices
     - Formatting differences

4. **Scalable Classification**
   - Adding new license classes only requires training data
   - No need to write new detection rules

5. **Confidence Scores**
   - ML models can provide prediction confidence
   - Useful for flagging uncertain classifications for review

### Best Practice: Hybrid Approach

In production systems, the best results come from combining both approaches:
- Use ML for initial classification and handling variants
- Use rule-based verification for high-confidence licenses
- Flag edge cases for human review

This project demonstrates the ML component, which excels at handling the natural variation found in real-world license texts.

---

## Evaluation Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |

Both **macro** (unweighted average across classes) and **weighted** (weighted by class frequency) averages are reported.

## License

This project is for educational purposes. The ScanCode license dataset is from the [ScanCode Toolkit](https://github.com/nexB/scancode-toolkit) project.
