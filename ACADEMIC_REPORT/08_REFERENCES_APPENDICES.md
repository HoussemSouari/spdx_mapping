# References

## Academic Publications

Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial decisions of the European Court of Human Rights: A natural language processing perspective. *PeerJ Computer Science*, 2, e93.

Chalkidis, I., Androutsopoulos, I., & Michos, A. (2017). Extracting contract elements. In *Proceedings of the 16th International Conference on Artificial Intelligence and Law* (pp. 19-28).

Di Penta, M., German, D. M., Guéhéneuc, Y. G., & Antoniol, G. (2010). An exploratory study of the evolution of software licensing. In *Proceedings of the 32nd ACM/IEEE International Conference on Software Engineering* (pp. 145-154).

German, D. M., Di Penta, M., Guéhéneuc, Y. G., & Antoniol, G. (2009). Code siblings: Technical and legal implications of copying code between applications. In *Proceedings of the 6th IEEE Working Conference on Mining Software Repositories* (pp. 81-90).

Kim, M. Y., Xu, Y., & Goebel, R. (2019). Legal question answering using ranking SVM and deep convolutional neural network. In *Proceedings of the COLIEE Workshop at ICAIL*.

Palmirani, M., Martoni, M., Rossi, A., Bartolini, C., & Robaldo, L. (2018). Legal ontology for modelling GDPR concepts and norms. In *Legal Knowledge and Information Systems* (pp. 91-100).

Vendome, C., Bavota, G., Di Penta, M., Linares-Vásquez, M., German, D., & Poshyvanyk, D. (2017). License usage and changes: A large-scale study on GitHub. *Empirical Software Engineering*, 22(3), 1537-1577.

## Technical Standards and Specifications

Linux Foundation. (2021). *Software Package Data Exchange (SPDX) Specification Version 2.2*. Retrieved from https://spdx.dev/specifications/

ISO/IEC. (2021). *ISO/IEC 5962:2021 - SPDX Specification V2.2.1*. International Organization for Standardization.

Open Source Initiative. (2024). *The Open Source Definition*. Retrieved from https://opensource.org/osd

## Software Tools and Frameworks

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

nexB. (2024). *ScanCode Toolkit*. GitHub repository. Retrieved from https://github.com/nexB/scancode-toolkit

McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

## License Resources

Free Software Foundation. (2024). *GNU General Public License*. Retrieved from https://www.gnu.org/licenses/

Open Source Initiative. (2024). *Licenses & Standards*. Retrieved from https://opensource.org/licenses

SPDX Legal Team. (2024). *SPDX License List*. Retrieved from https://spdx.org/licenses/

GitHub. (2024). *Licensee - A Ruby Gem to detect under what license a project is distributed*. GitHub repository. Retrieved from https://github.com/licensee/licensee

Amazon. (2024). *Askalono - A library and command-line tool to help detect license texts*. GitHub repository. Retrieved from https://github.com/amzn/askalono

## Machine Learning and NLP Resources

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed. draft). Retrieved from https://web.stanford.edu/~jurafsky/slp3/

Aggarwal, C. C., & Zhai, C. (Eds.). (2012). *Mining Text Data*. Springer Science & Business Media.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

## Web Resources and Documentation

scikit-learn developers. (2024). *scikit-learn: Machine Learning in Python*. Retrieved from https://scikit-learn.org/stable/

SPDX Workgroup. (2024). *SPDX Documentation*. Retrieved from https://spdx.dev/

Software Freedom Conservancy. (2024). *Compliance Resources*. Retrieved from https://sfconservancy.org/compliance/

Linux Foundation. (2024). *OpenChain Project*. Retrieved from https://www.openchainproject.org/

---

# Appendices

## Appendix A: Complete Benchmark Results

### A.1 Full Performance Table

| Detector | Dataset | Accuracy | F1-Macro | Precision-Macro | Recall-Macro | Time (s) | Time/Sample (ms) |
|----------|---------|----------|----------|----------------|-------------|----------|------------------|
| ml_classifier | scancode | 0.7940 | 0.6894 | 0.6854 | 0.7106 | 0.806 | 2.00 |
| naive_bayes | scancode | 0.6526 | 0.5264 | 0.5482 | 0.5209 | 0.162 | 0.40 |
| tfidf_similarity | scancode | 0.2481 | 0.2981 | 0.3124 | 0.3053 | 0.815 | 2.02 |
| keyword_matching | scancode | 0.2084 | 0.0547 | 0.1892 | 0.1124 | 0.209 | 0.52 |
| random_guess | scancode | 0.0074 | 0.0032 | 0.0048 | 0.0037 | 0.001 | 0.00 |

### A.2 Per-License Performance (Top 30)

*Detailed per-class metrics available in supplementary materials*

## Appendix B: Configuration Files

### B.1 Python Requirements

```
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1
PyYAML==6.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.65.0
psutil==5.9.5
requests==2.31.0
```

### B.2 Hyperparameter Configuration

```python
HYPERPARAMETERS = {
    'tfidf_word': {
        'ngram_range': (1, 2),
        'max_features': 12000,
        'sublinear_tf': True,
        'min_df': 2,
        'max_df': 0.85,
        'norm': 'l2'
    },
    'tfidf_char': {
        'analyzer': 'char_wb',
        'ngram_range': (3, 5),
        'max_features': 5000,
        'sublinear_tf': True,
        'min_df': 3,
        'max_df': 0.90,
        'norm': 'l2'
    },
    'feature_selection': {
        'score_func': 'chi2',
        'k': 10000
    },
    'classifier': {
        'C': 0.5,
        'max_iter': 20000,
        'class_weight': 'balanced',
        'random_state': 42,
        'loss': 'squared_hinge'
    },
    'calibration': {
        'cv': 3,
        'method': 'sigmoid'
    }
}
```

### B.3 Dataset Configuration

```python
DATASET_CONFIG = {
    'min_samples_per_class': 10,
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'data_dir': 'data/scancode_licenses'
}
```

## Appendix C: Edge Case Examples

### C.1 Typo Example (MIT)

```
Input Text:
"Permision is hereby granted, free of charge, to any person obtainning 
a copy of this sofware and associated documentation files (the "Software"), 
to deal in the Software without restricton, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND."

Expected: MIT
ML Prediction: MIT (confidence: 0.87)
Keyword Prediction: MIT (confidence: 1.0)
TF-IDF Prediction: BSD-2-Clause (confidence: 0.62)
```

### C.2 Paraphrased Example (MIT)

```
Input Text:
"This software may be used freely at no cost. Anyone can modify and 
redistribute this code. The authors provide no guarantees or warranties. 
All liability is disclaimed. You must include this notice in copies."

Expected: MIT
ML Prediction: MIT (confidence: 0.72)
Keyword Prediction: UNKNOWN
TF-IDF Prediction: Apache-2.0 (confidence: 0.45)
```

### C.3 Partial Text Example (Apache-2.0)

```
Input Text:
"Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0"

Expected: Apache-2.0
ML Prediction: Apache-1.1 (confidence: 0.58)
Keyword Prediction: Apache-2.0 (confidence: 1.0)
TF-IDF Prediction: LGPL-2.1 (confidence: 0.51)
```

### C.4 Informal Description (BSD-3-Clause)

```
Input Text:
"# License

This project uses a BSD-style license. You can use, modify, and 
redistribute the code as long as you keep the copyright notice.
See the LICENSE file for full details."

Expected: BSD-3-Clause
ML Prediction: BSD-3-Clause (confidence: 0.65)
Keyword Prediction: UNKNOWN
TF-IDF Prediction: MIT (confidence: 0.48)
```

### C.5 Version Ambiguity (GPL-2.0-only)

```
Input Text:
"GNU General Public License version 2.0

This software is licensed under GPL v2. You may redistribute and modify
this program under the terms of the GNU General Public License as published
by the Free Software Foundation, version 2 of the License only."

Expected: GPL-2.0-only
ML Prediction: GPL-2.0-or-later (confidence: 0.69)
Keyword Prediction: GPL-2.0-or-later (confidence: 1.0)
TF-IDF Prediction: GPL-3.0-only (confidence: 0.54)
```

## Appendix D: Implementation Code Samples

### D.1 Preprocessing Function

```python
import re

def create_preprocessor():
    """Create preprocessing function for license text."""
    def preprocess(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    return preprocess
```

### D.2 Pipeline Construction

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def create_pipeline():
    preprocessor = create_preprocessor()
    
    feature_extraction = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            preprocessor=preprocessor,
            analyzer='word',
            ngram_range=(1, 2),
            max_features=12000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
            norm='l2',
        )),
        ('char_tfidf', TfidfVectorizer(
            preprocessor=preprocessor,
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=5000,
            sublinear_tf=True,
            min_df=3,
            max_df=0.90,
            norm='l2',
        )),
    ])
    
    base_classifier = LinearSVC(
        C=0.5,
        max_iter=20000,
        class_weight='balanced',
        random_state=42,
        dual='auto',
        loss='squared_hinge',
    )
    
    calibrated_classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        cv=3,
        method='sigmoid',
    )
    
    pipeline = Pipeline([
        ('features', feature_extraction),
        ('select_best', SelectKBest(chi2, k=10000)),
        ('classifier', calibrated_classifier),
    ])
    
    return pipeline
```

### D.3 Prediction Interface

```python
def predict_license(pipeline, license_text: str):
    """
    Predict license from text.
    
    Args:
        pipeline: Trained classification pipeline
        license_text: Raw license text
        
    Returns:
        (spdx_id, confidence) tuple
    """
    prediction = pipeline.predict([license_text])[0]
    probabilities = pipeline.predict_proba([license_text])[0]
    confidence = probabilities.max()
    
    return prediction, confidence
```

## Appendix E: Dataset Statistics

### E.1 License Distribution (Full Dataset)

| License | Train Samples | Test Samples | Total | Percentage |
|---------|--------------|-------------|-------|------------|
| GPL-2.0-only | 309 | 78 | 387 | 4.80% |
| MIT | 249 | 63 | 312 | 3.87% |
| Apache-2.0 | 231 | 58 | 289 | 3.59% |
| LGPL-2.1 | 161 | 40 | 201 | 2.49% |
| BSD-3-Clause | 140 | 36 | 176 | 2.18% |
| GPL-3.0-only | 187 | 47 | 234 | 2.90% |
| LGPL-3.0 | 89 | 22 | 111 | 1.38% |
| BSD-2-Clause | 112 | 28 | 140 | 1.74% |
| MPL-2.0 | 67 | 17 | 84 | 1.04% |
| Apache-1.1 | 58 | 15 | 73 | 0.91% |
| ... (100 more licenses) | ... | ... | ... | ... |
| **Total** | **6,446** | **1,612** | **8,058** | **100%** |

### E.2 Text Length Distribution

| Statistic | Value |
|-----------|-------|
| Mean Length | 4,621 characters |
| Median Length | 1,847 characters |
| Std Deviation | 8,234 characters |
| Min Length | 23 characters |
| Max Length | 89,432 characters |
| 25th Percentile | 712 characters |
| 75th Percentile | 5,183 characters |

### E.3 License Family Distribution

| Family | Count | Percentage |
|--------|-------|------------|
| GPL (all versions) | 1,247 | 15.5% |
| MIT-style | 823 | 10.2% |
| Apache | 612 | 7.6% |
| BSD | 497 | 6.2% |
| LGPL | 456 | 5.7% |
| Creative Commons | 389 | 4.8% |
| Mozilla Public License | 234 | 2.9% |
| Other Copyleft | 627 | 7.8% |
| Other Permissive | 1,342 | 16.7% |
| Proprietary/Commercial | 412 | 5.1% |
| Historical/Deprecated | 1,129 | 14.0% |
| Domain-Specific | 290 | 3.6% |

## Appendix F: Confusion Matrix (Top 10 Licenses)

*See visualization in outputs/confusion_matrix.png*

Notable confusion patterns:
- GPL-2.0-only ↔ GPL-2.0-or-later: 8 confusions
- BSD-2-Clause ↔ BSD-3-Clause: 5 confusions
- Apache-1.1 ↔ Apache-2.0: 4 confusions
- LGPL-2.1 ↔ LGPL-3.0: 3 confusions
- MIT ↔ ISC: 2 confusions

## Appendix G: Benchmark Framework Usage

### G.1 Running Benchmarks

```bash
# Run full benchmark
python benchmarks/run_benchmark.py

# Generate visualizations
python benchmarks/visualize.py

# Check configuration
python -c "from benchmarks.config import TOOLS, DATASETS; print(TOOLS); print(DATASETS)"
```

### G.2 Adding New Detector

```python
from benchmarks.base_detector import BaseLicenseDetector

class MyDetector(BaseLicenseDetector):
    def __init__(self):
        super().__init__("my_detector")
    
    def setup(self):
        # Initialize your detector
        pass
    
    def detect(self, license_text: str) -> str:
        # Implement detection logic
        return predicted_spdx_id
```

### G.3 Adding New Dataset

```python
from benchmarks.dataset_loaders import BaseDatasetLoader

class MyDatasetLoader(BaseDatasetLoader):
    def load(self):
        # Load your dataset
        return texts, labels
```

## Appendix H: Glossary

**SPDX**: Software Package Data Exchange - standardized format for license information

**TF-IDF**: Term Frequency-Inverse Document Frequency - text vectorization method

**SVM**: Support Vector Machine - maximum margin classifier

**LinearSVC**: Linear Support Vector Classification - efficient SVM for large datasets

**Macro-averaging**: Averaging metric across all classes equally, regardless of class size

**Micro-averaging**: Averaging metric weighted by class size (equivalent to accuracy for classification)

**Calibration**: Technique to convert classifier outputs into reliable probability estimates

**Stratified Split**: Train/test split preserving class distribution

**Feature Union**: Combining multiple feature extractors (e.g., word and character n-grams)

**Chi-squared Test**: Statistical test measuring independence between features and target

**Copyleft**: License requiring derivatives maintain same license (e.g., GPL)

**Permissive License**: License allowing proprietary derivatives (e.g., MIT, BSD)

**License Family**: Group of related licenses (e.g., GPL-2.0, GPL-3.0 are in GPL family)

**SBOM**: Software Bill of Materials - inventory of software components and licenses

---

**End of Report**

---
