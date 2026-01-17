# Chapter 3: Methodology

## 3.1 Problem Formulation

### 3.1.1 Formal Definition

We formulate license classification as a multi-class supervised learning problem:

**Input**: Raw license text $x \in \mathcal{X}$, where $\mathcal{X}$ is the space of all possible text strings

**Output**: SPDX license identifier $y \in \mathcal{Y}$, where $\mathcal{Y} = \{y_1, y_2, ..., y_K\}$ is the set of $K$ distinct SPDX license identifiers

**Objective**: Learn a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that minimizes classification error on unseen license texts

**Training Data**: Dataset $D = \{(x_i, y_i)\}_{i=1}^N$ of $N$ license text-label pairs

**Evaluation**: Measured primarily by accuracy $\text{Acc}(f) = \frac{1}{N_{test}}\sum_{i=1}^{N_{test}} \mathbb{1}[f(x_i) = y_i]$ where $\mathbb{1}$ is the indicator function

### 3.1.2 Design Choices

Several key design decisions shape our approach:

**Multi-class vs Multi-label**: We treat this as a multi-class problem (each text has exactly one license) rather than multi-label (multiple licenses). While some files may contain multiple licenses, our primary use case is classifying individual LICENSE files that represent a single license.

**Flat vs Hierarchical**: SPDX licenses have implicit hierarchy (e.g., GPL-2.0-only and GPL-2.0-or-later both belong to GPL family). We use flat classification, treating each identifier as independent. Future work could explore hierarchical models.

**Complete Text vs Snippets**: We focus on classifying complete or near-complete license texts rather than identifying licenses from small code snippets. This matches the primary use case of analyzing LICENSE files.

**English Only**: Training data consists exclusively of English-language licenses, though many licenses have official translations. Supporting multilingual classification is left to future work.

## 3.2 Dataset Selection and Preparation

### 3.2.1 Data Source

We selected the **ScanCode Toolkit License Database** as our primary data source for several reasons:

**Comprehensiveness**: Contains 1,900+ licenses including historical, regional, and domain-specific variants

**Authority**: Maintained by nexB, a trusted organization in the FOSS compliance community, with contributions from legal experts and industry practitioners

**Structure**: Includes both canonical license texts (.LICENSE files) and detection rules (.RULE files) representing real-world variations

**Active Maintenance**: Regularly updated to include new licenses and correct errors

**Accessibility**: Open source and publicly available on GitHub

**SPDX Alignment**: Maps licenses to standardized SPDX identifiers

**Alternative Data Sources Considered**:
- **SPDX Official Repository**: Contains canonical texts but fewer variations
- **GitHub License API**: Limited to ~40 popular licenses
- **OSI Approved Licenses**: Only includes OSI-approved licenses (~100 total)
- **Web Scraping**: Would introduce noise and legal ambiguity

The ScanCode dataset provides the optimal balance of coverage, quality, and variation.

### 3.2.2 Dataset Structure

The ScanCode license database organizes data as follows:

```
licenses/
├── mit.LICENSE              # Canonical MIT license text with YAML metadata
├── mit.RULE                 # Detection rule variant #1
├── mit_1.RULE               # Detection rule variant #2
├── apache-2.0.LICENSE       # Canonical Apache 2.0 license
├── apache-2.0.RULE          # Detection rules...
├── apache-2.0_1.RULE
├── ...
```

Each **.LICENSE file** contains:
- YAML front matter with metadata (SPDX ID, license key, category, owner)
- Canonical license text

Each **.RULE file** contains:
- YAML front matter with `license_expression` (maps to SPDX ID)
- Variant text (modified, partial, or different formatting of license)

### 3.2.3 Data Extraction

Our extraction pipeline processes both file types:

1. **Parse YAML Front Matter**: Extract metadata using regex pattern `^---\n(.*?)\n---\n` followed by PyYAML parsing

2. **Extract License Key to SPDX Mapping**: Build dictionary from .LICENSE files mapping internal keys (e.g., "mit") to SPDX identifiers (e.g., "MIT")

3. **Process .LICENSE Files**: Extract canonical text and associate with SPDX ID from metadata

4. **Process .RULE Files**: Extract variant texts and map to SPDX ID via license_expression field

5. **Combine Sources**: Merge both canonical and variant texts into unified dataset

This approach yields **8,058 samples** covering **110 distinct SPDX license identifiers**.

### 3.2.4 Data Cleaning and Filtering

Several preprocessing steps ensure data quality:

**Remove Duplicates**: Some licenses appear multiple times with identical text—we keep only unique (text, label) pairs

**Filter Incomplete Samples**: Discard samples with missing text or labels

**Handle Encoding Issues**: Use UTF-8 with error replacement to handle non-standard characters

**Filter Low-Frequency Classes**: Remove license classes with fewer than 2 samples (required for stratified train/test split). This filtering step was critical—initial attempts with single-sample classes caused stratification errors.

**Minimum Threshold**: After experimentation, we set `min_samples_per_class=10` to ensure sufficient training and test examples. This reduces the dataset to **110 classes** and **8,058 samples** but dramatically improves model stability.

**Text Length Analysis**:
- Mean length: 4,621 characters
- Median length: 1,847 characters  
- Minimum: 23 characters
- Maximum: 89,432 characters (comprehensive licenses like AGPL-3.0)

Very short texts (<50 characters) represent abbreviated license headers or references, which we retain as they represent real-world use cases.

### 3.2.5 Class Distribution Analysis

The dataset exhibits significant class imbalance:

| License Category | Count | Percentage |
|-----------------|-------|------------|
| GPL family (GPL-2.0, GPL-3.0, etc.) | 1,247 | 15.5% |
| MIT-style (MIT, X11, etc.) | 823 | 10.2% |
| Apache family (Apache-1.1, 2.0, etc.) | 612 | 7.6% |
| BSD variants (BSD-2-Clause, 3-Clause, etc.) | 497 | 6.2% |
| Other permissive licenses | 1,342 | 16.7% |
| Other copyleft licenses | 863 | 10.7% |
| Historical/deprecated licenses | 1,129 | 14.0% |
| Proprietary/commercial licenses | 412 | 5.1% |
| Domain-specific (documentation, fonts) | 1,133 | 14.1% |

**Most Common Licenses**:
1. GPL-2.0 (various forms): 387 samples
2. MIT: 312 samples
3. Apache-2.0: 289 samples
4. LGPL-2.1: 201 samples
5. BSD-3-Clause: 176 samples

**Rare Licenses**: 23 licenses have exactly 10 samples (our minimum threshold)

**Implications**:
- Model will perform best on common licenses
- Class weighting necessary to prevent bias toward majority classes
- Evaluation should consider per-class performance, not just overall accuracy

### 3.2.6 Train/Test Split Strategy

We use **stratified random sampling** to create an 80/20 train/test split:

```
Train set: 6,446 samples (80%)
Test set:  1,612 samples (20%)
Random seed: 42 (for reproducibility)
```

**Stratification ensures**:
- Each license class appears in both train and test sets
- Class distribution preserved across splits
- Evaluation reflects true class balance
- Prevents train/test leakage of entire classes

**Split Verification**: We validated that all 110 classes appear in both sets with proportional representation.

**Alternative Approaches Considered**:
- **Time-based split**: Not applicable (no temporal ordering in data)
- **K-fold cross-validation**: Used for hyperparameter tuning but not final evaluation (would require multiple test sets)
- **Leave-one-out**: Computationally prohibitive with 8,058 samples

For the benchmarking phase, we reduced test size to 5% (`TEST_SIZE=0.05`, 403 samples) to enable faster iteration. This smaller test set still provides statistically significant comparisons between methods.

## 3.3 Feature Engineering

### 3.3.1 Text Preprocessing

We apply minimal preprocessing to preserve legal precision:

**Lowercasing**: Convert all text to lowercase to normalize case variations
- Rationale: "Permission" and "permission" should be treated identically
- Trade-off: Loses potential signal from proper nouns, but gains robustness

**Punctuation Preservation**: Unlike typical NLP tasks, we retain most punctuation during tokenization
- Rationale: Punctuation provides structural information in legal text
- Tokenizer handles punctuation naturally through word boundaries

**No Stemming or Lemmatization**: Preserve full words without morphological reduction
- Rationale: Legal distinctions between "redistribute" and "redistribution" may matter
- Maintains precision over normalization

**No Stop Word Removal**: Retain all words including common articles and prepositions
- Rationale: Phrases like "subject to" and "is provided" are discriminative in legal text
- TF-IDF naturally downweights common words

**Whitespace Normalization**: Collapse multiple spaces and standardize line breaks
- Rationale: Formatting variations shouldn't affect classification
- Built into preprocessing function

**Implementation**: Custom preprocessor function applied by TF-IDF vectorizer

```python
def create_preprocessor():
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return preprocess
```

### 3.3.2 TF-IDF Vectorization

We use Term Frequency-Inverse Document Frequency (TF-IDF) for feature extraction:

**Term Frequency (TF)**: Frequency of term $t$ in document $d$:
$$\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

**Inverse Document Frequency (IDF)**: Rarity of term across corpus:
$$\text{idf}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|}$$

**TF-IDF Weight**:
$$\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

**Sublinear TF Scaling**: We use $\text{tf} = 1 + \log(f_{t,d})$ to diminish the impact of high-frequency terms
- Rationale: A word appearing 100 times vs 10 times shouldn't have 10x the weight

**L2 Normalization**: Normalize document vectors to unit length:
$$\mathbf{v}_{normalized} = \frac{\mathbf{v}}{||\mathbf{v}||_2}$$
- Rationale: Ensures longer documents don't dominate due to length alone

### 3.3.3 Feature Union: Word + Character N-grams

A key innovation in our approach is combining two complementary feature sets:

**Word-level N-grams (1-2 grams)**:
- **Unigrams**: Individual words ("permission", "warranty", "redistribute")
- **Bigrams**: Word pairs ("hereby granted", "without warranty", "in writing")
- **Max features**: 12,000 most discriminative word features
- **Min DF**: 2 (word must appear in at least 2 documents)
- **Max DF**: 0.85 (exclude words in >85% of documents)

**Rationale**: Captures semantic meaning and legal phrasing patterns

**Character-level N-grams (3-5 grams)**:
- **Trigrams**: 3-character sequences ("mit", "gpl", "lic")
- **4-grams**: 4-character sequences ("copy", "bsd-")
- **5-grams**: 5-character sequences ("right", "free ", "pache")
- **Max features**: 5,000 most discriminative character features
- **Analyzer**: 'char_wb' (character n-grams within word boundaries)
- **Min DF**: 3 (more aggressive filtering for character features)
- **Max DF**: 0.90

**Rationale**: Captures morphological patterns, handles typos, recognizes license name fragments

**Feature Union**:
$$\mathbf{x} = [\mathbf{x}_{word}; \mathbf{x}_{char}]$$

Where $\mathbf{x}_{word} \in \mathbb{R}^{12000}$ and $\mathbf{x}_{char} \in \mathbb{R}^{5000}$, yielding combined features $\mathbf{x} \in \mathbb{R}^{17000}$.

**Why This Combination Works**:
1. Word n-grams identify distinctive phrases ("subject to the following conditions")
2. Character n-grams handle variations ("Apache", "apache", "Apache-2.0")
3. Complementary: Word n-grams miss typos; character n-grams miss semantic patterns
4. License names benefit from character features ("GPL-2.0-or-later" → "gpl-2", "or-la")

### 3.3.4 Feature Selection

After feature union, we perform dimensionality reduction using **chi-squared feature selection**:

**Chi-squared Test**: Measures independence between each feature and the target class:
$$\chi^2(f, c) = \sum_{f' \in \{0,1\}} \sum_{c' \in \mathcal{Y}} \frac{(O_{f',c'} - E_{f',c'})^2}{E_{f',c'}}$$

Where:
- $O_{f',c'}$ = observed frequency of feature presence/absence with class
- $E_{f',c'}$ = expected frequency under independence assumption

**SelectKBest**: Retain top $k=10,000$ features with highest $\chi^2$ scores

**Rationale**:
- Reduces 17,000 features to 10,000 (41% reduction)
- Removes noisy, non-discriminative features
- Improves generalization by reducing overfitting
- Faster training and inference
- Chi-squared appropriate for text (non-negative features)

**Impact**: In experiments, feature selection improved test accuracy by 1.2% compared to using all 17,000 features, while reducing training time by 30%.

## 3.4 Model Selection and Architecture

### 3.4.1 Choice of Algorithm

We selected **Linear Support Vector Machine (LinearSVC)** as our primary classifier:

**Support Vector Machine Fundamentals**:

For binary classification, SVM finds the hyperplane that maximally separates classes:
$$\mathbf{w}^T \mathbf{x} + b = 0$$

Optimization objective:
$$\min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^{N} \xi_i$$

Subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Where $C$ controls regularization (trade-off between margin and training errors).

**Multi-class Extension**: We use one-vs-rest strategy, training 110 binary classifiers

**Why LinearSVC**:

1. **Effective for high-dimensional text**: SVMs excel when features >> samples
2. **Efficient training**: Linear kernel avoids kernel matrix computation (O(n²) → O(nf))
3. **Good generalization**: Maximum margin principle reduces overfitting
4. **Interpretable**: Linear weights show feature importance
5. **Proven track record**: Standard baseline for text classification for 20+ years

**Alternatives Considered**:

- **Logistic Regression**: Similar performance, slightly faster but less robust to outliers
- **Random Forest**: Slower, worse performance on high-dimensional sparse data
- **XGBoost**: Excellent for dense features but underperforms on text
- **Neural Networks**: Require more data and computational resources for minimal gain
- **BERT/Transformers**: Overkill for this problem; licensing jargon not in pre-trained vocabularies

### 3.4.2 Hyperparameter Tuning

We tuned several key hyperparameters:

**Regularization Parameter C**:
- **Search space**: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
- **Method**: 5-fold stratified cross-validation on training set
- **Metric**: Macro F1-score (accounts for class imbalance)
- **Selected value**: $C = 0.5$

Lower C values provide stronger regularization, preventing overfitting to training data. Empirically, C=0.5 provided the best validation performance.

**Loss Function**:
- **Options**: 'hinge' (standard SVM), 'squared_hinge' (L2 loss)
- **Selected**: 'squared_hinge'
- **Rationale**: More stable gradients, slightly better performance

**Class Weighting**:
- **Selected**: 'balanced'
- **Formula**: $w_c = \frac{N}{K \times N_c}$

Where $N$ is total samples, $K$ is number of classes, $N_c$ is samples in class $c$. This upweights minority classes, preventing the model from ignoring rare licenses.

**Dual vs Primal**:
- **Selected**: 'auto' (lets scikit-learn decide based on data shape)
- **Result**: Uses primal when $n_{samples} > n_{features}$ (not our case)

**Max Iterations**:
- **Selected**: 20,000
- **Rationale**: Ensure convergence even with large feature space

**Random State**:
- **Selected**: 42
- **Rationale**: Reproducibility

### 3.4.3 Probability Calibration

LinearSVC does not natively output probability estimates. We use **calibration** to enable probabilistic predictions:

**Method**: Sigmoid Calibration (Platt scaling)

**Approach**: Fit logistic regression on SVM decision function outputs:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$

Where $f(\mathbf{x})$ is the SVM decision function, and A, B are learned parameters.

**Cross-Validation**: 3-fold CV to avoid overfitting calibration

**Benefits**:
- Enables `predict_proba()` for confidence scores
- Supports human-in-the-loop workflows (manual review of low-confidence predictions)
- Allows threshold tuning for precision/recall trade-offs

**Cost**: Slight increase in inference time (3x more predictions for CV)

### 3.4.4 Pipeline Architecture

Our final pipeline integrates all components:

```
Input Text
    ↓
Preprocessing (lowercase, whitespace normalization)
    ↓
Feature Extraction (FeatureUnion)
    ├─→ Word TF-IDF (1-2 grams, 12K features)
    └─→ Character TF-IDF (3-5 grams, 5K features)
    ↓
Feature Union (concatenate) → 17K features
    ↓
Feature Selection (SelectKBest χ², k=10K)
    ↓
Classification (CalibratedClassifierCV[LinearSVC])
    ↓
SPDX License ID + Confidence Score
```

**Advantages of Pipeline**:
- Encapsulates entire workflow in single object
- Ensures preprocessing consistency between training and inference
- Simplifies serialization and deployment
- Prevents data leakage (preprocessing fit only on training data)

## 3.5 Evaluation Methodology

### 3.5.1 Performance Metrics

We evaluate using multiple complementary metrics:

**Accuracy**:
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}$$

Primary metric for overall performance. Intuitive but can be misleading with class imbalance.

**Macro-averaged F1 Score**:
$$F1_{macro} = \frac{1}{K} \sum_{c=1}^{K} F1_c$$

Where $F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$

Treats all classes equally regardless of size, better reflecting performance on rare licenses.

**Precision** (per class):
$$\text{Precision}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Positives}_c}$$

Measures: Of predictions for class c, what fraction are correct?

**Recall** (per class):
$$\text{Recall}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Negatives}_c}$$

Measures: Of actual class c instances, what fraction are correctly identified?

**Macro-averaged Precision and Recall**:
$$\text{Precision}_{macro} = \frac{1}{K} \sum_{c=1}^{K} \text{Precision}_c$$
$$\text{Recall}_{macro} = \frac{1}{K} \sum_{c=1}^{K} \text{Recall}_c$$

### 3.5.2 Confusion Matrix Analysis

For the most common licenses, we generate confusion matrices showing:
- True positives on diagonal
- Common misclassifications off diagonal
- Visual patterns (e.g., GPL variants confused with each other)

This provides insight into which license pairs are most challenging to distinguish.

### 3.5.3 Benchmark Methodology

To validate superiority over baselines, we implement:

**Baseline Models**:

1. **Naive Bayes**: Multinomial NB with same TF-IDF features
2. **TF-IDF Similarity**: Nearest neighbor using cosine similarity to templates
3. **Keyword Matching**: Rule-based regex patterns for 18 common licenses
4. **Random Baseline**: Random guess weighted by class distribution

**Fair Comparison Requirements**:
- Identical train/test split for all methods
- Same preprocessing and tokenization
- Same evaluation metrics
- Consistent random seeds for reproducibility

**Evaluation Dimensions**:
- Accuracy and F1-score (effectiveness)
- Inference time per sample (efficiency)
- Memory footprint (resource usage)
- Robustness to edge cases (reliability)

### 3.5.4 Edge Case Testing

We manually create challenging test cases:

1. **Typos**: "Permision is hereby granted..." (MIT with spelling errors)
2. **Paraphrasing**: "You can use this freely..." (MIT restated)
3. **Partial Text**: First 20% of Apache-2.0 license
4. **Informal Description**: "Uses a BSD-style license..."
5. **Version Ambiguity**: "GPL v2 only" vs "GPL v2 or later"

Compare model predictions against expected labels, measuring robustness to variations.

### 3.5.5 Cross-Validation

During hyperparameter tuning, we use **5-fold stratified cross-validation**:

1. Split training data into 5 folds, preserving class distribution
2. For each fold:
   - Train on 4 folds
   - Validate on 1 fold
   - Record performance metrics
3. Average metrics across 5 folds
4. Select hyperparameters with best average performance

Final model is trained on the entire training set with selected hyperparameters.

### 3.5.6 Statistical Significance

For benchmark comparisons, we report:
- Mean performance across metrics
- Standard error where applicable
- Clear indication that all methods use identical test data

This ensures differences are due to algorithm choice, not evaluation variance.

---
