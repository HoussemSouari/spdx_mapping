# Chapter 4: Implementation

## 4.1 System Architecture

### 4.1.1 Overview

The implementation consists of several modular components organized in a pipeline architecture:

```
projet_ml/
├── data/                      # Dataset storage
│   └── scancode_licenses/
│       ├── licenses/          # License files (.LICENSE and .RULE)
│       └── rules/             # Additional detection rules
├── src/                       # Core implementation modules
│   ├── data_loader.py         # Dataset extraction and loading
│   ├── preprocessor.py        # Text preprocessing utilities
│   ├── train.py               # Model training pipeline
│   └── evaluate.py            # Evaluation and visualization
├── benchmarks/                # Benchmarking framework
│   ├── base_detector.py       # Abstract base class
│   ├── config.py              # Configuration
│   ├── detectors/             # Tool implementations
│   ├── dataset_loaders.py     # Dataset loaders
│   ├── run_benchmark.py       # Main runner
│   └── visualize.py           # Visualization
├── outputs/                   # Trained models and results
│   ├── license_classifier.pkl # Serialized model
│   ├── metrics.txt            # Performance metrics
│   └── confusion_matrix.png   # Visualizations
├── main.py                    # Main entry point
└── requirements.txt           # Python dependencies
```

### 4.1.2 Technology Stack

**Programming Language**: Python 3.10+
- Chosen for rich ML ecosystem and widespread adoption

**Core Dependencies**:
- **scikit-learn 1.3.0**: Machine learning framework
- **pandas 2.0.3**: Data manipulation and analysis
- **numpy 1.24.3**: Numerical computations
- **scipy 1.11.1**: Sparse matrix operations
- **PyYAML 6.0**: YAML parsing for license metadata
- **matplotlib 3.7.2**: Visualization
- **seaborn 0.12.2**: Statistical visualizations

**Additional Tools**:
- **tqdm**: Progress bars for long operations
- **psutil**: System resource monitoring
- **requests**: HTTP requests for dataset downloads

**Development Environment**:
- Python virtual environment (venv)
- Git for version control
- VS Code / PyCharm for IDE

### 4.1.3 Design Patterns

**Pipeline Pattern**: The scikit-learn Pipeline encapsulates the entire workflow, ensuring consistent preprocessing and preventing data leakage.

**Factory Pattern**: `create_pipeline()` and `create_preprocessor()` functions instantiate configured objects with appropriate parameters.

**Strategy Pattern**: Different baseline detectors implement the `BaseLicenseDetector` interface, enabling polymorphic benchmarking.

**Template Method Pattern**: `BaseLicenseDetector` defines the benchmark workflow with customizable `setup()` and `detect()` methods.

## 4.2 Data Loading Module (`src/data_loader.py`)

### 4.2.1 Core Functions

**`parse_yaml_front_matter(content: str) -> Tuple[Optional[dict], str]`**

Extracts YAML metadata from license files:
1. Searches for YAML front matter delimited by `---` markers
2. Parses YAML content using PyYAML
3. Returns metadata dictionary and remaining text
4. Handles malformed YAML gracefully (returns None)

**Implementation Details**:
```python
front_matter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)
```

The regex captures everything between opening and closing `---` markers. The `re.DOTALL` flag allows `.` to match newlines.

**`build_license_key_to_spdx_map(licenses_dir: Path) -> Dict[str, str]`**

Creates mapping from internal license keys to SPDX identifiers:
1. Scans all .LICENSE files in directory
2. Extracts `key` and `spdx_license_key` from metadata
3. Builds dictionary (e.g., {"mit": "MIT", "apache-2.0": "Apache-2.0"})
4. This mapping is essential for processing .RULE files that reference licenses by key

**`parse_license_expression(expr: str) -> Optional[str]`**

Extracts simple license identifiers from expressions:
- Handles simple cases: "mit" → "mit"
- Rejects complex expressions: "mit AND apache-2.0" → None
- Uses regex to detect boolean operators (AND, OR, WITH)

**Rationale**: We focus on single-license classification. Multi-license files require different handling.

**`load_dataset(dataset_dir: str, min_samples_per_class: int = 10) -> pd.DataFrame`**

Main entry point for data loading:

1. **Phase 1**: Build license key to SPDX mapping
   ```python
   key_to_spdx = build_license_key_to_spdx_map(licenses_dir)
   ```

2. **Phase 2**: Process .LICENSE files
   - Read file content with UTF-8 encoding
   - Parse YAML front matter
   - Extract canonical license text
   - Associate with SPDX ID

3. **Phase 3**: Process .RULE files
   - Read rule variant texts
   - Extract license_expression from metadata
   - Map to SPDX ID using key_to_spdx dictionary
   - Add to dataset

4. **Phase 4**: Quality filtering
   - Remove duplicates based on (text, spdx_id) pairs
   - Filter empty texts or missing labels
   - Count samples per class
   - Remove classes with < min_samples_per_class

5. **Phase 5**: Return DataFrame
   ```python
   return pd.DataFrame({'text': texts, 'spdx_id': labels})
   ```

**Error Handling**:
- Catches IOError for unreadable files
- Handles encoding errors with 'replace' strategy
- Skips malformed YAML gracefully
- Logs warnings for skipped files (via print statements)

### 4.2.2 Dataset Statistics

After loading with `min_samples_per_class=10`:

```
Total samples: 8,058
Unique licenses: 110
Samples per class: min=10, max=387, mean=73.3, median=42
```

**Class Distribution Verification**:
```python
class_counts = df['spdx_id'].value_counts()
assert (class_counts >= min_samples_per_class).all()
```

## 4.3 Preprocessing Module (`src/preprocessor.py`)

### 4.3.1 Preprocessor Function

**`create_preprocessor() -> Callable[[str], str]`**

Returns a preprocessing function suitable for TF-IDF vectorizer:

```python
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

**Design Rationale**:

1. **Minimal Preprocessing**: Unlike typical NLP pipelines, we avoid aggressive normalization:
   - No stemming (preserve legal terminology precision)
   - No lemmatization (maintain exact word forms)
   - No stop word removal (phrases like "subject to" are important)
   - Retain punctuation (handled by tokenizer)

2. **Case Normalization**: Lowercasing is necessary because:
   - "Permission" and "permission" should match
   - Reduces vocabulary size
   - Improves generalization

3. **Whitespace Normalization**: Collapses multiple spaces/newlines to single space:
   - Different formatting shouldn't affect classification
   - Reduces noise in feature space
   - Still preserves word boundaries

### 4.3.2 Integration with TF-IDF

The preprocessor is passed to TfidfVectorizer:

```python
TfidfVectorizer(
    preprocessor=create_preprocessor(),
    analyzer='word',  # or 'char_wb' for character n-grams
    ...
)
```

This ensures preprocessing is applied consistently during both training and inference, and is included in the serialized pipeline.

## 4.4 Training Module (`src/train.py`)

### 4.4.1 Pipeline Construction

**`create_pipeline() -> Pipeline`**

Constructs the complete classification pipeline:

```python
def create_pipeline() -> Pipeline:
    preprocessor = create_preprocessor()
    
    # Feature extraction: word + character n-grams
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
    
    # Base classifier
    base_classifier = LinearSVC(
        C=0.5,
        max_iter=20000,
        class_weight='balanced',
        random_state=42,
        dual='auto',
        loss='squared_hinge',
    )
    
    # Calibration for probability estimates
    calibrated_classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        cv=3,
        method='sigmoid',
    )
    
    # Complete pipeline
    pipeline = Pipeline([
        ('features', feature_extraction),
        ('select_best', SelectKBest(chi2, k=10000)),
        ('classifier', calibrated_classifier),
    ])
    
    return pipeline
```

**Parameter Justifications**:

- **ngram_range=(1,2)**: Captures both individual words and meaningful phrases
- **max_features**: Limits vocabulary to most important terms (computational efficiency)
- **sublinear_tf=True**: Uses log scaling for term frequency (1 + log(tf))
- **min_df=2/3**: Removes extremely rare features (likely noise or typos)
- **max_df=0.85/0.90**: Removes ubiquitous terms (provide little signal)
- **C=0.5**: Regularization strength (tuned via cross-validation)
- **class_weight='balanced'**: Compensates for class imbalance
- **cv=3**: 3-fold cross-validation for calibration

### 4.4.2 Train/Test Split

**`split_dataset(df, test_size=0.2, random_state=42, min_samples_per_class=2)`**

Performs stratified splitting:

```python
def split_dataset(df, test_size=0.2, random_state=42, 
                  min_samples_per_class=2):
    # Filter classes with insufficient samples
    class_counts = df['spdx_id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_filtered = df[df['spdx_id'].isin(valid_classes)]
    
    # Stratified split
    X = df_filtered['text'].values
    y = df_filtered['spdx_id'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class distribution
    )
    
    # Convert back to DataFrames
    train_df = pd.DataFrame({'text': X_train, 'spdx_id': y_train})
    test_df = pd.DataFrame({'text': X_test, 'spdx_id': y_test})
    
    return train_df, test_df
```

**Stratification Importance**: Without stratification, rare license classes might appear only in training or only in test set, leading to:
- Biased evaluation (can't predict unseen classes)
- Unrealistic train/test similarity

### 4.4.3 Model Training

**`train_model(pipeline, X_train, y_train) -> Pipeline`**

Trains the pipeline on training data:

```python
def train_model(pipeline, X_train, y_train):
    print("Training model...")
    start_time = time.time()
    
    # Fit the entire pipeline
    pipeline.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    return pipeline
```

**Training Process**:
1. **Feature Extraction**: TF-IDF vectorizers learn vocabulary and IDF weights from training data
2. **Feature Selection**: Chi-squared test identifies top 10K discriminative features
3. **Calibration CV**: 3-fold internal cross-validation
   - For each fold: Train LinearSVC, then train calibration layer
   - Final model uses ensemble of 3 calibrated classifiers
4. **Final Training**: On full training set

**Training Time**: Typically 120-180 seconds on standard laptop (Intel i5, 16GB RAM)

### 4.4.4 Model Serialization

**`save_model(pipeline, filepath)`**

Serializes trained pipeline for deployment:

```python
def save_model(pipeline, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {filepath}")
```

**`load_model(filepath) -> Pipeline`**

Loads serialized model:

```python
def load_model(filepath):
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
```

**Serialization Considerations**:
- Uses pickle (scikit-learn standard)
- Includes entire pipeline (preprocessing, vectorization, model)
- Model file size: ~15-20 MB
- Loading time: <1 second

## 4.5 Evaluation Module (`src/evaluate.py`)

### 4.5.1 Model Evaluation

**`evaluate_model(pipeline, X_test, y_test) -> dict`**

Computes comprehensive performance metrics:

```python
def evaluate_model(pipeline, X_test, y_test):
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'classification_report': report
    }
    
    return results
```

**Metric Justifications**:
- **Accuracy**: Overall correctness, intuitive metric
- **Macro-averaged metrics**: Equal weight to all classes (important for imbalanced data)
- **zero_division=0**: Handles classes with no predictions gracefully
- **Classification report**: Detailed per-class breakdown

### 4.5.2 Confusion Matrix Visualization

**`plot_confusion_matrix(y_test, y_pred, top_n=10)`**

Generates visual confusion matrix for top N licenses:

```python
def plot_confusion_matrix(y_test, y_pred, top_n=10):
    # Identify top N most common licenses in test set
    top_licenses = pd.Series(y_test).value_counts().head(top_n).index
    
    # Filter predictions to top licenses only
    mask = pd.Series(y_test).isin(top_licenses)
    y_test_filtered = np.array(y_test)[mask]
    y_pred_filtered = y_pred[mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, 
                          labels=top_licenses)
    
    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=top_licenses, yticklabels=top_licenses)
    plt.xlabel('Predicted License')
    plt.ylabel('True License')
    plt.title(f'Confusion Matrix - Top {top_n} Licenses (Normalized)')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
```

**Visualization Choices**:
- **Top N filtering**: Full 110x110 matrix is unreadable; focus on common licenses
- **Normalization**: Row normalization shows recall for each license
- **Color scheme**: Blues gradient, darker = higher value
- **Annotations**: Display normalized values directly on heatmap
- **High DPI**: 300 DPI for publication quality

### 4.5.3 Prediction Interface

**`predict_license(pipeline, license_text: str) -> Tuple[str, float]`**

Predicts license for new text:

```python
def predict_license(pipeline, license_text: str):
    # Predict with probability
    prediction = pipeline.predict([license_text])[0]
    probabilities = pipeline.predict_proba([license_text])[0]
    confidence = probabilities.max()
    
    return prediction, confidence
```

**Usage Example**:
```python
model = load_model('outputs/license_classifier.pkl')
license_text = "Permission is hereby granted, free of charge..."
spdx_id, confidence = predict_license(model, license_text)
print(f"License: {spdx_id} (confidence: {confidence:.2f})")
```

## 4.6 Main Pipeline (`main.py`)

### 4.6.1 Command-Line Interface

**`parse_args()`**

Provides user-friendly CLI:

```python
parser = argparse.ArgumentParser(description='FOSS License Classification System')
parser.add_argument('--data-dir', default='data/scancode_licenses',
                    help='Path to dataset directory')
parser.add_argument('--output-dir', default='outputs',
                    help='Directory to save outputs')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='Test set fraction')
parser.add_argument('--top-n', type=int, default=10,
                    help='Number of licenses for confusion matrix')
parser.add_argument('--no-plot', action='store_true',
                    help='Skip plotting')
```

**Example Usage**:
```bash
# Default settings
python main.py

# Custom test size
python main.py --test-size 0.3

# Headless mode (no plots)
python main.py --no-plot
```

### 4.6.2 Execution Flow

**`main()`**

Orchestrates the complete workflow:

```python
def main():
    args = parse_args()
    
    # 1. Load dataset
    print("Loading dataset...")
    df = load_dataset(args.data_dir, min_samples_per_class=10)
    print(f"Loaded {len(df)} samples, {df['spdx_id'].nunique()} classes")
    
    # 2. Split data
    print("\nSplitting dataset...")
    train_df, test_df = split_dataset(df, test_size=args.test_size)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 3. Create pipeline
    print("\nCreating pipeline...")
    pipeline = create_pipeline()
    
    # 4. Train model
    print("\nTraining model...")
    pipeline = train_model(pipeline, train_df['text'], train_df['spdx_id'])
    
    # 5. Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(pipeline, test_df['text'], test_df['spdx_id'])
    
    # 6. Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision_macro']:.4f}")
    print(f"Recall:    {results['recall_macro']:.4f}")
    print(f"F1 Score:  {results['f1_macro']:.4f}")
    
    # 7. Save model and results
    save_model(pipeline, f"{args.output_dir}/license_classifier.pkl")
    
    # 8. Generate visualizations
    if not args.no_plot:
        y_pred = pipeline.predict(test_df['text'])
        plot_confusion_matrix(test_df['spdx_id'], y_pred, args.top_n)
    
    print("\nPipeline completed successfully!")
```

## 4.7 Benchmarking Framework

### 4.7.1 Abstract Base Class (`benchmarks/base_detector.py`)

**`BaseLicenseDetector`**

Defines interface for all detectors:

```python
class BaseLicenseDetector(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def setup(self):
        """Initialize detector (load model, prepare resources)."""
        pass
    
    @abstractmethod
    def detect(self, license_text: str) -> str:
        """Predict SPDX ID for given text."""
        pass
    
    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """Predict for batch of texts with progress bar."""
        iterator = tqdm(texts, desc=f"{self.name}") if show_progress else texts
        return [self.detect(text) for text in iterator]
    
    def benchmark_batch(self, texts: List[str], labels: List[str]) -> dict:
        """Benchmark on labeled data."""
        start_time = time.time()
        predictions = self.detect_batch(texts)
        elapsed = time.time() - start_time
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            'detector': self.name,
            'accuracy': accuracy,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall,
            'execution_time': elapsed,
            'avg_time_per_sample': elapsed / len(texts) if texts else 0
        }
```

**Design Benefits**:
- Enforces consistent interface across all detectors
- Provides default batch processing implementation
- Includes progress bar support (useful for slow detectors)
- Standardizes benchmark metrics

### 4.7.2 Detector Implementations

**ML Detector** (`benchmarks/detectors/ml_detector.py`):

```python
class MLDetector(BaseLicenseDetector):
    def __init__(self):
        super().__init__("ml_classifier")
        self.model = None
    
    def setup(self):
        self.model = load_model('outputs/license_classifier.pkl')
    
    def detect(self, license_text: str) -> str:
        prediction, _ = predict_license(self.model, license_text)
        return prediction
    
    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        # Optimized batch prediction
        return self.model.predict(texts).tolist()
```

**Keyword Detector** (`benchmarks/detectors/keyword_detector.py`):

```python
class KeywordDetector(BaseLicenseDetector):
    def __init__(self):
        super().__init__("keyword_matching")
        self.patterns = {
            'MIT': re.compile(r'permission is hereby granted.*without restriction', re.I | re.DOTALL),
            'Apache-2.0': re.compile(r'apache license.*version 2\.0', re.I | re.DOTALL),
            'GPL-3.0-only': re.compile(r'gnu general public license.*version 3', re.I | re.DOTALL),
            # ... 15 more patterns
        }
    
    def setup(self):
        pass  # Patterns already defined
    
    def detect(self, license_text: str) -> str:
        for license_id, pattern in self.patterns.items():
            if pattern.search(license_text):
                return license_id
        return "UNKNOWN"
```

**TF-IDF Similarity Detector** (`benchmarks/detectors/tfidf_similarity_detector.py`):

```python
class TFIDFSimilarityDetector(BaseLicenseDetector):
    def __init__(self):
        super().__init__("tfidf_similarity")
        self.vectorizer = None
        self.templates = None
        self.labels = None
    
    def setup(self):
        # Load training data
        df = load_dataset('data/scancode_licenses', min_samples_per_class=10)
        
        # Use longest text per license as template
        templates = df.groupby('spdx_id')['text'].apply(
            lambda x: max(x, key=len)
        ).reset_index()
        
        self.labels = templates['spdx_id'].values
        
        # Fit TF-IDF on templates
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.templates = self.vectorizer.fit_transform(templates['text'])
    
    def detect(self, license_text: str) -> str:
        # Vectorize query
        query_vec = self.vectorizer.transform([license_text])
        
        # Compute cosine similarity to all templates
        similarities = cosine_similarity(query_vec, self.templates)[0]
        
        # Return most similar
        best_idx = similarities.argmax()
        return self.labels[best_idx]
```

### 4.7.3 Benchmark Runner (`benchmarks/run_benchmark.py`)

**Main execution logic**:

```python
def main():
    # Setup detectors
    detectors = setup_detectors()
    
    # Setup datasets
    datasets = setup_datasets()
    
    # Run benchmarks
    results = []
    for detector in detectors:
        detector.setup()
        for dataset_name, (X, y) in datasets.items():
            print(f"\nBenchmarking {detector.name} on {dataset_name}...")
            result = detector.benchmark_batch(X, y)
            result['dataset'] = dataset_name
            results.append(result)
    
    # Save results
    save_results(results)
    generate_report(results)
```

## 4.8 Error Handling and Logging

### 4.8.1 Exception Handling

Throughout the codebase, we handle common errors:

**File I/O Errors**:
```python
try:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
except IOError as e:
    print(f"Warning: Could not read {file_path}: {e}")
    continue
```

**YAML Parsing Errors**:
```python
try:
    metadata = yaml.safe_load(yaml_content)
except yaml.YAMLError:
    return None, content
```

**Model Loading Errors**:
```python
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
```

### 4.8.2 Progress Reporting

We use `tqdm` for user feedback on long operations:

```python
for text in tqdm(texts, desc="Processing licenses"):
    prediction = detector.detect(text)
    predictions.append(prediction)
```

### 4.8.3 Logging

While the current implementation uses print statements, production systems should use Python's logging module:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading dataset...")
logger.warning("Skipping malformed file: %s", file_path)
logger.error("Failed to load model: %s", error)
```

## 4.9 Testing and Validation

### 4.9.1 Unit Tests

Key functions have informal tests:

**Data loading validation**:
```python
df = load_dataset('data/scancode_licenses')
assert len(df) > 0, "Dataset should not be empty"
assert 'text' in df.columns and 'spdx_id' in df.columns
assert df['text'].isna().sum() == 0, "No missing texts"
```

**Pipeline validation**:
```python
pipeline = create_pipeline()
assert hasattr(pipeline, 'fit') and hasattr(pipeline, 'predict')
```

### 4.9.2 Integration Tests

**End-to-end test**:
```python
# Load small sample
df = load_dataset('data/scancode_licenses', min_samples_per_class=10)
train_df, test_df = split_dataset(df, test_size=0.2)

# Train
pipeline = create_pipeline()
pipeline.fit(train_df['text'], train_df['spdx_id'])

# Predict
predictions = pipeline.predict(test_df['text'])
assert len(predictions) == len(test_df)
```

### 4.9.3 Smoke Tests

**Model serialization roundtrip**:
```python
# Train and save
pipeline = train_model(pipeline, X_train, y_train)
save_model(pipeline, 'temp_model.pkl')

# Load and predict
loaded_pipeline = load_model('temp_model.pkl')
pred1 = pipeline.predict(X_test)
pred2 = loaded_pipeline.predict(X_test)

assert (pred1 == pred2).all(), "Loaded model should match original"
```

---
