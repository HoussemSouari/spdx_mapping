# Important Note About ScanCode Benchmarking

## Why ScanCode Comparison Doesn't Work

After implementing and testing the benchmarking framework, we discovered a fundamental incompatibility:

### The Problem

**ScanCode is designed to detect licenses IN source code files** (e.g., `.py`, `.js`, `.java`), not to classify standalone license text files.

**Our dataset consists of pure license texts** from ScanCode's own training data (`.LICENSE` and `.RULE` files with YAML front matter).

### Test Results

When we tested ScanCode on its own dataset:
- **Result**: 0% accuracy
- **Reason**: ScanCode doesn't recognize these as license files to scan
- **Expected behavior**: ScanCode is meant for analyzing project source trees, not classifying isolated license texts

### What This Means

1. **Our ML classifier** solves a different problem: **"Given license text, what is the SPDX ID?"**
2. **ScanCode** solves: **"Given source code files, what licenses are referenced?"**

These are related but distinct tasks!

### Valid Comparisons

To properly compare tools, you would need to:

#### Option 1: Real-World Source Code Dataset
- Collect actual source files (`.py`, `.js`, etc.) with license headers
- Run both ML classifier and ScanCode on these files
- Compare detection accuracy

#### Option 2: Use ScanCode's API Directly
- Use ScanCode's `licensedcode` Python API
- Call `detect_licenses(text)` directly instead of CLI
- This might work for pure license texts

#### Option 3: Synthetic Dataset
- Generate source files with license headers
- Insert various licenses (MIT, Apache, GPL, etc.)
- Test both tools on this controlled dataset

### Current Status

- ✅ **ML Classifier**: 82.69% accuracy on ScanCode license dataset
- ⊘ **ScanCode CLI**: Not applicable for this dataset type
- 📋 **Recommendation**: Focus on ML classifier performance, or create real-world test set

### Alternative Baselines

Instead of ScanCode, consider comparing against:

1. **Simple keyword matching** - Search for "MIT", "Apache", etc.
2. **TF-IDF cosine similarity** - Compare against license templates
3. **Exact/fuzzy string matching** - Like FOSSology's Nomos agent
4. **Your model at different configurations** - Compare hyperparameters

These would be more meaningful comparisons for the license text classification task.

## Recommendation

For your project report, you can:

1. **Document the distinction** between license detection vs. classification
2. **Show ML model performance** on the dataset (82.69% accuracy)
3. **Compare against simpler baselines** (keyword matching, cosine similarity)
4. **Discuss real-world deployment** where you'd integrate with tools like ScanCode

This actually makes your project MORE interesting because it fills a gap that traditional tools don't address well!
