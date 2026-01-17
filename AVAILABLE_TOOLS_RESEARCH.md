# License Text Classification Tools

## Research: Tools That Classify License TEXT (Not Source Code)

After researching available tools, here's what exists:

---

## 1. **Askalono** (Amazon/jpeddicord)

**Repository:** https://github.com/jpeddicord/askalono

**What it does:**
- Uses **text matching algorithm** to identify license text
- Compares input against a database of known license texts
- Returns SPDX identifier with confidence score

**Approach:**
- Bag-of-words + TF-IDF for text comparison
- Optimized for speed using Rust
- Works with **license text files**, not source code

**Strengths:**
- Very fast (Rust implementation)
- Good accuracy on canonical licenses
- CLI tool ready to use

**Limitations:**
- Requires mostly complete license text
- Less robust to heavy modifications
- Database limited to ~500 licenses

**Installation:**
```bash
# Requires Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install askalono-cli
```

**Usage:**
```bash
askalono identify LICENSE.txt
# Output: License: MIT (100% confidence)
```

**Benchmarking Potential:** ✅ **YES - This is directly comparable!**

---

## 2. **Licensee** (GitHub)

**Repository:** https://github.com/licensee/licensee

**What it does:**
- Ruby gem for license detection
- Compares files against license templates
- Used by GitHub to detect repository licenses

**Approach:**
- Dice coefficient for text similarity
- Removes boilerplate (copyright notices, etc.)
- Focuses on substantive license text

**Strengths:**
- Battle-tested (powers GitHub's license detection)
- Handles common variations
- Good for standard licenses

**Limitations:**
- Primarily designed for LICENSE files in repos
- Less effective on heavily modified text
- Ruby dependency

**Installation:**
```bash
gem install licensee
```

**Usage:**
```bash
licensee detect /path/to/project
# Or for single file:
licensee diff LICENSE.txt
```

**Benchmarking Potential:** ✅ **YES - Can compare on license text**

---

## 3. **License-Identifier** (ClearlyDefined)

**Repository:** https://github.com/clearlydefined/license-identifier

**What it does:**
- Node.js tool for license identification
- Uses fuzzy matching against SPDX licenses
- Part of ClearlyDefined project

**Approach:**
- String similarity algorithms
- Normalization (whitespace, copyright removal)
- Returns SPDX identifier

**Strengths:**
- JavaScript/Node.js (easy to integrate)
- Open source, well-maintained
- SPDX-focused

**Limitations:**
- Requires substantial text match
- Not as robust to variations

**Installation:**
```bash
npm install -g license-identifier
```

**Benchmarking Potential:** ✅ **YES**

---

## 4. **LicenseDB** (go-enry)

**Repository:** https://github.com/go-enry/go-license-detector

**What it does:**
- Go library for license detection
- Uses n-gram similarity
- Fast and efficient

**Approach:**
- Character n-grams for matching
- Confidence scoring
- Database of common licenses

**Strengths:**
- Very fast (Go implementation)
- Good for high-volume scanning
- Handles license files directly

**Limitations:**
- Go dependency
- Focus on speed over robustness

**Installation:**
```bash
go get github.com/go-enry/go-license-detector/v4/licensedb
```

**Benchmarking Potential:** ✅ **YES**

---

## 5. **LicenseCheck** (Debian devscripts)

**Repository:** https://metacpan.org/pod/licensecheck

**What it does:**
- Perl script for license detection
- Scans source files AND license files
- Used in Debian packaging

**Approach:**
- Regex-based pattern matching
- Heuristics for common licenses
- Returns license name

**Strengths:**
- Ships with Debian/Ubuntu
- Well-tested on millions of packages
- Handles common cases well

**Limitations:**
- Perl-based (less modern)
- Regex patterns can be brittle
- Not SPDX-focused

**Installation:**
```bash
sudo apt-get install licensecheck
```

**Usage:**
```bash
licensecheck LICENSE
```

**Benchmarking Potential:** ⚠️ **MAYBE - Less sophisticated**

---

## 6. **ScanCode LicenseDB** (Direct API)

**Repository:** https://github.com/nexB/scancode-toolkit

**What it does:**
- ScanCode has a Python API for license detection
- `licensedcode.cache.get_index()` provides direct access
- Can match license text directly

**Approach:**
- Uses same detection as CLI but via API
- Avoids subprocess overhead
- Direct license text matching

**Python API:**
```python
from licensedcode.cache import get_index
from licensedcode.spans import Span

idx = get_index()
matches = idx.match(text=license_text)
for match in matches:
    print(match.rule.identifier, match.score())
```

**Strengths:**
- Industry standard detection engine
- Very comprehensive database
- Python-native (no subprocess)

**Limitations:**
- Still optimized for source code scanning
- Complex API
- Heavy dependency

**Benchmarking Potential:** ✅ **YES - Best comparison!**

---

## 7. **License-Expression-Tools** (nexB)

**Repository:** https://github.com/nexB/license-expression

**What it does:**
- Parse and compare license expressions
- Handles compound licenses (GPL-2.0 OR Apache-2.0)
- Validates SPDX expressions

**Note:** This is more for **parsing expressions**, not detecting licenses in text.

**Benchmarking Potential:** ❌ **NO - Different use case**

---

## Recommended Tools for Benchmarking

Based on your use case (license text → SPDX ID), here are the best comparisons:

### Tier 1: Direct Competitors ✅

1. **Askalono** - Text matching, very fast, SPDX output
2. **Licensee** - GitHub's tool, battle-tested
3. **ScanCode API** - Industry standard, Python-native

### Tier 2: Worth Testing ⚠️

4. **License-Identifier** - Node.js, fuzzy matching
5. **go-license-detector** - Fast, n-gram based

### Tier 3: Less Relevant ❌

6. **LicenseCheck** - Regex-based, less sophisticated
7. **License-Expression-Tools** - Different use case

---

## Implementation Plan

Let me create detectors for the Tier 1 tools:

### 1. Askalono Detector
```python
class AskalonoDetector(BaseLicenseDetector):
    def detect(self, text: str) -> str:
        # Write to temp file
        # Run: askalono identify temp_file.txt --format json
        # Parse JSON, return SPDX ID
```

### 2. Licensee Detector
```python
class LicenseeDetector(BaseLicenseDetector):
    def detect(self, text: str) -> str:
        # Write to temp file
        # Run: licensee detect --json temp_file.txt
        # Parse JSON, return SPDX ID
```

### 3. ScanCode API Detector
```python
class ScanCodeAPIDetector(BaseLicenseDetector):
    def detect(self, text: str) -> str:
        from licensedcode.cache import get_index
        idx = get_index()
        matches = idx.match(text=text)
        # Return best match SPDX ID
```

---

## Expected Results

Based on their approaches:

| Tool | Expected Accuracy | Speed | Robustness to Variations |
|------|-------------------|-------|--------------------------|
| **Your ML Model** | **79-83%** | Fast | ✅ High |
| Askalono | 70-80% | Very Fast | Medium |
| Licensee | 65-75% | Medium | Medium |
| ScanCode API | 75-85% | Slow | Medium-High |
| License-Identifier | 60-70% | Medium | Low-Medium |

**Hypothesis:** Your ML model will outperform on **robustness** (typos, variations) while Askalono/ScanCode may have edge on **canonical licenses**.

---

## Next Steps

Would you like me to:

1. ✅ **Implement Askalono detector** (if you install Rust/Cargo)
2. ✅ **Implement Licensee detector** (if you install Ruby)
3. ✅ **Implement ScanCode API detector** (already have ScanCode installed)
4. 📊 **Run comprehensive benchmark** comparing all tools
5. 📈 **Generate comparison visualizations**

Let me know which tools you want to compare against!
