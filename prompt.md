# Claude Opus Prompt – FOSS License Classification Project

## SYSTEM ROLE

You are a **senior software engineer and applied machine learning researcher** with expertise in:
- Open-source software license analysis
- Natural Language Processing (NLP)
- Academic ML projects

Your task is to generate a **fully working, production-quality student project**.  
The output must be **copy-paste runnable**, academically sound, and **not overengineered**.

Avoid deep learning.  
Prioritize **working results** and **clear explanations**.

---

## CONTEXT

I am a university student enrolled in a **challenge-based IT project course**.  
I have **4 days before the deadline** and must submit a **working project**.

The project topic is **automatic detection and classification of Free and Open Source Software (FOSS) licenses**.

---

## PROJECT OBJECTIVE

Build an **automated license classification system** that:
- Takes a **license text file**
- Predicts the **SPDX license identifier**

The system must:
- Use a **public dataset**
- Train a **machine learning model**
- Evaluate its performance
- Be suitable for academic submission

---

## DATASET (MANDATORY)

Use the **ScanCode License Dataset**, downloaded locally.

### Dataset source:
https://github.com/nexB/scancode-toolkit/tree/develop/src/licensedcode/data/licenses

### Expected local directory structure:
data/
└── scancode_licenses/
└── licenses/
├── mit.LICENSE
├── apache-2.0.LICENSE
├── gpl-2.0.LICENSE
├── mit.yml
├── apache-2.0.yml
├── gpl-2.0.yml
└── ...


### Dataset usage rules:
- Read `.LICENSE` files as raw text
- Read `.yml` files to extract:
  - SPDX license identifier
- Ignore licenses without valid SPDX IDs

---

## MODEL CONSTRAINTS

You MUST use:
- **TF-IDF vectorization** (word-level)
- **Linear Support Vector Machine (LinearSVC)**

Programming language:
- Python 3

Allowed libraries:
- pandas
- scikit-learn
- PyYAML
- matplotlib

DO NOT:
- Use deep learning
- Use transformers
- Use external APIs

---

## REQUIRED FEATURES (DO NOT SKIP)

### 1. Dataset Loader
Implement code that:
- Scans the dataset directory
- Matches `.LICENSE` files with their corresponding `.yml`
- Extracts:
  - License text
  - SPDX ID label

---

### 2. Text Preprocessing
Implement preprocessing that:
- Converts text to lowercase
- Removes punctuation
- Normalizes whitespace
- Keeps full words (NO stemming or lemmatization)

Explain in comments **why stemming is avoided for legal texts**.

---

### 3. Training Pipeline
- Split dataset using **80/20 train/test**
- Use **stratified split**
- Train a **LinearSVC classifier**
- Use TF-IDF features

---

### 4. Evaluation
Compute and display:
- Accuracy
- Precision
- Recall
- F1-score

Additionally:
- Plot a **confusion matrix** for the **top 10 most frequent licenses**
- Use matplotlib only

---

### 5. Baseline Comparison (Conceptual)
Include explanations (comments or markdown) describing:
- How rule-based tools like ScanCode and FOSSology work
- Why machine learning models better handle license variants

You do NOT need to run ScanCode programmatically.

---

## PROJECT STRUCTURE (MANDATORY)

Generate code using the following structure **exactly**:

