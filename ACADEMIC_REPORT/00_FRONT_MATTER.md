# Machine Learning-Based FOSS License Classification System

## Academic Technical Report

---

**Project Type:** Final Year Academic Project  
**Domain:** Machine Learning, Natural Language Processing, Software Engineering  
**Date:** January 2026

---

## Abstract

This report presents a comprehensive machine learning system for automated classification of Free and Open Source Software (FOSS) licenses. The project addresses the critical challenge of license identification in modern software development, where organizations must manage hundreds or thousands of dependencies with varying license terms. Traditional rule-based tools struggle with text variations, partial content, and non-standard formatting, creating compliance risks and requiring extensive manual review.

We developed a text classification system using TF-IDF vectorization combined with Support Vector Machines (SVM) that achieves **79.4% accuracy** on a test set of 403 samples across 110 license classes, significantly outperforming baseline approaches including Naive Bayes (65.3%), TF-IDF similarity (24.8%), and keyword matching (20.8%). The system demonstrates particular strength in handling edge cases such as typos, paraphrasing, and partial license text, achieving 60% accuracy on challenging scenarios where template-matching approaches achieve 0%.

The dataset comprises 8,058 license samples extracted from the ScanCode Toolkit, including both canonical license texts and 36,472 detection rules representing real-world variations. Our optimized pipeline combines word-level n-grams (1-2 grams) and character-level n-grams (3-5 grams) with chi-squared feature selection, reducing the feature space from 17,000 to 10,000 dimensions while maintaining high accuracy.

This work demonstrates that machine learning approaches offer superior robustness and practical value for license classification compared to traditional exact-matching tools, particularly in environments with modified or non-standard license texts. The system processes licenses in 2.0 milliseconds per sample, making it suitable for real-time CI/CD integration and large-scale code auditing.

**Keywords:** License Classification, Machine Learning, Text Classification, TF-IDF, Support Vector Machines, SPDX, Open Source Compliance, Natural Language Processing

---

## Acknowledgments

We would like to express our gratitude to the nexB organization for maintaining the ScanCode Toolkit and making their comprehensive license dataset publicly available. This project would not have been possible without their extensive work in cataloging and standardizing FOSS licenses.

We also acknowledge the broader open source community for establishing the SPDX (Software Package Data Exchange) standard, which provides the standardized license identifiers used throughout this work.

---

## List of Figures

1. System Architecture Overview
2. Dataset Distribution and Class Balance
3. Feature Extraction Pipeline
4. Model Training and Evaluation Workflow
5. Confusion Matrix - Top 10 License Classes
6. Benchmark Performance Comparison
7. Accuracy vs F1-Score Trade-off
8. Speed Performance Comparison
9. Edge Case Demonstration Results
10. Feature Selection Impact Analysis

---

## List of Tables

1. Dataset Statistics and Composition
2. Class Distribution Analysis
3. Training and Test Set Split
4. Hyperparameter Configuration
5. Model Performance Metrics
6. Comparative Benchmark Results
7. Edge Case Test Results
8. Processing Time Analysis
9. Memory Usage Statistics
10. Baseline Comparison Summary

---

## Abbreviations and Acronyms

| Acronym | Full Form |
|---------|-----------|
| API | Application Programming Interface |
| ASCII | American Standard Code for Information Interchange |
| BSD | Berkeley Software Distribution |
| CI/CD | Continuous Integration/Continuous Deployment |
| CLI | Command Line Interface |
| FOSS | Free and Open Source Software |
| FSF | Free Software Foundation |
| GPL | GNU General Public License |
| HTML | HyperText Markup Language |
| JSON | JavaScript Object Notation |
| LGPL | GNU Lesser General Public License |
| MIT | Massachusetts Institute of Technology |
| ML | Machine Learning |
| MPL | Mozilla Public License |
| NLP | Natural Language Processing |
| OSI | Open Source Initiative |
| ROC | Receiver Operating Characteristic |
| SBOM | Software Bill of Materials |
| SPDX | Software Package Data Exchange |
| SVM | Support Vector Machine |
| TF-IDF | Term Frequency-Inverse Document Frequency |
| YAML | YAML Ain't Markup Language |
| χ² | Chi-Squared |

---
