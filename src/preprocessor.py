"""
Text Preprocessing for License Classification.

This module provides text preprocessing utilities specifically designed
for legal license texts. The preprocessing pipeline includes:
- Lowercase conversion
- Punctuation removal
- Whitespace normalization

IMPORTANT: Why we avoid stemming and lemmatization for legal texts
--------------------------------------------------------------
1. Legal Precision: Legal documents use precise terminology where word forms
   carry specific legal meaning. For example, "licensor" vs "licensee" are
   distinct legal entities - stemming both to "licens" loses critical meaning.

2. Defined Terms: Licenses often define specific terms (e.g., "Derivative Works",
   "Contribution") that must remain unchanged for accurate classification.

3. Standard Phrases: Licenses contain standard legal phrases where exact wording
   matters for identification (e.g., "AS IS" vs "AS-IS" vs "as is").

4. Version Sensitivity: Minor word changes distinguish license versions
   (GPL-2.0 vs GPL-3.0). Aggressive normalization may blur these distinctions.

5. Copyright Holders: Names and entity references should remain intact for
   matching license templates.

TF-IDF with full words provides sufficient discrimination while preserving
the legal precision needed for accurate license classification.
"""

import re
import string


# Pre-compile regex patterns at module level for efficiency
_PUNCTUATION_PATTERN = re.compile(f'[{re.escape(string.punctuation)}]')
_WHITESPACE_PATTERN = re.compile(r'\s+')


def preprocess_text(text: str) -> str:
    """
    Preprocess a license text for classification.
    
    This function is defined at module level (not nested) so it can be
    pickled when saving the trained model.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Normalize whitespace (collapse multiple spaces, strip)
    
    Args:
        text: Raw license text.
        
    Returns:
        Preprocessed text ready for TF-IDF vectorization.
    """
    if not text:
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation
    # Replace punctuation with space to avoid joining words
    text = _PUNCTUATION_PATTERN.sub(' ', text)
    
    # Step 3: Normalize whitespace
    # Collapse multiple whitespace characters into single space
    text = _WHITESPACE_PATTERN.sub(' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def create_preprocessor():
    """
    Return the preprocessing function for use in sklearn pipelines.
    
    Returns:
        The preprocess_text function (module-level, picklable).
    """
    return preprocess_text


if __name__ == "__main__":
    # Quick test
    sample = """
    MIT License
    
    Copyright (c) 2024 Example Corp.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction...
    """
    
    print("Original:")
    print(sample)
    print("\nPreprocessed:")
    print(preprocess_text(sample))
