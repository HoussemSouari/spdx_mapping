"""
Dataset Loader for ScanCode License Dataset.

This module scans the dataset directory and extracts license texts along with
their SPDX license identifiers from:
1. .LICENSE files (containing YAML front matter with spdx_license_key)
2. .RULE files (containing license_expression and text variants)

The RULE files provide multiple text variants for each license, which is
essential for training an ML classifier (need >= 2 samples per class).

Dataset source:
https://github.com/nexB/scancode-toolkit/tree/develop/src/licensedcode/data/licenses
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import yaml
import pandas as pd


def parse_yaml_front_matter(content: str) -> Tuple[Optional[dict], str]:
    """
    Parse YAML front matter from a file content.
    
    Args:
        content: Full file content.
        
    Returns:
        Tuple of (metadata dict, remaining text). metadata may be None.
    """
    # Pattern to match YAML front matter: starts with ---, ends with ---
    front_matter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)
    match = front_matter_pattern.match(content)
    
    if not match:
        return None, content
    
    yaml_content = match.group(1)
    remaining_text = content[match.end():].strip()
    
    try:
        metadata = yaml.safe_load(yaml_content)
        if not isinstance(metadata, dict):
            return None, content
        return metadata, remaining_text
    except yaml.YAMLError:
        return None, content


def build_license_key_to_spdx_map(licenses_dir: Path) -> Dict[str, str]:
    """
    Build a mapping from license key (e.g., 'mit') to SPDX ID (e.g., 'MIT').
    
    Reads all .LICENSE files to extract the key -> spdx_license_key mapping.
    
    Args:
        licenses_dir: Path to the licenses directory.
        
    Returns:
        Dictionary mapping license keys to SPDX IDs.
    """
    key_to_spdx = {}
    
    for license_path in licenses_dir.glob("*.LICENSE"):
        try:
            with open(license_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except IOError:
            continue
        
        metadata, _ = parse_yaml_front_matter(content)
        
        if metadata is None:
            continue
        
        # Get the license key from metadata (should match filename)
        license_key = metadata.get('key')
        spdx_id = metadata.get('spdx_license_key')
        
        if license_key and spdx_id:
            key_to_spdx[license_key] = spdx_id
    
    return key_to_spdx


def parse_license_expression(expr: str) -> Optional[str]:
    """
    Parse a license expression to extract the primary license key.
    
    License expressions can be complex (e.g., "mit AND apache-2.0"),
    but for classification we only consider simple single-license expressions.
    
    Args:
        expr: License expression string.
        
    Returns:
        Simple license key if expression is simple, None otherwise.
    """
    if not expr or not isinstance(expr, str):
        return None
    
    expr = expr.strip().lower()
    
    # Skip complex expressions with AND, OR, WITH
    if ' and ' in expr or ' or ' in expr or ' with ' in expr:
        return None
    
    return expr


def load_dataset(
    data_dir: str,
    use_rules: bool = True,
    min_samples_per_class: int = 10,  # Require more samples for reliable training
    min_text_length: int = 150        # Filter out short samples for better quality
) -> pd.DataFrame:
    """
    Load the ScanCode license dataset from a local directory.
    
    Uses both .LICENSE files (for SPDX mapping) and .RULE files (for
    text variants). This provides multiple samples per license class,
    which is essential for training ML classifiers.
    
    Args:
        data_dir: Path to the base scancode_licenses directory
                  (should contain 'licenses/' and optionally 'rules/').
        use_rules: Whether to load rules for additional samples.
        min_samples_per_class: Minimum samples required per class.
        min_text_length: Minimum text length in characters.
        
    Returns:
        A DataFrame with columns:
            - 'license_key': The license key (e.g., 'mit')
            - 'text': The license/rule text
            - 'spdx_id': The SPDX license identifier
            - 'source': 'license' or 'rule'
    """
    data_path = Path(data_dir)
    
    # Handle both direct licenses folder and parent folder
    if (data_path / 'licenses').exists():
        licenses_dir = data_path / 'licenses'
        rules_dir = data_path / 'rules'
    elif data_path.name == 'licenses':
        licenses_dir = data_path
        rules_dir = data_path.parent / 'rules'
    else:
        raise FileNotFoundError(f"Could not find licenses directory in: {data_dir}")
    
    if not licenses_dir.exists():
        raise FileNotFoundError(f"Licenses directory not found: {licenses_dir}")
    
    # Build mapping from license key to SPDX ID
    print("Building license key to SPDX ID mapping...")
    key_to_spdx = build_license_key_to_spdx_map(licenses_dir)
    print(f"Found {len(key_to_spdx)} license keys with SPDX IDs")
    
    records: List[dict] = []
    
    # Load .LICENSE files
    license_files = list(licenses_dir.glob("*.LICENSE"))
    print(f"Found {len(license_files)} .LICENSE files")
    
    for license_path in license_files:
        try:
            with open(license_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except IOError:
            continue
        
        metadata, license_text = parse_yaml_front_matter(content)
        
        if metadata is None or not license_text.strip():
            continue
        
        # Skip very short license texts
        if len(license_text) < min_text_length:
            continue
        
        license_key = metadata.get('key', license_path.stem)
        spdx_id = metadata.get('spdx_license_key')
        
        if not spdx_id:
            continue
        
        records.append({
            'license_key': license_key,
            'text': license_text,
            'spdx_id': spdx_id,
            'source': 'license'
        })
    
    # Load .RULE files for additional samples
    if use_rules and rules_dir.exists():
        rule_files = list(rules_dir.glob("*.RULE"))
        print(f"Found {len(rule_files)} .RULE files")
        
        rule_count = 0
        for rule_path in rule_files:
            try:
                with open(rule_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except IOError:
                continue
            
            metadata, rule_text = parse_yaml_front_matter(content)
            
            if metadata is None or not rule_text.strip():
                continue
            
            # Get license expression
            license_expr = metadata.get('license_expression', '')
            license_key = parse_license_expression(license_expr)
            
            if not license_key:
                continue
            
            # Map to SPDX ID
            spdx_id = key_to_spdx.get(license_key)
            
            if not spdx_id:
                continue
            
            # Skip very short rules (use same threshold as license files)
            if len(rule_text) < min_text_length:
                continue
            
            records.append({
                'license_key': license_key,
                'text': rule_text,
                'spdx_id': spdx_id,
                'source': 'rule'
            })
            rule_count += 1
        
        print(f"Loaded {rule_count} rules with valid SPDX mappings")
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("Warning: No valid samples loaded!")
        return df
    
    # Filter to classes with minimum samples
    class_counts = df['spdx_id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df['spdx_id'].isin(valid_classes)].copy()
    
    print(f"\nLoaded {len(df)} total samples")
    print(f"Classes with >= {min_samples_per_class} samples: {df['spdx_id'].nunique()}")
    
    return df


def get_label_distribution(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Get the distribution of license labels in the dataset.
    
    Args:
        df: The license DataFrame.
        top_n: Number of top labels to return.
        
    Returns:
        A Series with label counts, sorted descending.
    """
    return df['spdx_id'].value_counts().head(top_n)


if __name__ == "__main__":
    # Quick test
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/scancode_licenses/licenses"
    
    df = load_dataset(data_dir)
    print("\nDataset shape:", df.shape)
    print("\nTop 10 licenses:")
    print(get_label_distribution(df, 10))
