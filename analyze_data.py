#!/usr/bin/env python3
"""Analyze dataset to find improvement opportunities."""

from src.data_loader import load_dataset

df = load_dataset('data/scancode_licenses')
print('\n=== DATA ANALYSIS ===')
print(f'Total samples: {len(df)}')
print(f'Unique classes: {df["spdx_id"].nunique()}')

# Class distribution
dist = df['spdx_id'].value_counts()
print(f'\nClasses with >= 10 samples: {(dist >= 10).sum()}')
print(f'Classes with >= 5 samples: {(dist >= 5).sum()}')
print(f'Classes with 2-4 samples: {((dist >= 2) & (dist < 5)).sum()}')

# Text length analysis
df['text_len'] = df['text'].str.len()
print(f'\nText length stats:')
print(f'  Mean: {df["text_len"].mean():.0f} chars')
print(f'  Median: {df["text_len"].median():.0f} chars')
print(f'  Min: {df["text_len"].min()} chars')
print(f'  < 50 chars: {(df["text_len"] < 50).sum()} samples')
print(f'  < 100 chars: {(df["text_len"] < 100).sum()} samples')

# Show some very short samples
print('\nExamples of very short texts:')
short = df[df['text_len'] < 50].head(5)
for _, row in short.iterrows():
    print(f'  [{row["spdx_id"]}]: "{row["text"][:60]}"')
