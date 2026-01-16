"""
Training Pipeline for License Classification.

This module implements the machine learning pipeline using:
- TF-IDF vectorization (word-level + character n-grams)
- Feature selection (SelectKBest with chi-squared)
- Linear Support Vector Machine (LinearSVC) with calibration

The pipeline uses stratified train/test split (80/20) to ensure
balanced representation of license classes in both sets.
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from .preprocessor import create_preprocessor


def create_pipeline() -> Pipeline:
    """
    Create an enhanced classification pipeline with:
    - Word n-grams (1-2) + Character n-grams (3-5) via FeatureUnion
    - Feature selection using chi-squared test
    - Calibrated LinearSVC for probability estimates
    
    The combination of word and character features captures both:
    - Semantic meaning (word n-grams)
    - Morphological patterns (char n-grams for legal terminology)
    
    Returns:
        A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor()
    
    # Combined word + character n-gram features
    feature_extraction = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            preprocessor=preprocessor,
            analyzer='word',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=12000,  # Word features
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
            norm='l2',
        )),
        ('char_tfidf', TfidfVectorizer(
            preprocessor=preprocessor,
            analyzer='char_wb',  # Character n-grams within word boundaries
            ngram_range=(3, 5),  # 3-5 character sequences
            max_features=5000,   # Character features
            sublinear_tf=True,
            min_df=3,
            max_df=0.90,
            norm='l2',
        )),
    ])
    
    # Base classifier with optimized parameters
    base_classifier = LinearSVC(
        C=0.5,
        max_iter=20000,
        class_weight='balanced',
        random_state=42,
        dual='auto',
        loss='squared_hinge',
    )
    
    # Calibrated classifier provides probability estimates
    calibrated_classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        cv=3,  # 3-fold cross-validation for calibration
        method='sigmoid',
    )
    
    pipeline = Pipeline([
        ('features', feature_extraction),
        ('select_best', SelectKBest(chi2, k=10000)),  # Select top 10k features
        ('classifier', calibrated_classifier),
    ])
    
    return pipeline


def create_simple_pipeline() -> Pipeline:
    """
    Create a simpler pipeline without feature union (faster training).
    Use this for quick experiments.
    """
    preprocessor = create_preprocessor()
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocessor,
            analyzer='word',
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
            norm='l2',
        )),
        ('classifier', LinearSVC(
            C=0.5,
            max_iter=20000,
            class_weight='balanced',
            random_state=42,
            dual='auto',
            loss='squared_hinge',
        ))
    ])
    
    return pipeline


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples_per_class: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets using stratified sampling.
    
    For stratified splitting to work, each class must have at least 2 samples.
    Classes with fewer samples are filtered out.
    
    Args:
        df: The full dataset DataFrame.
        test_size: Fraction of data for testing (default 0.2 = 20%).
        random_state: Random seed for reproducibility.
        min_samples_per_class: Minimum samples required per class.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    # Filter classes with insufficient samples for stratified split
    class_counts = df['spdx_id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    
    filtered_df = df[df['spdx_id'].isin(valid_classes)].copy()
    
    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(f"Removed {removed_count} samples from classes with < {min_samples_per_class} samples")
    
    # Stratified split
    train_df, test_df = train_test_split(
        filtered_df,
        test_size=test_size,
        random_state=random_state,
        stratify=filtered_df['spdx_id']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Number of classes: {train_df['spdx_id'].nunique()}")
    
    return train_df, test_df


def train_model(
    train_df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None
) -> Pipeline:
    """
    Train the classification model on the training data.
    
    Args:
        train_df: Training DataFrame with 'text' and 'spdx_id' columns.
        pipeline: Optional pre-configured pipeline. If None, creates default.
        
    Returns:
        The trained pipeline.
    """
    if pipeline is None:
        pipeline = create_pipeline()
    
    X_train = train_df['text'].values
    y_train = train_df['spdx_id'].values
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    return pipeline


def save_model(pipeline: Pipeline, output_path: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        pipeline: The trained pipeline.
        output_path: Path to save the model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {output_path}")


def load_model(model_path: str) -> Pipeline:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file.
        
    Returns:
        The loaded pipeline.
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Quick test
    from .data_loader import load_dataset
    
    df = load_dataset("data/scancode_licenses/licenses")
    train_df, test_df = split_dataset(df)
    
    pipeline = train_model(train_df)
    
    # Quick accuracy check
    X_test = test_df['text'].values
    y_test = test_df['spdx_id'].values
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
