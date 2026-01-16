"""
Hyperparameter Tuning for License Classification.

This module implements grid search cross-validation to find
optimal parameters for TF-IDF and LinearSVC.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

from .preprocessor import preprocess_text


def tune_model(train_df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """
    Perform grid search to find optimal hyperparameters.
    
    Args:
        train_df: Training DataFrame with 'text' and 'spdx_id' columns.
        n_jobs: Number of parallel jobs (-1 for all cores).
        
    Returns:
        Dictionary with best parameters and scores.
    """
    X_train = train_df['text'].values
    y_train = train_df['spdx_id'].values
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess_text,
            analyzer='word',
            sublinear_tf=True,
        )),
        ('classifier', LinearSVC(
            class_weight='balanced',
            random_state=42,
            dual='auto',
            max_iter=20000,
        ))
    ])
    
    # Parameter grid
    param_grid = {
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'tfidf__max_features': [15000, 25000],
        'tfidf__min_df': [2, 3],
        'tfidf__max_df': [0.85, 0.90],
        'classifier__C': [0.1, 0.5, 1.0],
    }
    
    # Stratified 3-fold CV (faster than 5-fold)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Starting grid search (this may take a few minutes)...")
    print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=2,
        refit=True,
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest CV score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
    }


if __name__ == "__main__":
    from .data_loader import load_dataset
    from .train import split_dataset
    
    df = load_dataset("data/scancode_licenses")
    train_df, _ = split_dataset(df)
    
    results = tune_model(train_df)
    print("\nBest parameters found:")
    for k, v in results['best_params'].items():
        print(f"  {k}: {v}")
