"""
Ensemble Model for License Classification.

This module implements a voting ensemble combining multiple classifiers
for improved accuracy and robustness.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2

from .preprocessor import create_preprocessor


def create_ensemble_pipeline() -> Pipeline:
    """
    Create an ensemble pipeline combining multiple classifiers.
    
    Uses soft voting (probability averaging) between:
    - LinearSVC (calibrated)
    - Logistic Regression
    - SGD Classifier
    
    Returns:
        A scikit-learn Pipeline object.
    """
    preprocessor = create_preprocessor()
    
    # Feature extraction
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
    
    # Individual classifiers
    svc = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=0.5,
            max_iter=20000,
            class_weight='balanced',
            random_state=42,
            dual='auto',
        ),
        cv=3,
        method='sigmoid',
    )
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=5000,
        class_weight='balanced',
        random_state=42,
        solver='saga',
        n_jobs=-1,
    )
    
    sgd = CalibratedClassifierCV(
        estimator=SGDClassifier(
            loss='modified_huber',
            alpha=0.0001,
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        cv=3,
        method='sigmoid',
    )
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('svc', svc),
            ('lr', lr),
            ('sgd', sgd),
        ],
        voting='soft',  # Use probability averaging
        weights=[2, 1, 1],  # Give more weight to SVC
        n_jobs=-1,
    )
    
    pipeline = Pipeline([
        ('features', feature_extraction),
        ('select_best', SelectKBest(chi2, k=10000)),
        ('classifier', ensemble),
    ])
    
    return pipeline


if __name__ == "__main__":
    from .data_loader import load_dataset
    from .train import split_dataset, save_model
    from .evaluate import evaluate_model
    
    print("Loading data...")
    df = load_dataset("data/scancode_licenses")
    train_df, test_df = split_dataset(df)
    
    print("\nCreating ensemble pipeline...")
    pipeline = create_ensemble_pipeline()
    
    print("Training ensemble model...")
    X_train = train_df['text'].values
    y_train = train_df['spdx_id'].values
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluating ensemble model...")
    evaluate_model(pipeline, test_df)
    
    save_model(pipeline, "outputs/license_classifier_ensemble.pkl")
