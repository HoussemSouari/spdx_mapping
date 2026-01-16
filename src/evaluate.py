"""
Evaluation Module for License Classification.

This module provides functions to evaluate the trained model and
visualize results including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix for top N licenses
"""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    print_report: bool = True
) -> dict:
    """
    Evaluate the model on test data and compute classification metrics.
    
    Args:
        pipeline: The trained classification pipeline.
        test_df: Test DataFrame with 'text' and 'spdx_id' columns.
        print_report: Whether to print the classification report.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    X_test = test_df['text'].values
    y_test = test_df['spdx_id'].values
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"\nPrecision (macro):   {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"\nRecall (macro):   {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"\nF1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    print("="*60 + "\n")
    
    if print_report:
        # Get unique labels for the report
        labels = sorted(set(y_test) | set(y_pred))
        print("\nDetailed Classification Report:")
        print("-"*60)
        print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    
    return metrics


def get_top_n_labels(
    test_df: pd.DataFrame,
    n: int = 10
) -> List[str]:
    """
    Get the top N most frequent labels in the test set.
    
    Args:
        test_df: Test DataFrame with 'spdx_id' column.
        n: Number of top labels to return.
        
    Returns:
        List of top N label names.
    """
    return test_df['spdx_id'].value_counts().head(n).index.tolist()


def plot_confusion_matrix(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    top_n: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot a confusion matrix for the top N most frequent licenses.
    
    Args:
        pipeline: The trained classification pipeline.
        test_df: Test DataFrame with 'text' and 'spdx_id' columns.
        top_n: Number of top licenses to include in the matrix.
        output_path: Optional path to save the figure.
        figsize: Figure size as (width, height).
    """
    # Get top N labels from test set
    top_labels = get_top_n_labels(test_df, top_n)
    
    # Filter test data to only include top N labels
    mask = test_df['spdx_id'].isin(top_labels)
    filtered_df = test_df[mask].copy()
    
    X_test = filtered_df['text'].values
    y_test = filtered_df['spdx_id'].values
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=top_labels)
    
    # Normalize by row (true labels) for percentage
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Proportion')
    
    # Set labels
    ax.set(
        xticks=np.arange(len(top_labels)),
        yticks=np.arange(len(top_labels)),
        xticklabels=top_labels,
        yticklabels=top_labels,
        xlabel='Predicted License',
        ylabel='True License',
        title=f'Confusion Matrix (Top {top_n} Licenses)'
    )
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(len(top_labels)):
        for j in range(len(top_labels)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            text = f'{count}\n({pct:.0%})'
            ax.text(
                j, i, text,
                ha='center', va='center',
                color='white' if pct > thresh else 'black',
                fontsize=8
            )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    plt.show()


def predict_license(pipeline: Pipeline, text: str) -> str:
    """
    Predict the license type for a given text.
    
    Args:
        pipeline: The trained classification pipeline.
        text: The license text to classify.
        
    Returns:
        The predicted SPDX license identifier.
    """
    return pipeline.predict([text])[0]


if __name__ == "__main__":
    # Quick test
    from .data_loader import load_dataset
    from .train import split_dataset, train_model
    
    df = load_dataset("data/scancode_licenses/licenses")
    train_df, test_df = split_dataset(df)
    
    pipeline = train_model(train_df)
    
    metrics = evaluate_model(pipeline, test_df)
    plot_confusion_matrix(pipeline, test_df, top_n=10, output_path="outputs/confusion_matrix.png")
