#!/usr/bin/env python3
"""
FOSS License Classification System

This is the main entry point for the license classification project.
It orchestrates the full pipeline: data loading, preprocessing,
training, evaluation, and visualization.

Usage:
    python main.py [--data-dir PATH] [--output-dir PATH]

Example:
    python main.py --data-dir data/scancode_licenses/licenses
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, get_label_distribution
from src.train import create_pipeline, split_dataset, train_model, save_model
from src.evaluate import evaluate_model, plot_confusion_matrix, predict_license


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FOSS License Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train and evaluate with default paths
    python main.py

    # Specify custom data directory
    python main.py --data-dir /path/to/licenses

    # Save outputs to custom directory
    python main.py --output-dir results/
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/scancode_licenses',
        help='Path to the ScanCode license dataset directory (containing licenses/ and rules/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save model and outputs'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top licenses for confusion matrix (default: 10)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting confusion matrix'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("FOSS LICENSE CLASSIFICATION SYSTEM")
    print("="*60 + "\n")
    
    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================
    print("Step 1: Loading dataset...")
    print("-"*40)
    
    try:
        df = load_dataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the ScanCode license dataset:")
        print("  git clone --depth 1 https://github.com/nexB/scancode-toolkit.git temp_scancode")
        print("  mkdir -p data/scancode_licenses")
        print("  cp -r temp_scancode/src/licensedcode/data/licenses data/scancode_licenses/")
        print("  rm -rf temp_scancode")
        sys.exit(1)
    
    if len(df) == 0:
        print("Error: No valid licenses found in dataset!")
        sys.exit(1)
    
    print("\nDataset summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique licenses: {df['spdx_id'].nunique()}")
    
    print(f"\nTop {args.top_n} most frequent licenses:")
    for i, (label, count) in enumerate(get_label_distribution(df, args.top_n).items(), 1):
        print(f"  {i:2d}. {label}: {count}")
    
    # =========================================================================
    # Step 2: Split Dataset
    # =========================================================================
    print("\n" + "="*60)
    print("Step 2: Splitting dataset (stratified)...")
    print("-"*40)
    
    train_df, test_df = split_dataset(df, test_size=args.test_size)
    
    # =========================================================================
    # Step 3: Train Model
    # =========================================================================
    print("\n" + "="*60)
    print("Step 3: Training TF-IDF + LinearSVC model...")
    print("-"*40)
    
    pipeline = create_pipeline()
    pipeline = train_model(train_df, pipeline)
    
    # Save the trained model
    model_path = output_dir / 'license_classifier.pkl'
    save_model(pipeline, str(model_path))
    
    # =========================================================================
    # Step 4: Evaluate Model
    # =========================================================================
    print("\n" + "="*60)
    print("Step 4: Evaluating model...")
    print("-"*40)
    
    metrics = evaluate_model(pipeline, test_df, print_report=False)
    
    # Save metrics to file
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("FOSS License Classification - Evaluation Metrics\n")
        f.write("="*50 + "\n\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    print(f"Metrics saved to {metrics_path}")
    
    # =========================================================================
    # Step 5: Plot Confusion Matrix
    # =========================================================================
    if not args.no_plot:
        print("\n" + "="*60)
        print(f"Step 5: Plotting confusion matrix (top {args.top_n} licenses)...")
        print("-"*40)
        
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(
            pipeline, test_df,
            top_n=args.top_n,
            output_path=str(cm_path)
        )
    
    # =========================================================================
    # Demo: Predict a sample license
    # =========================================================================
    print("\n" + "="*60)
    print("Demo: Sample prediction")
    print("-"*40)
    
    # Get a random sample from test set
    sample = test_df.sample(1).iloc[0]
    prediction = predict_license(pipeline, sample['text'])
    
    print(f"True license: {sample['spdx_id']}")
    print(f"Predicted:    {prediction}")
    print(f"Correct:      {'✓' if prediction == sample['spdx_id'] else '✗'}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("  - license_classifier.pkl (trained model)")
    print("  - metrics.txt (evaluation metrics)")
    if not args.no_plot:
        print("  - confusion_matrix.png (visualization)")
    
    print("\nTo use the model for prediction:")
    print("  from src.train import load_model")
    print("  from src.evaluate import predict_license")
    print("  model = load_model('outputs/license_classifier.pkl')")
    print("  result = predict_license(model, your_license_text)")


if __name__ == '__main__':
    main()
