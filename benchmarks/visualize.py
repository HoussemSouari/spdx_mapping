"""
Visualization utilities for benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_accuracy_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Plot accuracy comparison across tools and datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot for grouped bar chart
    pivot_df = results_df.pivot(index='dataset', columns='detector', values='accuracy')
    
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Accuracy Comparison Across Tools and Datasets', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Detector', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_f1_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Plot F1 score comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df = results_df.pivot(index='dataset', columns='detector', values='f1_macro')
    
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('F1-Score (Macro) Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Detector', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_speed_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Plot execution speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to milliseconds per sample
    results_df['ms_per_sample'] = results_df['avg_time_per_sample'] * 1000
    
    pivot_df = results_df.pivot(index='dataset', columns='detector', values='ms_per_sample')
    
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Execution Speed Comparison (Lower is Better)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax.legend(title='Detector', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "speed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_precision_recall(results_df: pd.DataFrame, output_dir: Path):
    """Plot precision vs recall scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for detector in results_df['detector'].unique():
        detector_data = results_df[results_df['detector'] == detector]
        ax.scatter(
            detector_data['recall_macro'],
            detector_data['precision_macro'],
            s=200,
            alpha=0.6,
            label=detector,
            edgecolors='black',
            linewidth=1.5
        )
        
        # Add dataset labels
        for _, row in detector_data.iterrows():
            ax.annotate(
                row['dataset'],
                (row['recall_macro'], row['precision_macro']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    ax.set_title('Precision vs Recall (Macro Average)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Recall (Macro)', fontsize=12)
    ax.set_ylabel('Precision (Macro)', fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Detector', loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    output_path = output_dir / "precision_recall.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_metrics_heatmap(results_df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of all metrics."""
    # Create a unique identifier for each combination
    results_df['combination'] = results_df['detector'] + '\n' + results_df['dataset']
    
    # Select metrics to visualize
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Create heatmap data
    heatmap_data = results_df.set_index('combination')[metrics].T
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title('Performance Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Detector + Dataset', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / "metrics_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_radar_chart(results_df: pd.DataFrame, output_dir: Path):
    """Plot radar chart comparing multiple metrics."""
    from math import pi
    
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Create one radar chart per dataset
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for _, row in dataset_data.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['detector'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True)
        
        ax.set_title(f'Performance Radar Chart - {dataset}', 
                     size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        output_path = output_dir / f"radar_chart_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
        plt.close()


def generate_all_visualizations(results_csv: Path, output_dir: Path):
    """Generate all visualization plots."""
    print("\nGenerating visualizations...")
    
    # Load results
    results_df = pd.read_csv(results_csv)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_accuracy_comparison(results_df, output_dir)
    plot_f1_comparison(results_df, output_dir)
    plot_speed_comparison(results_df, output_dir)
    plot_precision_recall(results_df, output_dir)
    plot_metrics_heatmap(results_df, output_dir)
    plot_radar_chart(results_df, output_dir)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    from benchmarks.config import BENCHMARK_OUTPUT_DIR
    
    results_file = BENCHMARK_OUTPUT_DIR / "benchmark_results.csv"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Run benchmarks first: python benchmarks/run_benchmark.py")
    else:
        generate_all_visualizations(results_file, BENCHMARK_OUTPUT_DIR)
