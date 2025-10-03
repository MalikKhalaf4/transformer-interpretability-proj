"""
Visualization functions for DSI document analysis results.

This module provides plotting and visualization functions for analyzing
neuron manipulation results, frequency sweeps, and activation statistics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import os
from datetime import datetime

from .utils import create_default_paths


def plot_frequency_sweep_results(sweep_results: Dict[str, Any],
                                title: str = "Frequency Threshold Sweep Analysis",
                                save_path: Optional[Union[str, Path]] = None,
                                show_plot: bool = True) -> plt.Figure:
    """
    Plot the results of frequency threshold sweep analysis.

    Args:
        sweep_results: Results from frequency threshold sweep
        title: Title for the plot
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Extract data
    thresholds = sweep_results.get('threshold_percentages', [])
    n_neurons = sweep_results.get('n_neurons_filtered', [])
    target_drops = sweep_results.get('target_accuracy_drop', [])
    other_drops = sweep_results.get('other_accuracy_drop', [])

    # Plot 1: Number of neurons vs threshold
    axes[0, 0].plot(thresholds, n_neurons, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Frequency Threshold (%)')
    axes[0, 0].set_ylabel('Number of Filtered Neurons')
    axes[0, 0].set_title('Neurons vs Frequency Threshold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Target accuracy drop vs threshold
    axes[0, 1].plot(thresholds, target_drops, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Frequency Threshold (%)')
    axes[0, 1].set_ylabel('Target Accuracy Drop (%)')
    axes[0, 1].set_title('Target Document Impact')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Other accuracy drop vs threshold
    axes[1, 0].plot(thresholds, other_drops, 'o-', color='orange', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Frequency Threshold (%)')
    axes[1, 0].set_ylabel('Other Documents Accuracy Drop (%)')
    axes[1, 0].set_title('Side Effects on Other Documents')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Specificity score (target drop / other drop)
    specificity_scores = []
    for target, other in zip(target_drops, other_drops):
        if other > 0:
            specificity_scores.append(target / other)
        else:
            specificity_scores.append(target if target > 0 else 0)

    axes[1, 1].plot(thresholds, specificity_scores, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Frequency Threshold (%)')
    axes[1, 1].set_ylabel('Specificity Score (Target/Other)')
    axes[1, 1].set_title('Neuron Specificity Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_activation_distributions(activation_stats: Dict[str, Dict[str, Any]],
                                layer_keys: Optional[List[str]] = None,
                                title: str = "Activation Statistics by Layer",
                                save_path: Optional[Union[str, Path]] = None,
                                show_plot: bool = True) -> plt.Figure:
    """
    Plot activation distribution statistics by layer.

    Args:
        activation_stats: Statistics with mean/variance data by layer
        layer_keys: Specific layers to plot, or None for all
        title: Title for the plot
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    if layer_keys is None:
        layer_keys = list(activation_stats.keys())

    n_layers = len(layer_keys)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * n_layers, 8))
    if n_layers == 1:
        axes = [axes]
    elif n_layers <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, layer_key in enumerate(layer_keys):
        if i >= len(axes):
            break

        layer_stats = activation_stats.get(layer_key, {})
        means = list(layer_stats.get('mean', {}).values())
        stds = list(layer_stats.get('std', {}).values())

        if means and stds:
            # Plot mean activations
            axes[i].hist(means, bins=50, alpha=0.7, label='Means', color='blue')
            axes[i].axvline(np.mean(means), color='red', linestyle='--',
                          label=f'Avg Mean: {np.mean(means):.3f}')

            # Plot standard deviations as secondary axis
            ax2 = axes[i].twinx()
            ax2.hist(stds, bins=50, alpha=0.5, label='Std Devs', color='orange')
            ax2.axvline(np.mean(stds), color='green', linestyle='--',
                       label=f'Avg Std: {np.mean(stds):.3f}')

            axes[i].set_title(f'{layer_key.replace("_", " ").title()}')
            axes[i].set_xlabel('Activation Value')
            axes[i].set_ylabel('Count (Means)', color='blue')
            ax2.set_ylabel('Count (Std Devs)', color='orange')

            # Combine legends
            lines1, labels1 = axes[i].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Hide extra subplots
    for i in range(len(layer_keys), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def create_analysis_summary_table(results: Dict[str, Any],
                                save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Create a summary table of analysis results.

    Args:
        results: Analysis results dictionary
        save_path: Path to save the table as CSV

    Returns:
        Pandas DataFrame with summary statistics
    """
    # Extract key metrics
    summary_data = []

    if isinstance(results, dict) and 'test_results' in results:
        # Single document analysis
        result = results
        doc_id = result.get('target_doc_id', 'Unknown')
        neuron_data = result.get('neuron_data', {})
        test_results = result.get('test_results', {})
        summary = test_results.get('summary', {})

        n_queries = len(neuron_data.get('target_queries', []))
        n_neurons = sum(len(neurons) for neurons in neuron_data.get('filtered_neurons', {}).values())

        summary_data.append({
            'Document_ID': doc_id,
            'Target_Queries': n_queries,
            'Filtered_Neurons': n_neurons,
            'Target_Accuracy_Before': summary.get('target_accuracy_before', 0),
            'Target_Accuracy_After': summary.get('target_accuracy_after', 0),
            'Target_Accuracy_Drop': summary.get('target_accuracy_drop', 0),
            'Other_Accuracy_Before': summary.get('other_accuracy_before', 0),
            'Other_Accuracy_After': summary.get('other_accuracy_after', 0),
            'Other_Accuracy_Drop': summary.get('other_accuracy_drop', 0),
            'Target_Queries_Changed': test_results.get('queries_changed', {}).get('n_target_changed', 0),
            'Other_Queries_Changed': test_results.get('queries_changed', {}).get('n_other_changed', 0),
            'Specificity_Score': (summary.get('target_accuracy_drop', 0) /
                                 (summary.get('other_accuracy_drop', 0) + 1e-6))
        })

    elif isinstance(results, dict):
        # Multiple document comparison
        for doc_id, result in results.items():
            if result is None:
                continue

            neuron_data = result.get('neuron_data', {})
            test_results = result.get('test_results', {})
            summary = test_results.get('summary', {})

            n_queries = len(neuron_data.get('target_queries', []))
            n_neurons = sum(len(neurons) for neurons in neuron_data.get('filtered_neurons', {}).values())

            summary_data.append({
                'Document_ID': doc_id,
                'Target_Queries': n_queries,
                'Filtered_Neurons': n_neurons,
                'Target_Accuracy_Before': summary.get('target_accuracy_before', 0),
                'Target_Accuracy_After': summary.get('target_accuracy_after', 0),
                'Target_Accuracy_Drop': summary.get('target_accuracy_drop', 0),
                'Other_Accuracy_Before': summary.get('other_accuracy_before', 0),
                'Other_Accuracy_After': summary.get('other_accuracy_after', 0),
                'Other_Accuracy_Drop': summary.get('other_accuracy_drop', 0),
                'Target_Queries_Changed': test_results.get('queries_changed', {}).get('n_target_changed', 0),
                'Other_Queries_Changed': test_results.get('queries_changed', {}).get('n_other_changed', 0),
                'Specificity_Score': (summary.get('target_accuracy_drop', 0) /
                                     (summary.get('other_accuracy_drop', 0) + 1e-6))
            })

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to {save_path}")

    return df


def plot_multi_document_comparison(comparison_results: Dict[str, Any],
                                 title: str = "Multi-Document Neuron Specificity Comparison",
                                 save_path: Optional[Union[str, Path]] = None,
                                 show_plot: bool = True) -> plt.Figure:
    """
    Plot comparison results across multiple documents.

    Args:
        comparison_results: Results from compare_multiple_documents
        title: Title for the plot
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    # Create summary table
    df = create_analysis_summary_table(comparison_results)

    if df.empty:
        print("No data to plot")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Target vs Other accuracy drops
    axes[0, 0].scatter(df['Target_Accuracy_Drop'], df['Other_Accuracy_Drop'],
                      s=100, alpha=0.7, c=df['Specificity_Score'], cmap='viridis')
    axes[0, 0].set_xlabel('Target Accuracy Drop (%)')
    axes[0, 0].set_ylabel('Other Accuracy Drop (%)')
    axes[0, 0].set_title('Target vs Other Document Impact')
    axes[0, 0].grid(True, alpha=0.3)

    # Add diagonal line for reference
    max_val = max(df['Target_Accuracy_Drop'].max(), df['Other_Accuracy_Drop'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Impact')
    axes[0, 0].legend()

    # Plot 2: Number of neurons vs specificity
    axes[0, 1].scatter(df['Filtered_Neurons'], df['Specificity_Score'], s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Number of Filtered Neurons')
    axes[0, 1].set_ylabel('Specificity Score')
    axes[0, 1].set_title('Neurons vs Specificity')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Bar chart of target accuracy drops
    doc_ids = df['Document_ID'].astype(str)
    axes[1, 0].bar(range(len(doc_ids)), df['Target_Accuracy_Drop'], alpha=0.7)
    axes[1, 0].set_xlabel('Document ID')
    axes[1, 0].set_ylabel('Target Accuracy Drop (%)')
    axes[1, 0].set_title('Target Accuracy Drop by Document')
    axes[1, 0].set_xticks(range(len(doc_ids)))
    axes[1, 0].set_xticklabels(doc_ids, rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Specificity scores
    axes[1, 1].bar(range(len(doc_ids)), df['Specificity_Score'], alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Document ID')
    axes[1, 1].set_ylabel('Specificity Score')
    axes[1, 1].set_title('Neuron Specificity by Document')
    axes[1, 1].set_xticks(range(len(doc_ids)))
    axes[1, 1].set_xticklabels(doc_ids, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def analyze_and_visualize_sweep_results(sweep_results: Dict[str, Any],
                                      output_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Comprehensive analysis and visualization of sweep results with enhanced styling and relevant docs accuracy.

    Args:
        sweep_results: Dictionary containing the sweep analysis results
        output_dir: Directory to save outputs

    Returns:
        Tuple of (DataFrame with detailed results, matplotlib figure)
    """
    # Set up enhanced plotting style with seaborn
    sns.set_theme(context='paper', style='whitegrid', palette='deep')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'figure.dpi': 300
    })

    # Define professional color palette for paper
    colors = {
        'primary': '#1f77b4',      # Strong blue
        'secondary': '#d62728',    # Strong red
        'tertiary': '#2ca02c',     # Strong green
        'quaternary': '#ff7f0e',   # Strong orange
        'accent': '#9467bd',       # Purple
        'neutral': '#8c564b'       # Brown
    }

    print(" Creating comprehensive analysis tables and visualizations...")

    # Create output directory
    if output_dir is None:
        paths = create_default_paths()
        output_dir = paths['results_dir'] / "neuron_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Created output directory: {output_dir}")

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f" Timestamp: {timestamp}")

    # === 1. CREATE COMPREHENSIVE RESULTS TABLE ===
    print("\n Creating detailed results table...")

    table_data = []
    for doc_id, doc_data in sweep_results['documents'].items():
        for freq, freq_data in doc_data['frequencies'].items():
            for replacement_type, result in freq_data.items():
                if 'error' not in result:
                    # Extract relevant docs accuracy data (with fallback for backwards compatibility)
                    target_rel_queries = result.get('target_relevant_queries', {})
                    other_rel_queries = result.get('other_relevant_queries', {})

                    table_data.append({
                        'Document_ID': doc_id,
                        'Frequency': freq,
                        'Replacement_Type': replacement_type,
                        'Target_Queries_Total': result['target_queries']['total'],
                        'Target_Accuracy_Before': result['target_queries']['accuracy_before'],
                        'Target_Accuracy_After': result['target_queries']['accuracy_after'],
                        'Target_Accuracy_Drop': result['target_queries']['accuracy_drop'],
                        'Target_Relevant_Accuracy_Before': target_rel_queries.get('accuracy_before', 0),
                        'Target_Relevant_Accuracy_After': target_rel_queries.get('accuracy_after', 0),
                        'Target_Relevant_Accuracy_Drop': target_rel_queries.get('accuracy_drop', 0),
                        'Other_Queries_Total': result['other_queries']['total'],
                        'Other_Accuracy_Before': result['other_queries']['accuracy_before'],
                        'Other_Accuracy_After': result['other_queries']['accuracy_after'],
                        'Other_Accuracy_Drop': result['other_queries']['accuracy_drop'],
                        'Other_Relevant_Accuracy_Before': other_rel_queries.get('accuracy_before', 0),
                        'Other_Relevant_Accuracy_After': other_rel_queries.get('accuracy_after', 0),
                        'Other_Relevant_Accuracy_Drop': other_rel_queries.get('accuracy_drop', 0),
                        'Overall_Accuracy_Before': result['overall']['accuracy_before'],
                        'Overall_Accuracy_After': result['overall']['accuracy_after'],
                        'Overall_Accuracy_Drop': result['overall']['accuracy_drop'],
                        'Execution_Time': result['execution_time']
                    })

    df = pd.DataFrame(table_data)

    # Save detailed table
    table_path = output_dir / f"sweep_results_detailed_{timestamp}.csv"
    df.to_csv(table_path, index=False)
    print(f" Saved detailed results table: {table_path}")

    # === 2. CREATE ENHANCED VISUALIZATIONS ===
    print("\n Creating enhanced visualizations with relevant docs accuracy...")

    # Prepare data for plotting
    documents = df['Document_ID'].unique()
    replacement_types = df['Replacement_Type'].unique()
    doc_colors = sns.color_palette('deep', len(documents))

    # Helper function to create individual plots
    def create_individual_plot(plot_type, save_separate=True):
        """Create individual plot and optionally save as separate PNG"""
        fig_single, ax = plt.subplots(1, 1, figsize=(10, 6))

        if plot_type == 'neurons_vs_threshold':
            # This would need neuron count data - for now, we'll skip this plot
            # since the current data structure doesn't include neuron counts per frequency
            ax.text(0.5, 0.5, 'Neuron count data not available in current format',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Neurons vs Frequency Threshold')

        elif plot_type == 'target_exact':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Target_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2.5, markersize=6, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Target Accuracy Drop (%)')
            ax.set_title('Target Document Impact - Exact Accuracy')
            ax.legend()

        elif plot_type == 'target_relevant':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Target_Relevant_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2.5, markersize=6, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Target Relevant-Docs Accuracy Drop (%)')
            ax.set_title('Target Document Impact - Relevant Docs Accuracy')
            ax.legend()

        elif plot_type == 'other_exact':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Other_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2.5, markersize=6, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Other Documents Accuracy Drop (%)')
            ax.set_title('Side Effects on Other Documents - Exact Accuracy')
            ax.legend()

        elif plot_type == 'other_relevant':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Other_Relevant_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2.5, markersize=6, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Other Documents Relevant-Docs Accuracy Drop (%)')
            ax.set_title('Side Effects on Other Documents - Relevant Docs Accuracy')
            ax.legend()

        elif plot_type == 'specificity':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    doc_data = doc_data.copy()
                    specificity = (doc_data['Target_Accuracy_Drop'] / (doc_data['Other_Accuracy_Drop'] + 1e-6)).replace([np.inf], 0)
                    ax.plot(doc_data['Frequency'].astype(float), specificity,
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2.5, markersize=6, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Specificity Score (Target/Other)')
            ax.set_title('Neuron Specificity Analysis')
            ax.legend()

        # Apply consistent styling
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save individual plot if requested
        if save_separate:
            individual_path = output_dir / f"{plot_type}_{timestamp}.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f" Saved {plot_type}: {individual_path}")

        return fig_single, ax

    # Create main combined figure
    print(" Creating combined visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Document Frequency Sweep Analysis Results", fontsize=16, fontweight='bold')

    # Create individual plots and extract their content for the combined figure
    plot_configs = [
        ('neurons_vs_threshold', (0, 0)),
        ('target_exact', (0, 1)),
        ('target_relevant', (0, 2)),
        ('other_exact', (1, 0)),
        ('other_relevant', (1, 1)),
        ('specificity', (1, 2))
    ]

    individual_figures = []
    for plot_type, (row, col) in plot_configs:
        print(f"   Creating {plot_type} plot...")

        # Create individual plot and save
        fig_single, ax_single = create_individual_plot(plot_type, save_separate=True)
        individual_figures.append(fig_single)

        # Make sure we're working with the combined figure
        plt.figure(fig.number)

        # Copy to combined figure
        ax = axes[row, col]

        if plot_type == 'neurons_vs_threshold':
            ax.text(0.5, 0.5, 'Neuron count data not available\nin current format',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Neurons vs Frequency Threshold')

        elif plot_type == 'target_exact':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Target_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2, markersize=5, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Target Accuracy Drop (%)')
            ax.set_title('Target Impact - Exact')
            if len(documents) <= 10:  # Only show legend if not too many documents
                ax.legend(fontsize=8)

        elif plot_type == 'target_relevant':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Target_Relevant_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2, markersize=5, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Target Relevant-Docs Drop (%)')
            ax.set_title('Target Impact - Relevant Docs')
            if len(documents) <= 10:
                ax.legend(fontsize=8)

        elif plot_type == 'other_exact':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Other_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2, markersize=5, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Other Accuracy Drop (%)')
            ax.set_title('Side Effects - Exact')
            if len(documents) <= 10:
                ax.legend(fontsize=8)

        elif plot_type == 'other_relevant':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    ax.plot(doc_data['Frequency'].astype(float), doc_data['Other_Relevant_Accuracy_Drop'],
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2, markersize=5, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Other Relevant-Docs Drop (%)')
            ax.set_title('Side Effects - Relevant Docs')
            if len(documents) <= 10:
                ax.legend(fontsize=8)

        elif plot_type == 'specificity':
            for j, doc_id in enumerate(documents):
                doc_data = df[df['Document_ID'] == doc_id].sort_values('Frequency')
                if len(doc_data) > 0:
                    doc_data = doc_data.copy()
                    specificity = (doc_data['Target_Accuracy_Drop'] / (doc_data['Other_Accuracy_Drop'] + 1e-6)).replace([np.inf], 0)
                    ax.plot(doc_data['Frequency'].astype(float), specificity,
                           'o-', label=f'Doc {doc_id}', color=doc_colors[j], linewidth=2, markersize=5, alpha=0.9)
            ax.set_xlabel('Frequency Threshold')
            ax.set_ylabel('Specificity Score')
            ax.set_title('Neuron Specificity')
            if len(documents) <= 10:
                ax.legend(fontsize=8)

        # Apply styling to combined plot
        ax.grid(True, alpha=0.3)

    # Make sure we're working with the combined figure for final operations
    plt.figure(fig.number)
    plt.tight_layout()

    # Save combined visualization (explicitly save the main figure)
    viz_path = output_dir / f"sweep_analysis_visualization_{timestamp}.png"
    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f" Saved combined visualization: {viz_path}")

    # Close individual figures to save memory
    for fig_single in individual_figures:
        plt.close(fig_single)

    print(f"\n Analysis complete! All outputs saved to: {output_dir}")

    return df, fig