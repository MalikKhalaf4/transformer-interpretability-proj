# Transformer Interpretability Project

Transformer MLP interpretability project for analyzing document-specific neurons in DSI (Differentiable Search Index) models.

## Overview

This project analyzes which MLP neurons in transformer models are specifically activated for certain documents, and measures their causal impact on model predictions through systematic manipulation experiments.

## Quick Start

```bash
# Run analysis on document 47788 with default settings
python run_analysis_and_plots.py --target_docs "47788"

# Run analysis on top 5 popular documents
python run_analysis_and_plots.py

# See all options
python run_analysis_and_plots.py --help
```

## Project Structure

```
transformer-interpretability-proj/
├── dsi_analyzer/                  # Core analysis package
│   ├── analyzer.py               # Main DSIDocumentAnalyzer class
│   ├── analysis_core.py          # Core analysis functions
│   ├── data_loader.py            # TriviaQA data loading
│   ├── hook_manager.py           # Activation extraction/manipulation
│   ├── statistics_manager.py     # Statistics generation
│   ├── test_set_generator.py     # Test set utilities
│   ├── utils.py                  # Helper functions
│   └── visualization.py          # Plotting functions
├── run_analysis_and_plots.py     # Main analysis script
└── data/                         # Data and statistics directory
    ├── activated_neurons.json
    ├── neuron_activation_stats.json
    └── detailed_activation_stats.json
```

## Key Features

- **Document-Specific Neuron Detection**: Identify neurons that activate primarily for specific documents
- **Frequency-Based Filtering**: Filter neurons by activation frequency across queries
- **Causal Impact Testing**: Measure effect of neuron manipulation on model predictions
- **Comprehensive Visualization**: Generate plots and tables for analysis results
- **Batch Processing**: Optimized inference with configurable batch sizes

## Default Paths

The package expects the following directory structure:

```
interpretability/                    # Parent directory
├── DSI-large-TriviaQA/             # Model directory
├── TriviaQAData/                   # Dataset directory
└── transformer-interpretability-proj/
    └── data/                       # Statistics and results
        ├── activated_neurons.json
        ├── neuron_activation_stats.json
        └── results/                # Analysis output
```

You can override these paths using the `DSIDocumentAnalyzer` constructor parameters.

## Usage Examples

### Python API

```python
from dsi_analyzer import DSIDocumentAnalyzer

# Create analyzer with default paths
analyzer = DSIDocumentAnalyzer()

# Load model, data, and statistics
analyzer.load_model()
analyzer.load_data()
analyzer.load_statistics()

# Analyze a specific document
results = analyzer.analyze_document(
    target_doc_id=47788,
    frequency_threshold=0.05,
    other_queries_sample=100
)

# Run frequency sweep analysis
sweep_results = analyzer.frequency_sweep_analysis(
    target_doc_id=47788,
    frequency_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5],
    replacement_types=['zero_out']
)

# Visualize results
analyzer.visualize_results(sweep_results, plot_type='frequency_sweep')
```

### Command Line

```bash
# Custom document analysis with specific thresholds
python run_analysis_and_plots.py \
  --target_docs "47788,26530,12345" \
  --min_freq 0.05 \
  --max_freq 0.3 \
  --freq_step 0.05 \
  --replacement_types "zero_out" \
  --other_queries_sample 1000

# Quick test run
python run_analysis_and_plots.py \
  --target_docs "47788" \
  --replacement_types "zero_out" \
  --other_queries_sample 100 \
  --dry_run

# Visualization-only mode from existing results
python run_analysis_and_plots.py \
  --visualize_from data/results/neuron_analysis_20250927_183322/complete_analysis_results.json
```

## Output Format

Analysis results are saved to `data/results/neuron_analysis_{timestamp}/`:

```
neuron_analysis_20250927_183322/
├── complete_analysis_results.json           # Full analysis data
├── sweep_results_detailed_TIMESTAMP.csv     # Tabular results
├── sweep_analysis_visualization_TIMESTAMP.png  # Combined plots
├── target_exact_TIMESTAMP.png               # Individual plot: Target exact accuracy
├── target_relevant_TIMESTAMP.png            # Individual plot: Target relevant docs
├── other_exact_TIMESTAMP.png                # Individual plot: Other exact accuracy
├── other_relevant_TIMESTAMP.png             # Individual plot: Other relevant docs
└── specificity_TIMESTAMP.png                # Individual plot: Specificity scores
```

### Output Files Description

- **complete_analysis_results.json**: Contains all analysis data including neuron lists, accuracy metrics, and execution parameters
- **sweep_results_detailed_*.csv**: CSV table with columns for each threshold/replacement combination
- **sweep_analysis_visualization_*.png**: Combined 6-panel visualization showing all metrics
- **Individual plots**: Separate high-resolution plots for each metric

## Prerequisites

1. **Model and Data**:
   - DSI model in `../DSI-large-TriviaQA/`
   - TriviaQA data in `../TriviaQAData/`

2. **Statistics Files** (in `data/`):
   - `activated_neurons.json`: Query-level neuron activations
   - `neuron_activation_stats.json`: Global neuron frequency statistics
   - `detailed_activation_stats.json`: Mean activation values (optional, for mean_value replacement)

3. **System Requirements**:
   - Python 3.8+
   - PyTorch with CUDA support (recommended)
   - 15GB+ available memory for full analysis

## Technical Details

### Analysis Pipeline

1. **Neuron Collection**: Identify neurons activated for target document queries
2. **Frequency Filtering**: Filter out globally active neurons based on threshold
3. **Manipulation Testing**: Zero out or replace neuron activations during inference
4. **Impact Measurement**: Compare accuracy before/after manipulation
5. **Side Effect Analysis**: Test impact on other documents

### Frequency Threshold Logic

- Threshold filters neurons by how often they activate across ALL queries
- Lower threshold = more aggressive filtering = more document-specific neurons
- Example: `threshold=0.1` keeps neurons activated in <10% of queries

### Replacement Strategies

- **zero_out**: Set neuron activations to zero
- **mean_value**: Replace with layer-wise mean activations (requires detailed stats)
