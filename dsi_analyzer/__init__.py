"""
DSI Document Analyzer Package

A comprehensive package for analyzing document-specific neuron activations
in Differentiable Search Index (DSI) models using TransformerLens.

This package provides tools for:
- Loading and managing TriviaQA datasets
- Extracting and manipulating neuron activations
- Analyzing document-specific neuron behavior
- Visualizing analysis results
- Running systematic neuron manipulation experiments

Main Classes:
    DSIDocumentAnalyzer: Complete analysis pipeline
    TriviaQADataLoader: Data loading and management
    StatisticsManager: Statistics generation and storage
    MultiNeuronHookManager: Hook management for activation manipulation

Main Functions:
    analyze_document_specific_neurons: Core analysis function
    compare_multiple_documents: Multi-document comparison
    load_trivia_qa_data: Convenience data loading function
"""

__version__ = "1.0.0"

# Main classes
from .analyzer import DSIDocumentAnalyzer
from .data_loader import TriviaQADataLoader, QuestionsDataset, load_trivia_qa_data
from .statistics_manager import StatisticsManager
from .hook_manager import (
    ActivationExtractor, NeuronManipulator, MultiNeuronHookManager,
    extract_activated_neurons, run_inference_with_hooks
)

# Core analysis functions
from .analysis_core import (
    analyze_document_specific_neurons,
    compare_multiple_documents,
    collect_document_specific_neurons,
    test_document_neuron_effects,
    save_incorrect_queries_analysis
)

# Visualization functions
from .visualization import (
    plot_frequency_sweep_results,
    plot_activation_distributions,
    create_analysis_summary_table,
    plot_multi_document_comparison,
    analyze_and_visualize_sweep_results
)

# Utility functions
from .utils import (
    create_default_paths,
    safe_file_write,
    safe_file_read,
    get_memory_usage,
    check_memory_availability,
    get_model_device,
    validate_layer_indices,
    validate_frequency_threshold,
    format_accuracy_results,
    print_analysis_summary
)

# Define what gets imported with "from dsi_analyzer import *"
__all__ = [
    # Main classes
    'DSIDocumentAnalyzer',
    'TriviaQADataLoader',
    'QuestionsDataset',
    'StatisticsManager',
    'ActivationExtractor',
    'NeuronManipulator',
    'MultiNeuronHookManager',

    # Core functions
    'analyze_document_specific_neurons',
    'compare_multiple_documents',
    'collect_document_specific_neurons',
    'test_document_neuron_effects',
    'save_incorrect_queries_analysis',
    'extract_activated_neurons',
    'run_inference_with_hooks',
    'load_trivia_qa_data',

    # Visualization functions
    'plot_frequency_sweep_results',
    'plot_activation_distributions',
    'create_analysis_summary_table',
    'plot_multi_document_comparison',
    'analyze_and_visualize_sweep_results',

    # Utility functions
    'create_default_paths',
    'safe_file_write',
    'safe_file_read',
    'get_memory_usage',
    'check_memory_availability',
    'get_model_device',
    'validate_layer_indices',
    'validate_frequency_threshold',
    'format_accuracy_results',
    'print_analysis_summary'
]


# Package-level convenience functions
def create_analyzer(model_path=None, data_path=None, stats_path=None, config_path=None, device='auto'):
    """
    Create a DSIDocumentAnalyzer with default settings.

    Args:
        model_path: Path to DSI model
        data_path: Path to TriviaQA data
        stats_path: Path to statistics files
        config_path: Path to configuration file
        device: Device for computation

    Returns:
        DSIDocumentAnalyzer instance
    """
    return DSIDocumentAnalyzer(
        model_path=model_path,
        data_path=data_path,
        stats_path=stats_path,
        config_path=config_path,
        device=device
    )


def quick_analysis(target_doc_id,
                  model_path=None,
                  data_path=None,
                  stats_path=None,
                  frequency_threshold=0.05,
                  verbose=True):
    """
    Quick analysis of a document with default settings.

    Args:
        target_doc_id: Document ID to analyze
        model_path: Path to DSI model
        data_path: Path to TriviaQA data
        stats_path: Path to statistics files
        frequency_threshold: Frequency threshold for neuron filtering
        verbose: Whether to print detailed output

    Returns:
        Analysis results dictionary
    """
    analyzer = create_analyzer(model_path, data_path, stats_path)

    # Load everything
    analyzer.load_model()
    analyzer.load_data()
    analyzer.load_statistics()

    # Run analysis
    return analyzer.analyze_document(
        target_doc_id=target_doc_id,
        frequency_threshold=frequency_threshold,
        verbose=verbose
    )


# Package metadata
def get_package_info():
    """Get package information."""
    return {
        'name': 'dsi_analyzer',
        'version': __version__,
        'description': 'DSI Document-Specific Neuron Analysis Package',
        'main_classes': len([x for x in __all__ if x.endswith('Analyzer') or x.endswith('Manager') or x.endswith('Loader')]),
        'functions': len([x for x in __all__ if not x.endswith('Analyzer') and not x.endswith('Manager') and not x.endswith('Loader')])
    }