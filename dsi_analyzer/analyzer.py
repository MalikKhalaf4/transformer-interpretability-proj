"""
Main DSI Document Analyzer class.

This module provides the comprehensive DSIDocumentAnalyzer class that
integrates all functionality for document-specific neuron analysis.
"""

import json
import torch
from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .utils import (
    create_default_paths, safe_file_read, get_device_info,
    check_memory_availability, print_analysis_summary
)
from .data_loader import TriviaQADataLoader, load_trivia_qa_data
from .statistics_manager import StatisticsManager
from .analysis_core import (
    analyze_document_specific_neurons, compare_multiple_documents,
    collect_document_specific_neurons, test_document_neuron_effects
)
from .visualization import (
    plot_frequency_sweep_results, plot_activation_distributions,
    create_analysis_summary_table, plot_multi_document_comparison,
    analyze_and_visualize_sweep_results
)
from .hook_manager import MultiNeuronHookManager, run_inference_with_hooks


class DSIDocumentAnalyzer:
    """
    Comprehensive analyzer for document-specific neuron analysis in DSI models.

    This class integrates all functionality from the notebook into a cohesive
    interface for analyzing which neurons are specifically activated for
    certain documents and testing their causal impact.
    """

    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 data_path: Optional[Union[str, Path]] = None,
                 stats_path: Optional[Union[str, Path]] = None,
                 config_path: Optional[Union[str, Path]] = None,
                 device: str = 'auto'):
        """
        Initialize the DSI Document Analyzer.

        Args:
            model_path: Path to DSI model, or None for default
            data_path: Path to TriviaQA data, or None for default
            stats_path: Path to statistics files, or None for default
            config_path: Path to config file, or None for no config
            device: Device for model ('auto', 'cuda', 'cpu')
        """
        # Set up paths
        self.paths = create_default_paths()

        if model_path is None:
            model_path = self.paths['model_path']
        if data_path is None:
            data_path = self.paths['trivia_qa_path']
        if stats_path is None:
            stats_path = self.paths['data_dir']

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.stats_path = Path(stats_path)

        # Load config if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            self.config = safe_file_read(config_path)

        # Set device
        if device == 'auto':
            self.device = utils.get_device()
        else:
            self.device = device

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.hf_model = None

        self.data_loader = TriviaQADataLoader(self.data_path)
        self.stats_manager = StatisticsManager(self.stats_path)
        self.hook_manager = MultiNeuronHookManager()

        # Data storage
        self.result_dict = None
        self.test_result_dict = None
        self.neuron_stats = None
        self.detailed_activation_stats = None
        self.popular_docs = None

        print(f"DSI Document Analyzer initialized")
        print(f"Model path: {self.model_path}")
        print(f"Data path: {self.data_path}")
        print(f"Stats path: {self.stats_path}")
        print(f"Device: {self.device}")

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load the DSI model with TransformerLens integration.

        Args:
            model_path: Path to model checkpoint, or None to use instance path
        """
        if model_path:
            self.model_path = Path(model_path)

        print(f"Loading model from {self.model_path}")

        # Check memory
        check_memory_availability(15.0)  # Require 15GB for model loading

        try:
            # Load HuggingFace model
            self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path)).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            # Load with TransformerLens
            self.model = HookedEncoderDecoder.from_pretrained(
                str(self.model_path),
                hf_model=self.hf_model,
                device=self.device
            )

            print(f"Model loaded successfully")
            print(f"Device info: {get_device_info()}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_data(self,
                  include_test: bool = False,
                  train_file: str = "train_queries_trivia_qa.json",
                  val_file: str = "val_queries_trivia_qa.json",
                  test_file: str = "test_queries_trivia_qa.json") -> Dict[str, Any]:
        """
        Load TriviaQA training, validation, and optionally test data.

        Args:
            include_test: Whether to load test data
            train_file: Training data filename
            val_file: Validation data filename
            test_file: Test data filename

        Returns:
            Dictionary with data statistics
        """
        print(f"Loading data from {self.data_path}")

        try:
            # Load all data splits
            data_splits = self.data_loader.load_all_data(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                load_test=include_test
            )

            # Get combined data
            self.data_loader.get_combined_data(include_test=include_test)

            # Create result dict for analysis
            self.result_dict = self.data_loader.create_result_dict()

            # Get statistics
            stats = self.data_loader.get_data_statistics()

            print(f"Data loaded successfully")
            print(f"Train: {stats.get('train_size', 0)} queries")
            print(f"Val: {stats.get('val_size', 0)} queries")
            if include_test:
                print(f"Test: {stats.get('test_size', 0)} queries")
            print(f"Total: {stats.get('combined_size', 0)} queries")
            print(f"Unique documents: {stats.get('unique_documents', 0)}")

            return stats

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def load_statistics(self,
                       result_dict_path: Optional[Union[str, Path]] = None,
                       neuron_stats_path: Optional[Union[str, Path]] = None,
                       detailed_stats_path: Optional[Union[str, Path]] = None,
                       regenerate: bool = False,
                       use_test_set: bool = False,
                       test_data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load or generate neuron activation statistics.

        Args:
            result_dict_path: Path to result dictionary file
            neuron_stats_path: Path to neuron statistics file
            detailed_stats_path: Path to detailed activation stats file
            regenerate: Whether to regenerate statistics instead of loading
            use_test_set: Whether to load test set data for evaluation (training data still used for neuron filtering)
            test_data_path: Path to test set data file

        Returns:
            Dictionary with loaded statistics info
        """
        if regenerate:
            return self.generate_statistics()

        print(f" Loading statistics from {self.stats_path}")

        try:
            # Always load training data for neuron filtering
            if result_dict_path:
                self.result_dict = self.stats_manager.load_result_dict(result_dict_path, use_test_set=False)
            else:
                self.result_dict = self.stats_manager.load_result_dict(use_test_set=False)

            # If test mode, also load test data for evaluation
            if use_test_set:
                if test_data_path:
                    self.test_result_dict = self.stats_manager.load_result_dict(test_data_path, use_test_set=True)
                else:
                    self.test_result_dict = self.stats_manager.load_result_dict(use_test_set=True)
                print(f" Loaded test set data: {len(self.test_result_dict)} queries")
            else:
                self.test_result_dict = None

            # Load neuron statistics
            if neuron_stats_path:
                self.neuron_stats = self.stats_manager.load_neuron_stats(neuron_stats_path)
            else:
                self.neuron_stats = self.stats_manager.load_neuron_stats()

            # Try to load detailed stats
            try:
                if detailed_stats_path:
                    self.detailed_activation_stats = self.stats_manager.load_detailed_activation_stats(detailed_stats_path)
                else:
                    self.detailed_activation_stats = self.stats_manager.load_detailed_activation_stats()
            except FileNotFoundError:
                print(" Detailed activation stats not found - will limit to zero_out replacement")
                self.detailed_activation_stats = None

            # Create popular docs list
            from .data_loader import create_popular_docs_list
            self.popular_docs = create_popular_docs_list(self.result_dict)

            print(f" Statistics loaded successfully")
            print(f" Total queries in result_dict: {len(self.result_dict)}")
            if self.test_result_dict:
                print(f" Total queries in test_result_dict: {len(self.test_result_dict)}")
            print(f" Layers in neuron_stats: {len(self.neuron_stats)}")
            print(f" Top documents: {self.popular_docs[:5] if self.popular_docs else 'None'}")

            return {
                'result_dict_size': len(self.result_dict),
                'test_result_dict_size': len(self.test_result_dict) if self.test_result_dict else 0,
                'neuron_stats_layers': len(self.neuron_stats),
                'detailed_stats_available': self.detailed_activation_stats is not None,
                'top_documents': self.popular_docs[:10] if self.popular_docs else [],
                'test_mode': use_test_set
            }

        except Exception as e:
            print(f" Error loading statistics: {e}")
            raise

    def generate_statistics(self,
                          layer_indices: Optional[List[int]] = None,
                          batch_size: int = 16,
                          save_results: bool = True,
                          generate_detailed: bool = True,
                          detailed_sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate neuron activation statistics from loaded data.

        Args:
            layer_indices: Layers to analyze, or None for default (17-23)
            batch_size: Batch size for data processing
            save_results: Whether to save generated statistics
            generate_detailed: Whether to generate detailed activation stats
            detailed_sample_size: Limit for detailed stats generation

        Returns:
            Dictionary with generation results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before generating statistics. Call load_model() first.")

        if self.data_loader._combined_data is None:
            raise ValueError("Data must be loaded before generating statistics. Call load_data() first.")

        print(f" Generating activation statistics...")

        # Create data loader
        dataset = self.data_loader.create_dataset()
        dataloader = self.data_loader.create_dataloader(dataset, batch_size=batch_size)

        # Generate main statistics
        stats_result = self.stats_manager.generate_activation_statistics(
            self.model, self.tokenizer, dataloader, layer_indices,
            save_path=self.stats_path / "generated_stats.json" if save_results else None
        )

        self.result_dict = stats_result['result_dict']
        self.neuron_stats = stats_result['stats']

        # Generate detailed statistics if requested
        if generate_detailed:
            print(" Generating detailed activation statistics...")
            self.detailed_activation_stats = self.stats_manager.generate_detailed_activation_stats(
                self.model, self.tokenizer, dataloader, layer_indices,
                save_path=self.stats_path / "detailed_stats.json" if save_results else None,
                sample_size=detailed_sample_size
            )

        # Create popular docs list
        from .data_loader import create_popular_docs_list
        self.popular_docs = create_popular_docs_list(self.result_dict)

        print(f" Statistics generation completed")

        return {
            'total_queries_processed': stats_result['total_queries'],
            'layers_analyzed': len(self.neuron_stats),
            'detailed_stats_generated': generate_detailed,
            'top_documents': self.popular_docs[:10] if self.popular_docs else []
        }

    def analyze_document(self,
                        target_doc_id: Union[int, str],
                        layer_indices: Optional[List[int]] = None,
                        frequency_threshold: Optional[Union[int, float]] = 0.05,
                        other_queries_sample: int = 100,
                        replacement_type: str = 'zero_out',
                        return_activation_vectors: bool = False,
                        verbose: bool = True,
                        use_test_queries: bool = False) -> Dict[str, Any]:
        """
        Analyze document-specific neurons for a target document.

        This is the main analysis function that replicates the notebook's
        analyze_document_specific_neurons functionality.

        Args:
            target_doc_id: Document ID to analyze
            layer_indices: Layers to analyze, or None for default (17-23)
            frequency_threshold: Threshold for neuron filtering
            other_queries_sample: Number of other queries to test
            replacement_type: Type of neuron manipulation ('zero_out' or 'mean_value')
            return_activation_vectors: Whether to return activation analysis
            verbose: Whether to print detailed progress
            use_test_queries: Whether to use test queries for evaluation (neuron filtering still uses training data)

        Returns:
            Dictionary with complete analysis results
        """
        self._check_prerequisites()

        if replacement_type == 'mean_value' and self.detailed_activation_stats is None:
            raise ValueError(
                "detailed_activation_stats required for mean_value replacement. "
                "Load detailed stats or set replacement_type='zero_out'"
            )

        # Determine which result dictionary to use for evaluation
        evaluation_result_dict = self.test_result_dict if use_test_queries and self.test_result_dict else self.result_dict

        if use_test_queries and self.test_result_dict is None:
            raise ValueError("Test queries requested but test data not loaded. Use load_statistics with use_test_set=True.")

        return analyze_document_specific_neurons(
            model=self.model,
            tokenizer=self.tokenizer,
            result_dict=evaluation_result_dict,
            training_result_dict=self.result_dict,  # Always use training data for neuron filtering
            stats=self.neuron_stats,
            target_doc_id=target_doc_id,
            layer_indices=layer_indices,
            frequency_threshold=frequency_threshold,
            other_queries_sample=other_queries_sample,
            verbose=verbose,
            return_activation_vectors=return_activation_vectors,
            replacement_type=replacement_type,
            detailed_stats=self.detailed_activation_stats
        )

    def compare_documents(self,
                         doc_ids: Optional[List[Union[int, str]]] = None,
                         num_docs: int = 5,
                         layer_indices: Optional[List[int]] = None,
                         frequency_threshold: Optional[Union[int, float]] = 0.05,
                         other_queries_sample: int = 50,
                         verbose: bool = False) -> Dict[str, Any]:
        """
        Compare neuron specificity across multiple documents.

        Args:
            doc_ids: Specific document IDs to compare, or None for popular docs
            num_docs: Number of documents to analyze if doc_ids is None
            layer_indices: Layers to analyze
            frequency_threshold: Threshold for neuron filtering
            other_queries_sample: Number of other queries to test
            verbose: Whether to print detailed info for each document

        Returns:
            Dictionary with comparative analysis results
        """
        self._check_prerequisites()

        # Use popular documents if none specified
        if doc_ids is None:
            if self.popular_docs is None:
                raise ValueError("No popular documents available. Load statistics first.")
            doc_ids = [doc[0] for doc in self.popular_docs[:num_docs]]

        return compare_multiple_documents(
            model=self.model,
            tokenizer=self.tokenizer,
            result_dict=self.result_dict,
            stats=self.neuron_stats,
            doc_ids=doc_ids,
            layer_indices=layer_indices,
            frequency_threshold=frequency_threshold,
            other_queries_sample=other_queries_sample,
            verbose=verbose
        )

    def frequency_sweep_analysis(self,
                                target_doc_id: Union[int, str],
                                frequency_thresholds: Optional[List[float]] = None,
                                replacement_types: Optional[List[str]] = None,
                                layer_indices: Optional[List[int]] = None,
                                other_queries_sample: int = 100,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Perform frequency threshold sweep analysis.

        Args:
            target_doc_id: Document ID to analyze
            frequency_thresholds: List of thresholds to test
            replacement_types: List of replacement strategies to test
            layer_indices: Layers to analyze
            other_queries_sample: Number of other queries to test
            verbose: Whether to print progress

        Returns:
            Dictionary with sweep results
        """
        self._check_prerequisites()

        if frequency_thresholds is None:
            frequency_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

        if replacement_types is None:
            replacement_types = ['zero_out']
            if self.detailed_activation_stats is not None:
                replacement_types.append('mean_value')

        print(f" Running frequency sweep analysis for document {target_doc_id}")
        print(f" Testing {len(frequency_thresholds)} thresholds with {len(replacement_types)} replacement types")

        sweep_results = {
            'target_doc_id': target_doc_id,
            'frequency_thresholds': frequency_thresholds,
            'replacement_types': replacement_types,
            'results': {}
        }

        total_runs = len(frequency_thresholds) * len(replacement_types)
        current_run = 0

        for threshold in frequency_thresholds:
            sweep_results['results'][threshold] = {}

            for replacement_type in replacement_types:
                current_run += 1
                if verbose:
                    print(f"\n[{current_run}/{total_runs}] Testing threshold {threshold:.3f} with {replacement_type}")

                try:
                    result = self.analyze_document(
                        target_doc_id=target_doc_id,
                        layer_indices=layer_indices,
                        frequency_threshold=threshold,
                        other_queries_sample=other_queries_sample,
                        replacement_type=replacement_type,
                        verbose=False
                    )

                    sweep_results['results'][threshold][replacement_type] = result

                except Exception as e:
                    print(f" Error in sweep analysis: {e}")
                    sweep_results['results'][threshold][replacement_type] = {'error': str(e)}

        print(f" Frequency sweep analysis completed")
        return sweep_results

    def evaluate_model(self,
                      queries: Optional[List[str]] = None,
                      batch_size: int = 256,
                      with_hooks: bool = False,
                      hook_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate model performance with or without hooks.

        Args:
            queries: List of queries to evaluate, or None for all loaded queries
            batch_size: Batch size for evaluation
            with_hooks: Whether to use hooks during evaluation
            hook_config: Configuration for hooks if with_hooks=True

        Returns:
            Dictionary with evaluation results
        """
        self._check_prerequisites()

        if queries is None:
            queries = list(self.result_dict.keys())

        return run_inference_with_hooks(
            model=self.model,
            tokenizer=self.tokenizer,
            queries=queries,
            hook_manager=self.hook_manager if with_hooks else None,
            batch_size=batch_size,
            return_activations=hook_config.get('return_activations', False) if hook_config else False
        )

    def visualize_results(self,
                         results: Dict[str, Any],
                         plot_type: str = 'frequency_sweep',
                         save_path: Optional[Union[str, Path]] = None,
                         show_plot: bool = True) -> Any:
        """
        Create visualizations for analysis results.

        Args:
            results: Analysis results to visualize
            plot_type: Type of plot ('frequency_sweep', 'multi_document', 'activation_dist')
            save_path: Path to save plot
            show_plot: Whether to display plot

        Returns:
            Matplotlib figure object
        """
        if plot_type == 'frequency_sweep':
            return plot_frequency_sweep_results(results, save_path=save_path, show_plot=show_plot)
        elif plot_type == 'multi_document':
            return plot_multi_document_comparison(results, save_path=save_path, show_plot=show_plot)
        elif plot_type == 'activation_dist':
            return plot_activation_distributions(results, save_path=save_path, show_plot=show_plot)
        elif plot_type == 'comprehensive_sweep':
            return analyze_and_visualize_sweep_results(results, output_dir=save_path)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def create_summary_table(self, results: Dict[str, Any]) -> Any:
        """Create a summary table for analysis results."""
        return create_analysis_summary_table(results)

    def get_popular_documents(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most popular documents by query frequency.

        Args:
            n: Number of documents to return

        Returns:
            List of (document_id, query_count) tuples
        """
        if self.popular_docs is None:
            if self.result_dict is None:
                raise ValueError("Data must be loaded first. Call load_data() or load_statistics().")

            from .data_loader import create_popular_docs_list
            self.popular_docs = create_popular_docs_list(self.result_dict)

        return self.popular_docs[:n]

    def _check_prerequisites(self):
        """Check that all required components are loaded."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded. Call load_model() first.")

        if self.result_dict is None or self.neuron_stats is None:
            raise ValueError("Statistics must be loaded. Call load_statistics() or generate_statistics() first.")

    def __repr__(self):
        """String representation of the analyzer."""
        status = []
        status.append(f"Model: {'loaded' if self.model else 'not loaded'}")
        status.append(f"Data: {'loaded' if self.result_dict else 'not loaded'}")
        status.append(f"Stats: {'loaded' if self.neuron_stats else 'not loaded'}")
        status.append(f"Detailed: {'loaded' if self.detailed_activation_stats else 'not loaded'}")

        return f"DSIDocumentAnalyzer({', '.join(status)})"