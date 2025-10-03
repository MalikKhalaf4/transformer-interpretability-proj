"""
Statistics management for DSI document analysis.

This module handles generation, loading, and management of neuron activation
statistics and document-specific neuron frequency data.
"""

import json
import time
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm

from .utils import safe_file_write, safe_file_read, create_default_paths, check_memory_availability, get_model_device
from .hook_manager import ActivationExtractor, extract_activated_neurons


class StatisticsManager:
    """
    Manager for neuron activation statistics and frequency data.

    Handles generation, storage, and loading of statistics used for
    document-specific neuron analysis.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the statistics manager.

        Args:
            base_dir: Base directory for statistics files
        """
        if base_dir is None:
            paths = create_default_paths()
            base_dir = paths['data_dir']

        self.base_dir = Path(base_dir)
        self.result_dict = None
        self.neuron_stats = None
        self.detailed_activation_stats = None

    def load_result_dict(self, file_path: Optional[Union[str, Path]] = None,
                        use_test_set: bool = False) -> Dict[str, Any]:
        """
        Load the result dictionary containing activated neurons for each query.

        Args:
            file_path: Path to result dict file, or None for default
            use_test_set: If True, load test set data instead of training data

        Returns:
            Loaded result dictionary

        Raises:
            FileNotFoundError: If file is not found
        """
        if file_path is None:
            if use_test_set:
                file_path = self.base_dir / "activated_neurons_test.json"
            else:
                file_path = self.base_dir / "activated_neurons.json"

        self.result_dict = safe_file_read(file_path)
        return self.result_dict

    def load_test_result_dict(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load the test set result dictionary containing activated neurons for each query.

        Args:
            file_path: Path to test result dict file, or None for default

        Returns:
            Loaded test result dictionary

        Raises:
            FileNotFoundError: If file is not found
        """
        return self.load_result_dict(file_path, use_test_set=True)

    def load_neuron_stats(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load neuron activation frequency statistics.

        Args:
            file_path: Path to neuron stats file, or None for default

        Returns:
            Loaded neuron statistics

        Raises:
            FileNotFoundError: If file is not found
        """
        if file_path is None:
            file_path = self.base_dir / "neuron_activation_stats.json"

        self.neuron_stats = safe_file_read(file_path)
        return self.neuron_stats

    def load_detailed_activation_stats(self, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load detailed activation statistics (means, variances, etc.).

        Args:
            file_path: Path to detailed stats file, or None for default

        Returns:
            Loaded detailed activation statistics

        Raises:
            FileNotFoundError: If file is not found
        """
        if file_path is None:
            # Try common filenames for detailed stats
            possible_paths = [
                self.base_dir / "detailed_activation_stats.json",
                self.base_dir / "neuron_activation_stats_with_entries.json",
                self.base_dir / "activation_means_vars.json"
            ]

            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break

            if file_path is None:
                raise FileNotFoundError(
                    f"No detailed activation stats file found. Tried: {possible_paths}"
                )

        self.detailed_activation_stats = safe_file_read(file_path)
        return self.detailed_activation_stats

    def generate_activation_statistics(self,
                                     model,
                                     tokenizer,
                                     data_loader,
                                     layer_indices: Optional[List[int]] = None,
                                     save_path: Optional[Union[str, Path]] = None,
                                     verbose: bool = True) -> Dict[str, Any]:
        """
        Generate activation statistics from data loader.

        Args:
            model: The transformer model
            tokenizer: Model tokenizer
            data_loader: DataLoader with queries
            layer_indices: Layers to analyze
            save_path: Path to save statistics
            verbose: Whether to print progress

        Returns:
            Generated statistics dictionary
        """
        if verbose:
            print("Generating activation statistics...")

        # Check memory availability
        check_memory_availability(10.0)  # Require at least 10GB

        if layer_indices is None:
            layer_indices = list(range(17, 24))  # Default layers 17-23

        # Initialize statistics storage
        result_dict = defaultdict(lambda: {})
        activation_counts = defaultdict(lambda: defaultdict(int))
        total_activations = defaultdict(lambda: defaultdict(int))

        # Create activation extractor
        extractor = ActivationExtractor(layer_indices)

        count = 0
        progress_bar = tqdm(data_loader, desc="Processing queries") if verbose else data_loader

        try:
            for input_texts, target_texts, ids in progress_bar:
                # Tokenize inputs
                input_tokens = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
                input_ids = input_tokens['input_ids'].to(get_model_device(model))

                # Create decoder input
                decoder_input = torch.zeros((len(input_texts), 1), dtype=torch.long, device=get_model_device(model))

                # Clear previous activations
                extractor.clear_activations()

                # Create hooks for this batch
                hooks = extractor.create_extraction_hooks(model)

                try:
                    # Run inference
                    with torch.no_grad():
                        logits, cache = model.run_with_cache(input_ids, decoder_input, remove_batch_dim=False)

                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)

                    # Process each query in the batch
                    for i, (query, target_docs, query_id) in enumerate(zip(input_texts, target_texts, ids)):
                        # Extract activated neurons for this query
                        activated_neurons = {}
                        for layer_key, activation in extractor.activations.items():
                            # Get activation for this specific query (index i)
                            query_activation = activation[i:i+1]  # Keep batch dimension
                            activated_indices = (query_activation > 0).nonzero(as_tuple=False)
                            activated_neurons[layer_key] = activated_indices.cpu().numpy().tolist()

                        # Get correct document ID
                        correct_doc_id = target_docs[0] if target_docs else None

                        # Decode prediction
                        prediction_text = tokenizer.decode(predictions[i][0])

                        # Store in result dict
                        result_dict[query] = {
                            'query_id': query_id,
                            'correct_doc_id': correct_doc_id,
                            'prediction': prediction_text,
                            'activated_neurons': activated_neurons
                        }

                        # Update activation counts
                        for layer_key, indices in activated_neurons.items():
                            for idx_list in indices:
                                if len(idx_list) >= 2:  # [batch_idx, neuron_idx]
                                    neuron_idx = idx_list[1]
                                    activation_counts[layer_key][neuron_idx] += 1

                        count += 1

                finally:
                    # Always remove hooks
                    extractor.remove_hooks()

                if verbose and count % 100 == 0:
                    progress_bar.set_postfix({'processed': count})

        except KeyboardInterrupt:
            if verbose:
                print(f"\nInterrupted after processing {count} queries")
        except Exception as e:
            print(f"Error during statistics generation: {e}")
            raise

        # Convert to regular dicts for JSON serialization
        result_dict = dict(result_dict)
        stats_dict = {
            layer_key: dict(neuron_counts)
            for layer_key, neuron_counts in activation_counts.items()
        }

        if verbose:
            print(f"\nProcessed {count} queries total")
            print(f"Generated statistics for {len(stats_dict)} layers")

        # Save results
        if save_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save result dict
            result_path = Path(save_path).parent / f"activated_neurons_{timestamp}.json"
            safe_file_write(result_path, result_dict, overwrite=False)

            # Save stats
            stats_path = Path(save_path).parent / f"neuron_activation_stats_{timestamp}.json"
            safe_file_write(stats_path, stats_dict, overwrite=False)

            if verbose:
                print(f"Saved results to {result_path}")
                print(f"Saved statistics to {stats_path}")

        # Store in instance
        self.result_dict = result_dict
        self.neuron_stats = stats_dict

        return {
            'result_dict': result_dict,
            'stats': stats_dict,
            'total_queries': count
        }

    def generate_detailed_activation_stats(self,
                                         model,
                                         tokenizer,
                                         data_loader,
                                         layer_indices: Optional[List[int]] = None,
                                         save_path: Optional[Union[str, Path]] = None,
                                         sample_size: Optional[int] = None,
                                         verbose: bool = True) -> Dict[str, Any]:
        """
        Generate detailed activation statistics including means and variances.

        Args:
            model: The transformer model
            tokenizer: Model tokenizer
            data_loader: DataLoader with queries
            layer_indices: Layers to analyze
            save_path: Path to save statistics
            sample_size: Limit to first N samples (for memory efficiency)
            verbose: Whether to print progress

        Returns:
            Generated detailed statistics dictionary
        """
        if verbose:
            print("Generating detailed activation statistics (means, variances)...")

        # Check memory availability
        check_memory_availability(15.0)  # Require more memory for detailed stats

        if layer_indices is None:
            layer_indices = list(range(17, 24))

        # Storage for all activations
        all_activations = {f'layer_{i}': [] for i in layer_indices}

        # Create activation extractor
        extractor = ActivationExtractor(layer_indices)

        count = 0
        progress_bar = tqdm(data_loader, desc="Collecting activations") if verbose else data_loader

        try:
            for input_texts, target_texts, ids in progress_bar:
                if sample_size and count >= sample_size:
                    break

                # Tokenize inputs
                input_tokens = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
                input_ids = input_tokens['input_ids'].to(get_model_device(model))

                # Create decoder input
                decoder_input = torch.zeros((len(input_texts), 1), dtype=torch.long, device=get_model_device(model))

                # Clear previous activations
                extractor.clear_activations()

                # Create hooks
                hooks = extractor.create_extraction_hooks(model)

                try:
                    # Run inference
                    with torch.no_grad():
                        logits, cache = model.run_with_cache(input_ids, decoder_input, remove_batch_dim=False)

                    # Store activations
                    for layer_key, activation in extractor.activations.items():
                        all_activations[layer_key].append(activation.cpu())

                finally:
                    extractor.remove_hooks()

                count += len(input_texts)

                if verbose and count % 500 == 0:
                    progress_bar.set_postfix({'samples': count})

        except KeyboardInterrupt:
            if verbose:
                print(f"\nInterrupted after processing {count} samples")

        # Calculate statistics
        detailed_stats = {}

        for layer_key, activations_list in all_activations.items():
            if not activations_list:
                continue

            if verbose:
                print(f"Calculating statistics for {layer_key}...")

            # Concatenate all activations for this layer
            all_layer_activations = torch.cat(activations_list, dim=0)

            # Calculate means and variances per neuron
            means = torch.mean(all_layer_activations, dim=0)
            variances = torch.var(all_layer_activations, dim=0)
            stds = torch.std(all_layer_activations, dim=0)

            # Store as nested dict for easy JSON serialization
            neuron_stats = {}
            for neuron_idx in range(means.shape[0]):
                neuron_stats[str(neuron_idx)] = {
                    'mean': float(means[neuron_idx]),
                    'variance': float(variances[neuron_idx]),
                    'std': float(stds[neuron_idx])
                }

            detailed_stats[layer_key] = {
                'mean': {str(i): float(means[i]) for i in range(len(means))},
                'variance': {str(i): float(variances[i]) for i in range(len(variances))},
                'std': {str(i): float(stds[i]) for i in range(len(stds))},
                'shape': list(all_layer_activations.shape),
                'num_samples': all_layer_activations.shape[0]
            }

        # Save results
        if save_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            stats_path = Path(save_path).parent / f"detailed_activation_stats_{timestamp}.json"
            safe_file_write(stats_path, detailed_stats, overwrite=False)

            if verbose:
                print(f"Saved detailed statistics to {stats_path}")

        # Store in instance
        self.detailed_activation_stats = detailed_stats

        if verbose:
            print(f"Generated detailed statistics for {len(detailed_stats)} layers")
            print(f"Processed {count} total samples")

        return detailed_stats

    def get_neuron_frequency_stats(self,
                                 layer_key: str,
                                 neuron_idx: int,
                                 total_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        Get frequency statistics for a specific neuron.

        Args:
            layer_key: Layer identifier (e.g., 'layer_17')
            neuron_idx: Neuron index
            total_queries: Total number of queries for percentage calculation

        Returns:
            Dictionary with frequency statistics
        """
        if self.neuron_stats is None:
            raise ValueError("Neuron statistics not loaded. Call load_neuron_stats() first.")

        layer_stats = self.neuron_stats.get(layer_key, {})
        activation_count = layer_stats.get(str(neuron_idx), 0)

        stats = {
            'activation_count': activation_count,
            'layer': layer_key,
            'neuron_idx': neuron_idx
        }

        if total_queries:
            stats['activation_percentage'] = activation_count / total_queries
            stats['total_queries'] = total_queries

        return stats

    def filter_neurons_by_frequency(self,
                                   layer_key: str,
                                   frequency_threshold: Union[int, float],
                                   total_queries: int) -> List[int]:
        """
        Filter neurons by activation frequency threshold.

        Args:
            layer_key: Layer identifier
            frequency_threshold: Threshold (absolute count or percentage)
            total_queries: Total number of queries

        Returns:
            List of neuron indices below the threshold
        """
        if self.neuron_stats is None:
            raise ValueError("Neuron statistics not loaded.")

        layer_stats = self.neuron_stats.get(layer_key, {})

        if frequency_threshold <= 1:
            # Percentage threshold
            threshold_count = int(frequency_threshold * total_queries)
        else:
            # Absolute threshold
            threshold_count = int(frequency_threshold)

        filtered_neurons = []
        for neuron_idx_str, count in layer_stats.items():
            if count < threshold_count:
                filtered_neurons.append(int(neuron_idx_str))

        return sorted(filtered_neurons)

    def save_statistics(self,
                       result_dict: Optional[Dict[str, Any]] = None,
                       neuron_stats: Optional[Dict[str, Any]] = None,
                       detailed_stats: Optional[Dict[str, Any]] = None,
                       output_dir: Optional[Union[str, Path]] = None,
                       overwrite: bool = False) -> Dict[str, Path]:
        """
        Save statistics to files.

        Args:
            result_dict: Result dictionary to save
            neuron_stats: Neuron statistics to save
            detailed_stats: Detailed statistics to save
            output_dir: Output directory
            overwrite: Whether to overwrite existing files

        Returns:
            Dictionary mapping data types to saved file paths
        """
        if output_dir is None:
            output_dir = self.base_dir

        output_dir = Path(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        if result_dict is not None:
            path = output_dir / f"activated_neurons_{timestamp}.json"
            safe_file_write(path, result_dict, overwrite=overwrite)
            saved_files['result_dict'] = path

        if neuron_stats is not None:
            path = output_dir / f"neuron_activation_stats_{timestamp}.json"
            safe_file_write(path, neuron_stats, overwrite=overwrite)
            saved_files['neuron_stats'] = path

        if detailed_stats is not None:
            path = output_dir / f"detailed_activation_stats_{timestamp}.json"
            safe_file_write(path, detailed_stats, overwrite=overwrite)
            saved_files['detailed_stats'] = path

        return saved_files