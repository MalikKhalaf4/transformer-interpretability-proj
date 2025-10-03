"""
Utility functions and data structures for DSI document analysis.

This module provides helper functions for data manipulation, validation,
and common operations used throughout the DSI analysis pipeline.
"""

import os
import json
import psutil
import torch
import numpy as np
import regex as re
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        dict: Memory usage info with keys 'used_gb', 'available_gb', 'percent'
    """
    memory = psutil.virtual_memory()
    return {
        'used_gb': (memory.total - memory.available) / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent': memory.percent
    }


def check_memory_availability(required_gb: float = 10.0) -> bool:
    """
    Check if enough memory is available for analysis.

    Args:
        required_gb: Minimum required memory in GB

    Returns:
        bool: True if enough memory is available

    Raises:
        MemoryError: If insufficient memory is available
    """
    memory_info = get_memory_usage()
    if memory_info['available_gb'] < required_gb:
        raise MemoryError(
            f"Insufficient memory. Required: {required_gb:.1f}GB, "
            f"Available: {memory_info['available_gb']:.1f}GB"
        )
    return True


def safe_file_write(file_path: Union[str, Path], data: Any, overwrite: bool = False) -> None:
    """
    Safely write data to file with existence checking.

    Args:
        file_path: Path to output file
        data: Data to write (will be JSON serialized)
        overwrite: If True, allow overwriting existing files

    Raises:
        FileExistsError: If file exists and overwrite=False
        IOError: If file cannot be written
    """
    file_path = Path(file_path)

    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {file_path} already exists. Use overwrite=True to overwrite."
        )

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to write file {file_path}: {e}")


def safe_file_read(file_path: Union[str, Path]) -> Any:
    """
    Safely read JSON data from file.

    Args:
        file_path: Path to input file

    Returns:
        Loaded data from JSON file

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def validate_layer_indices(layer_indices: Optional[List[int]],
                         max_layers: int = 24) -> List[int]:
    """
    Validate and normalize layer indices.

    Args:
        layer_indices: List of layer indices, or None for default (17-23)
        max_layers: Maximum number of layers in the model

    Returns:
        List of validated layer indices

    Raises:
        ValueError: If layer indices are invalid
    """
    if layer_indices is None:
        return list(range(17, 24))  # Default layers 17-23

    if not isinstance(layer_indices, (list, tuple)):
        raise ValueError("layer_indices must be a list or tuple")

    for idx in layer_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= max_layers:
            raise ValueError(f"Invalid layer index {idx}. Must be 0 <= idx < {max_layers}")

    return list(layer_indices)


def pad_relevant_docs(relevant_docs: List[List[Any]]) -> List[List[Any]]:
    """
    Pad relevant documents to same length for batch processing.

    Args:
        relevant_docs: List of lists with varying lengths

    Returns:
        List of lists padded to same length with -1
    """
    relevant_docs = [[int(item) for item in sublist] for sublist in relevant_docs]
    max_len = max(len(sublist) for sublist in relevant_docs) if relevant_docs else 0

    padded_relevant_docs = []
    for sublist in relevant_docs:
        num_padding = max_len - len(sublist)
        padded_sublist = sublist + [-1] * num_padding
        padded_relevant_docs.append(padded_sublist)

    return padded_relevant_docs


def extract_doc_ids_from_output(decoder_outputs: List[str]) -> List[Optional[str]]:
    """
    Extract document IDs from model decoder outputs.

    Args:
        decoder_outputs: List of decoder output strings

    Returns:
        List of extracted document IDs as strings (None if not found)
    """
    doc_out_ids = []
    for output in decoder_outputs:
        doc_id = re.findall(r"@DOC_ID_([0-9]+)@", output)
        assert len(doc_id) <= 1, f"Multiple doc IDs found in output: {output}"
        # Keep as string to match dataset format, or None if not found
        doc_out_ids.append(doc_id[0] if len(doc_id) else None)

    return doc_out_ids


def validate_frequency_threshold(threshold: Union[int, float],
                               total_queries: int) -> Tuple[float, int]:
    """
    Validate and normalize frequency threshold.

    Args:
        threshold: Frequency threshold (absolute count or percentage)
        total_queries: Total number of queries for percentage calculation

    Returns:
        Tuple of (percentage_threshold, absolute_threshold)

    Raises:
        ValueError: If threshold is invalid
    """
    if threshold is None:
        return 1.0, total_queries

    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("Frequency threshold must be a positive number")

    if threshold <= 1:
        # Percentage threshold
        percentage_threshold = float(threshold)
        absolute_threshold = int(threshold * total_queries)
    else:
        # Absolute threshold
        absolute_threshold = int(threshold)
        percentage_threshold = threshold / total_queries

    return percentage_threshold, absolute_threshold


def create_default_paths(base_dir: Union[str, Path] = None) -> Dict[str, Path]:
    """
    Create default directory paths for DSI analysis.

    Args:
        base_dir: Base directory for all paths, or None for current directory

    Returns:
        Dictionary of default paths
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    paths = {
        'data_dir': base_dir / 'data',
        'models_dir': base_dir / 'data' / 'models',
        'datasets_dir': base_dir / 'data' / 'datasets',
        'statistics_dir': base_dir / 'data' / 'statistics',
        'results_dir': base_dir / 'data' / 'results',
        'config_dir': base_dir / 'config',
        'model_path': base_dir.parent / 'DSI-large-TriviaQA',  # Model in interpretability directory
        'trivia_qa_path': base_dir.parent / 'TriviaQAData',    # TriviaQA in interpretability directory
        'default_stats_path': base_dir / 'data' / 'activated_neurons.json',  # Stats under data/
        'default_neuron_stats_path': base_dir / 'data' / 'neuron_activation_stats.json'  # Stats under data/
    }

    return paths


def format_accuracy_results(correct_before: int, correct_after: int, total: int) -> Dict[str, float]:
    """
    Format accuracy results with proper percentage calculations.

    Args:
        correct_before: Number of correct predictions before manipulation
        correct_after: Number of correct predictions after manipulation
        total: Total number of queries

    Returns:
        Dictionary with accuracy metrics
    """
    if total == 0:
        return {
            'accuracy_before': 0.0,
            'accuracy_after': 0.0,
            'accuracy_drop': 0.0,
            'accuracy_drop_pct': 0.0
        }

    accuracy_before = correct_before / total
    accuracy_after = correct_after / total
    accuracy_drop = correct_before - correct_after
    accuracy_drop_pct = accuracy_drop / total

    return {
        'accuracy_before': accuracy_before,
        'accuracy_after': accuracy_after,
        'accuracy_drop': accuracy_drop,
        'accuracy_drop_pct': accuracy_drop_pct
    }


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of specified size.

    Args:
        items: List to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_model_device(model) -> str:
    """
    Get the device of a model, handling both regular models and HookedEncoderDecoder.

    Args:
        model: PyTorch model

    Returns:
        Device string (e.g., 'cuda:0', 'cpu')
    """
    if hasattr(model, 'device'):
        return model.device
    else:
        # For HookedEncoderDecoder and other models without device attribute
        return next(model.parameters()).device


def get_device_info() -> Dict[str, Any]:
    """
    Get device information for model loading.

    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_allocated': 0,
        'memory_reserved': 0
    }

    if torch.cuda.is_available():
        device_info['current_device'] = torch.cuda.current_device()
        device_info['device_name'] = torch.cuda.get_device_name()
        device_info['memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
        device_info['memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)  # GB

    return device_info


def print_analysis_summary(results: Dict[str, Any], title: str = "Analysis Summary") -> None:
    """
    Print a formatted summary of analysis results.

    Args:
        results: Analysis results dictionary
        title: Title for the summary
    """
    print(f"\n{'=' * len(title)}")
    print(title)
    print(f"{'=' * len(title)}")

    if 'summary' in results:
        summary = results['summary']
        print(f"Target Document Accuracy:")
        print(f"  Before: {summary.get('target_accuracy_before', 0):.1f}%")
        print(f"  After:  {summary.get('target_accuracy_after', 0):.1f}%")
        print(f"  Drop:   {summary.get('target_accuracy_drop', 0):.1f}%")

        print(f"\nOther Documents Accuracy:")
        print(f"  Before: {summary.get('other_accuracy_before', 0):.1f}%")
        print(f"  After:  {summary.get('other_accuracy_after', 0):.1f}%")
        print(f"  Drop:   {summary.get('other_accuracy_drop', 0):.1f}%")

        print(f"\nQueries Changed:")
        print(f"  Target: {results.get('queries_changed', {}).get('n_target_changed', 0)}")
        print(f"  Other:  {results.get('queries_changed', {}).get('n_other_changed', 0)}")