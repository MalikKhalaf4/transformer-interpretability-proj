"""
Core analysis functions for document-specific neuron analysis.

This module contains the main analysis algorithms extracted from the notebook,
including document-specific neuron collection, manipulation testing, and
comprehensive analysis pipelines.
"""

import torch
import numpy as np
import regex as re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm

from .utils import (
    validate_frequency_threshold, format_accuracy_results,
    batch_list, extract_doc_ids_from_output, print_analysis_summary, get_model_device,
    safe_file_write
)
from .hook_manager import NeuronManipulator, MultiNeuronHookManager


def collect_document_specific_neurons(result_dict: Dict[str, Any],
                                    stats: Dict[str, Any],
                                    target_doc_id: Union[int, str],
                                    layer_indices: Optional[List[int]] = None,
                                    frequency_threshold: Optional[Union[int, float]] = None) -> Dict[str, Any]:
    """
    Collect neurons that are activated for a specific document ID.

    Args:
        result_dict: Dictionary mapping queries to activated neurons and document IDs
        stats: Neuron activation frequency statistics
        target_doc_id: Document ID to analyze
        layer_indices: List of layer indices to analyze, or None for default (17-23)
        frequency_threshold: Threshold for filtering neurons by activation frequency

    Returns:
        Dictionary with collected neuron data
    """
    if layer_indices is None:
        layer_indices = list(range(17, 24))  # Default layers 17-23

    target_doc_id = str(target_doc_id)  # Ensure string format

    # Collect queries where this document is the correct answer
    target_queries = []
    for query, entry in result_dict.items():
        if str(entry.get("correct_doc_id")) == target_doc_id:
            target_queries.append(query)

    # Collect all neurons activated for these queries
    neurons_by_layer = {}
    neuron_frequencies = {}

    for layer_id in layer_indices:
        layer_key = f'layer_{layer_id}'
        activated_neurons = set()

        # Collect neurons from all target queries
        for query in target_queries:
            entry = result_dict.get(query, {})
            layer_neurons = entry.get('activated_neurons', {}).get(layer_key, [])

            # Handle different formats of neuron indices
            for neuron_data in layer_neurons:
                if isinstance(neuron_data, list) and len(neuron_data) >= 2:
                    neuron_idx = neuron_data[1]  # [batch_idx, neuron_idx]
                    activated_neurons.add(neuron_idx)
                elif isinstance(neuron_data, int):
                    activated_neurons.add(neuron_data)

        neurons_by_layer[layer_key] = list(activated_neurons)

        # Get frequency information for these neurons
        layer_stats = stats.get("by_layer", {}).get(layer_key, {})
        neuron_freq = {}
        for neuron_idx in activated_neurons:
            neuron_data = layer_stats.get(str(neuron_idx), {})
            # Use the count (absolute frequency) for filtering
            freq = neuron_data.get("count", 0) if isinstance(neuron_data, dict) else 0
            neuron_freq[neuron_idx] = freq

        neuron_frequencies[layer_key] = neuron_freq

    # Apply frequency filtering if specified
    filtered_neurons = {}
    if frequency_threshold is not None:
        total_queries = len(result_dict)
        _, absolute_threshold = validate_frequency_threshold(frequency_threshold, total_queries)

        for layer_key, neurons in neurons_by_layer.items():
            layer_freq = neuron_frequencies[layer_key]
            filtered = [n for n in neurons if layer_freq.get(n, 0) < absolute_threshold]
            filtered_neurons[layer_key] = filtered
    else:
        filtered_neurons = neurons_by_layer.copy()

    return {
        'target_doc_id': target_doc_id,
        'target_queries': target_queries,
        'neurons_by_layer': neurons_by_layer,
        'filtered_neurons': filtered_neurons,
        'neuron_frequencies': neuron_frequencies,
        'layer_indices': layer_indices,
        'frequency_threshold': frequency_threshold
    }


def test_document_neuron_effects(model,
                               tokenizer,
                               result_dict: Dict[str, Any],
                               target_doc_id: Union[int, str],
                               neurons_by_layer: Dict[str, List[int]],
                               other_queries_sample: Optional[int] = 100,
                               batch_size: int = 256,
                               replacement_type: str = 'zero_out',
                               detailed_stats: Optional[Dict[str, Any]] = None,
                               return_activation_vectors: bool = False) -> Dict[str, Any]:
    """
    Test the effect of manipulating document-specific neurons.

    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        result_dict: Dictionary with query data
        target_doc_id: Document ID being analyzed
        neurons_by_layer: Neurons to manipulate by layer
        other_queries_sample: Number of other queries to test for side effects (None = all)
        batch_size: Batch size for inference
        replacement_type: Type of neuron manipulation ('zero_out' or 'mean_value')
        detailed_stats: Statistics for mean_value replacement
        return_activation_vectors: Whether to return activation vectors for analysis

    Returns:
        Dictionary with test results
    """
    # Keep target_doc_id as string for consistent comparison with dataset
    target_doc_id = str(target_doc_id)

    # Separate target queries from other queries
    target_queries = []
    other_queries = []
    target_query_keys = []  # Keep track of original query keys for mapping
    other_query_keys = []

    for query, entry in result_dict.items():
        if str(entry.get("correct_doc_id")) == target_doc_id:
            target_queries.append(entry['input'])  # Use input field instead of query key
            target_query_keys.append(query)  # Keep original key for mapping
        else:
            other_queries.append(entry['input'])  # Use input field instead of query key
            other_query_keys.append(query)  # Keep original key for mapping

    # Sample other queries if needed
    if other_queries_sample is not None and len(other_queries) > other_queries_sample:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(other_queries), other_queries_sample, replace=False)
        other_queries = [other_queries[i] for i in indices]
        other_query_keys = [other_query_keys[i] for i in indices]

    # Test queries without manipulation (baseline)
    all_test_queries = target_queries + other_queries
    baseline_results = _run_batched_inference(
        model, tokenizer, all_test_queries, batch_size, return_activations=return_activation_vectors
    )

    # Test queries with neuron manipulation using context manager approach
    manipulator = NeuronManipulator(replacement_type, detailed_stats)
    hooks = manipulator.create_manipulation_hooks(neurons_by_layer)

    # Use context manager for hook management like working notebook
    with model.hooks(fwd_hooks=hooks):
        manipulation_results = _run_batched_inference(
            model, tokenizer, all_test_queries, batch_size, return_activations=return_activation_vectors
        )

    # Analyze results
    results = _analyze_manipulation_results(
        target_queries, other_queries, target_query_keys, other_query_keys,
        result_dict, target_doc_id, baseline_results, manipulation_results
    )

    # Add activation vectors if requested
    if return_activation_vectors:
        results['activation_vectors'] = {
            'baseline': baseline_results.get('activations'),
            'manipulated': manipulation_results.get('activations')
        }

    return results


def _run_batched_inference(model,
                         tokenizer,
                         queries: List[str],
                         batch_size: int,
                         return_activations: bool = False) -> Dict[str, Any]:
    """
    Run batched inference on a list of queries.

    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        queries: List of input queries
        batch_size: Batch size for processing
        return_activations: Whether to capture activations

    Returns:
        Dictionary with predictions and optionally activations
    """
    all_predictions = []
    all_activations = [] if return_activations else None

    query_batches = batch_list(queries, batch_size)

    for batch_queries in query_batches:
        # Run inference with direct string input (like working notebook)
        with torch.no_grad():
            if return_activations:
                logits, cache = model.run_with_cache(batch_queries, remove_batch_dim=False)
                # Store relevant activations if needed
                batch_activations = {k: v.cpu() for k, v in cache.items() if 'mlp.hook_post' in k}
                all_activations.append(batch_activations)
            else:
                logits = model(batch_queries)

        # Get predictions (handle full sequences like notebook)
        if logits.dim() == 3:  # [batch_size, seq_len, vocab_size]
            predicted_ids = torch.argmax(logits, dim=-1)
            # Decode each sequence in the batch
            for i in range(predicted_ids.shape[0]):
                output_text = tokenizer.decode(predicted_ids[i].squeeze(), skip_special_tokens=False)
                all_predictions.append(output_text)
        else:
            # Fallback for unexpected shapes
            predictions = torch.argmax(logits, dim=-1)
            decoded_predictions = [tokenizer.decode(pred.squeeze(), skip_special_tokens=False) for pred in predictions]
            all_predictions.extend(decoded_predictions)


    results = {
        'predictions': all_predictions,
        'queries': queries
    }

    if return_activations:
        results['activations'] = all_activations

    return results


def _analyze_manipulation_results(target_queries: List[str],
                                other_queries: List[str],
                                target_query_keys: List[str],
                                other_query_keys: List[str],
                                result_dict: Dict[str, Any],
                                target_doc_id: str,
                                baseline_results: Dict[str, Any],
                                manipulation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the results of neuron manipulation experiments.

    Args:
        target_queries: Input texts for the target document
        other_queries: Input texts for other documents
        target_query_keys: Original query keys for target document
        other_query_keys: Original query keys for other documents
        result_dict: Original result dictionary
        target_doc_id: Target document ID
        baseline_results: Results without manipulation
        manipulation_results: Results with manipulation

    Returns:
        Dictionary with detailed analysis results
    """
    baseline_preds = baseline_results['predictions']
    manipulation_preds = manipulation_results['predictions']

    # Split predictions by query type
    n_target = len(target_queries)
    target_baseline = baseline_preds[:n_target]
    target_manipulation = manipulation_preds[:n_target]
    other_baseline = baseline_preds[n_target:]
    other_manipulation = manipulation_preds[n_target:]

    # Analyze target document accuracy
    target_correct_before = 0
    target_correct_after = 0
    target_changed_queries = []

    # Track relevant docs accuracy (matches any relevant document)
    target_relevant_correct_before = 0
    target_relevant_correct_after = 0

    for i, (query, query_key) in enumerate(zip(target_queries, target_query_keys)):
        # Extract document IDs from predictions
        baseline_doc_ids = extract_doc_ids_from_output([target_baseline[i]])
        manipulation_doc_ids = extract_doc_ids_from_output([target_manipulation[i]])

        # Convert target_doc_id to string for consistent comparison with extracted IDs
        target_doc_id_str = str(target_doc_id)
        baseline_correct = baseline_doc_ids[0] == target_doc_id_str
        manipulation_correct = manipulation_doc_ids[0] == target_doc_id_str

        if baseline_correct:
            target_correct_before += 1
        if manipulation_correct:
            target_correct_after += 1

        # Check if predictions match any relevant document
        relevant_docs = result_dict[query_key].get('relevant_docs', [])
        # Convert to strings for consistent comparison
        relevant_docs_str = [str(doc) for doc in relevant_docs]

        baseline_relevant_correct = baseline_doc_ids[0] is not None and baseline_doc_ids[0] in relevant_docs_str
        manipulation_relevant_correct = manipulation_doc_ids[0] is not None and manipulation_doc_ids[0] in relevant_docs_str

        if baseline_relevant_correct:
            target_relevant_correct_before += 1
        if manipulation_relevant_correct:
            target_relevant_correct_after += 1

        # Track queries that changed from correct to incorrect
        if baseline_correct and not manipulation_correct:
            # Extract document IDs from predictions for cleaner output (with error handling)
            try:
                baseline_doc_id = extract_doc_ids_from_output([target_baseline[i]])[0]
            except (IndexError, TypeError):
                baseline_doc_id = None

            try:
                manipulation_doc_id = extract_doc_ids_from_output([target_manipulation[i]])[0]
            except (IndexError, TypeError):
                manipulation_doc_id = None

            # Get relevant docs from result_dict
            relevant_docs = result_dict[query_key].get('relevant_docs', [])

            target_changed_queries.append({
                'query_key': query_key,  # Use query key for tracking
                'query_input': query,    # Keep input for reference
                'baseline_pred': target_baseline[i],
                'manipulation_pred': target_manipulation[i],
                'expected_doc_id': target_doc_id_str,
                'baseline_doc_id': baseline_doc_id,
                'manipulation_doc_id': manipulation_doc_id,
                'relevant_docs': relevant_docs
            })

    # Analyze other documents accuracy
    other_correct_before = 0
    other_correct_after = 0
    other_changed_queries = []

    # Track relevant docs accuracy for other queries
    other_relevant_correct_before = 0
    other_relevant_correct_after = 0

    for i, (query, query_key) in enumerate(zip(other_queries, other_query_keys)):
        # Keep expected_doc_id as string for consistent comparison with dataset
        expected_doc_id = str(result_dict[query_key].get('correct_doc_id'))

        baseline_doc_ids = extract_doc_ids_from_output([other_baseline[i]])
        manipulation_doc_ids = extract_doc_ids_from_output([other_manipulation[i]])

        baseline_correct = baseline_doc_ids[0] == expected_doc_id
        manipulation_correct = manipulation_doc_ids[0] == expected_doc_id

        if baseline_correct:
            other_correct_before += 1
        if manipulation_correct:
            other_correct_after += 1

        # Check if predictions match any relevant document for other queries
        relevant_docs = result_dict[query_key].get('relevant_docs', [])
        # Convert to strings for consistent comparison
        relevant_docs_str = [str(doc) for doc in relevant_docs]

        baseline_relevant_correct = baseline_doc_ids[0] is not None and baseline_doc_ids[0] in relevant_docs_str
        manipulation_relevant_correct = manipulation_doc_ids[0] is not None and manipulation_doc_ids[0] in relevant_docs_str

        if baseline_relevant_correct:
            other_relevant_correct_before += 1
        if manipulation_relevant_correct:
            other_relevant_correct_after += 1

        # Track queries that changed from correct to incorrect
        if baseline_correct and not manipulation_correct:
            # Extract document IDs from predictions for cleaner output (with error handling)
            try:
                baseline_doc_id = extract_doc_ids_from_output([other_baseline[i]])[0]
            except (IndexError, TypeError):
                baseline_doc_id = None

            try:
                manipulation_doc_id = extract_doc_ids_from_output([other_manipulation[i]])[0]
            except (IndexError, TypeError):
                manipulation_doc_id = None

            # Get relevant docs from result_dict
            relevant_docs = result_dict[query_key].get('relevant_docs', [])

            other_changed_queries.append({
                'query_key': query_key,  # Use query key for tracking
                'query_input': query,    # Keep input for reference
                'baseline_pred': other_baseline[i],
                'manipulation_pred': other_manipulation[i],
                'expected_doc_id': expected_doc_id,
                'baseline_doc_id': baseline_doc_id,
                'manipulation_doc_id': manipulation_doc_id,
                'relevant_docs': relevant_docs
            })

    # Calculate summary statistics
    target_metrics = format_accuracy_results(
        target_correct_before, target_correct_after, len(target_queries)
    )
    other_metrics = format_accuracy_results(
        other_correct_before, other_correct_after, len(other_queries)
    )

    # Calculate relevant docs accuracy metrics
    target_relevant_metrics = format_accuracy_results(
        target_relevant_correct_before, target_relevant_correct_after, len(target_queries)
    )
    other_relevant_metrics = format_accuracy_results(
        other_relevant_correct_before, other_relevant_correct_after, len(other_queries)
    )

    # Overall metrics
    total_correct_before = target_correct_before + other_correct_before
    total_correct_after = target_correct_after + other_correct_after
    total_queries = len(target_queries) + len(other_queries)

    overall_metrics = format_accuracy_results(
        total_correct_before, total_correct_after, total_queries
    )

    # Overall relevant docs metrics
    total_relevant_correct_before = target_relevant_correct_before + other_relevant_correct_before
    total_relevant_correct_after = target_relevant_correct_after + other_relevant_correct_after

    overall_relevant_metrics = format_accuracy_results(
        total_relevant_correct_before, total_relevant_correct_after, total_queries
    )

    return {
        'target_results': {
            'total': len(target_queries),
            'correct_before': target_correct_before,
            'correct_after': target_correct_after,
            'newly_incorrect': len(target_changed_queries),
            'metrics': target_metrics
        },
        'other_results': {
            'total': len(other_queries),
            'correct_before': other_correct_before,
            'correct_after': other_correct_after,
            'newly_incorrect': len(other_changed_queries),
            'metrics': other_metrics
        },
        'overall_results': {
            'total': total_queries,
            'correct_before': total_correct_before,
            'correct_after': total_correct_after,
            'metrics': overall_metrics
        },
        'target_relevant_results': {
            'total': len(target_queries),
            'correct_before': target_relevant_correct_before,
            'correct_after': target_relevant_correct_after,
            'metrics': target_relevant_metrics
        },
        'other_relevant_results': {
            'total': len(other_queries),
            'correct_before': other_relevant_correct_before,
            'correct_after': other_relevant_correct_after,
            'metrics': other_relevant_metrics
        },
        'overall_relevant_results': {
            'total': total_queries,
            'correct_before': total_relevant_correct_before,
            'correct_after': total_relevant_correct_after,
            'metrics': overall_relevant_metrics
        },
        'queries_changed': {
            'target_queries_became_incorrect': target_changed_queries,
            'other_queries_became_incorrect': other_changed_queries,
            'n_target_changed': len(target_changed_queries),
            'n_other_changed': len(other_changed_queries)
        },
        'summary': {
            'target_accuracy_before': target_metrics['accuracy_before'] * 100,
            'target_accuracy_after': target_metrics['accuracy_after'] * 100,
            'target_accuracy_drop': target_metrics['accuracy_drop_pct'] * 100,
            'other_accuracy_before': other_metrics['accuracy_before'] * 100,
            'other_accuracy_after': other_metrics['accuracy_after'] * 100,
            'other_accuracy_drop': other_metrics['accuracy_drop_pct'] * 100,
            'target_relevant_accuracy_before': target_relevant_metrics['accuracy_before'] * 100,
            'target_relevant_accuracy_after': target_relevant_metrics['accuracy_after'] * 100,
            'target_relevant_accuracy_drop': target_relevant_metrics['accuracy_drop_pct'] * 100,
            'other_relevant_accuracy_before': other_relevant_metrics['accuracy_before'] * 100,
            'other_relevant_accuracy_after': other_relevant_metrics['accuracy_after'] * 100,
            'other_relevant_accuracy_drop': other_relevant_metrics['accuracy_drop_pct'] * 100,
            'other_newly_incorrect_pct': (len(other_changed_queries) / len(other_queries)) * 100 if other_queries else 0
        }
    }


def save_incorrect_queries_analysis(target_changed_queries: List[Dict[str, Any]],
                                   other_changed_queries: List[Dict[str, Any]],
                                   target_doc_id: Union[int, str],
                                   frequency_threshold: float,
                                   replacement_type: str,
                                   output_path: Union[str, Path],
                                   verbose: bool = True) -> Optional[Path]:
    """
    Save detailed analysis of queries that became incorrect after neuron manipulation.

    Args:
        target_changed_queries: List of target queries that became incorrect
        other_changed_queries: List of other queries that became incorrect
        target_doc_id: Document ID being analyzed
        frequency_threshold: Frequency threshold used
        replacement_type: Type of neuron replacement
        output_path: Path to save the JSON file
        verbose: Whether to print saving information

    Returns:
        Path to saved file, or None if no incorrect queries found
    """
    # Skip if no incorrect queries found
    if not target_changed_queries and not other_changed_queries:
        if verbose:
            print(" No incorrect queries to save - all predictions remained correct")
        return None

    # Prepare the output data structure
    incorrect_queries_data = {
        "analysis_metadata": {
            "target_doc_id": str(target_doc_id),
            "frequency_threshold": frequency_threshold,
            "replacement_type": replacement_type,
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S")
        },
        "target_queries": {},
        "other_queries": {},
        "summary": {
            "total_target_incorrect": len(target_changed_queries),
            "total_other_incorrect": len(other_changed_queries)
        }
    }

    # Process target queries
    for query_data in target_changed_queries:
        query_key = query_data['query_key']
        incorrect_queries_data["target_queries"][query_key] = {
            "input": query_data['query_input'],
            "expected_correct_answer": query_data['expected_doc_id'],
            "wrong_answer": query_data.get('manipulation_doc_id', 'unknown'),
            "relevant_docs": query_data.get('relevant_docs', [])
        }

    # Process other queries
    for query_data in other_changed_queries:
        query_key = query_data['query_key']
        incorrect_queries_data["other_queries"][query_key] = {
            "input": query_data['query_input'],
            "expected_correct_answer": query_data['expected_doc_id'],
            "wrong_answer": query_data.get('manipulation_doc_id', 'unknown'),
            "relevant_docs": query_data.get('relevant_docs', [])
        }

    # Save to file
    try:
        output_path = Path(output_path)
        safe_file_write(output_path, incorrect_queries_data, overwrite=True)

        if verbose:
            print(f" Saved incorrect queries analysis: {output_path}")
            print(f"   Target queries affected: {len(target_changed_queries)}")
            print(f"   Other queries affected: {len(other_changed_queries)}")

        return output_path

    except Exception as e:
        if verbose:
            print(f" Error saving incorrect queries analysis: {e}")
        return None


def analyze_document_specific_neurons(model,
                                    tokenizer,
                                    result_dict: Dict[str, Any],
                                    stats: Dict[str, Any],
                                    target_doc_id: Union[int, str],
                                    layer_indices: Optional[List[int]] = None,
                                    frequency_threshold: Optional[Union[int, float]] = None,
                                    other_queries_sample: Optional[int] = 100,
                                    verbose: bool = True,
                                    return_activation_vectors: bool = False,
                                    replacement_type: str = 'zero_out',
                                    detailed_stats: Optional[Dict[str, Any]] = None,
                                    training_result_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Complete analysis pipeline for document-specific neurons.

    This is the main analysis function that replicates the notebook functionality.

    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        result_dict: Dictionary with query data for evaluation (test data when in test mode)
        stats: Neuron activation statistics
        target_doc_id: The document ID to analyze
        layer_indices: List of layer indices to analyze, or None for all layers
        frequency_threshold: Only keep neurons activated for < threshold queries
        other_queries_sample: Number of other queries to test for side effects (None = all)
        verbose: Print detailed information
        return_activation_vectors: If True, return activation vectors for analysis
        replacement_type: Strategy for neuron manipulation ('zero_out' or 'mean_value')
        detailed_stats: Statistics dictionary required when replacement_type='mean_value'
        training_result_dict: Dictionary with training data for neuron filtering (if None, uses result_dict)

    Returns:
        Dictionary with complete analysis results and optionally activation vectors
    """
    print('ENTERED HERE')
    if layer_indices == -1:
        layer_indices = None  # Use all layers

    # Validate parameters
    if replacement_type == 'mean_value' and detailed_stats is None:
        raise ValueError("detailed_stats is required when replacement_type='mean_value'")

    # Use training data for neuron filtering, evaluation data for testing
    filtering_result_dict = training_result_dict if training_result_dict is not None else result_dict

    # Step 1: Collect document-specific neurons (use training data for filtering)
    if verbose:
        print(f"=== Analyzing Document ID: {target_doc_id} ===")
        print(f"Frequency threshold: {frequency_threshold}")
        if frequency_threshold is not None and frequency_threshold <= 1:
            print(f"  (Using percentage threshold: {frequency_threshold*100:.1f}% of total queries)")
        elif frequency_threshold is not None:
            print(f"  (Using absolute threshold: {frequency_threshold} queries)")
        print(f"Layer indices: {layer_indices if layer_indices else 'all (17-23)'}")
        print(f"Replacement strategy: {replacement_type}")
        if training_result_dict is not None:
            print(f" Using training data for neuron filtering, test data for evaluation")

    neuron_data = collect_document_specific_neurons(
        filtering_result_dict, stats, target_doc_id, layer_indices, frequency_threshold
    )

    if verbose:
        print(f"\nFound {len(neuron_data['target_queries'])} queries with target document")

        total_neurons = sum(len(neurons) for neurons in neuron_data['neurons_by_layer'].values())
        filtered_neurons = sum(len(neurons) for neurons in neuron_data['filtered_neurons'].values())

        print(f"Total neurons activated: {total_neurons}")
        print(f"Neurons after frequency filtering: {filtered_neurons}")

        print("\nNeurons by layer (after filtering):")
        for layer_key, neurons in neuron_data['filtered_neurons'].items():
            if neurons:
                print(f"  {layer_key}: {len(neurons)} neurons")
                if verbose and len(neurons) <= 10:
                    print(f"    Neuron IDs: {neurons}")
                elif verbose:
                    print(f"    Sample neuron IDs: {neurons[:10]}...")

    # Step 2: Test the effect of toggling these neurons
    if verbose:
        print(f"\n=== Testing neuron toggling effects ===")

    test_results = test_document_neuron_effects(
        model, tokenizer, result_dict, target_doc_id,
        neuron_data['filtered_neurons'],
        other_queries_sample=other_queries_sample,
        batch_size=256,
        replacement_type=replacement_type,
        detailed_stats=detailed_stats,
        return_activation_vectors=return_activation_vectors
    )

    # Step 3: Print summary
    if verbose:
        print(f"\n=== Results Summary ===")
        summary = test_results['summary']

        print(f"Target Document {target_doc_id}:")
        print(f"  Accuracy before: {summary['target_accuracy_before']:.1f}%")
        print(f"  Accuracy after:  {summary['target_accuracy_after']:.1f}%")
        print(f"  Accuracy drop:   {summary['target_accuracy_drop']:.1f}%")
        print(f"  Queries that became incorrect: {test_results['queries_changed']['n_target_changed']}")

        print(f"\nOther Documents:")
        print(f"  Accuracy before: {summary['other_accuracy_before']:.1f}%")
        print(f"  Accuracy after:  {summary['other_accuracy_after']:.1f}%")
        print(f"  Accuracy drop:   {summary['other_accuracy_drop']:.1f}%")
        print(f"  Newly incorrect: {summary['other_newly_incorrect_pct']:.1f}%")
        print(f"  Queries that became incorrect: {test_results['queries_changed']['n_other_changed']}")

        # Interpretation
        print(f"\n=== Interpretation ===")
        if summary['target_accuracy_drop'] > 10:
            print("These neurons appear important for the target document")
        else:
            print("These neurons may not be crucial for the target document")

        if summary['other_accuracy_drop'] < 5:
            print("Minimal impact on other documents - neurons seem document-specific")
        elif summary['other_accuracy_drop'] < 15:
            print("Moderate impact on other documents - neurons partially document-specific")
        else:
            print("High impact on other documents - neurons not document-specific")

    # Combine all results
    return {
        'target_doc_id': target_doc_id,
        'neuron_data': neuron_data,
        'test_results': test_results,
        'parameters': {
            'layer_indices': layer_indices,
            'frequency_threshold': frequency_threshold,
            'other_queries_sample': other_queries_sample,
            'replacement_type': replacement_type,
            'detailed_stats': detailed_stats is not None
        }
    }


def compare_multiple_documents(model,
                             tokenizer,
                             result_dict: Dict[str, Any],
                             stats: Dict[str, Any],
                             doc_ids: List[Union[int, str]],
                             layer_indices: Optional[List[int]] = None,
                             frequency_threshold: Optional[Union[int, float]] = None,
                             other_queries_sample: int = 50,
                             verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze multiple documents and compare their neuron specificity.

    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        result_dict: Dictionary with query data and activated neurons
        stats: Neuron activation statistics
        doc_ids: List of document IDs to analyze
        layer_indices: List of layer indices to analyze, or None for all layers
        frequency_threshold: Only keep neurons activated for < threshold queries
        other_queries_sample: Number of other queries to test for side effects (None = all)
        verbose: Print detailed information for each document

    Returns:
        Dictionary with comparative analysis
    """
    results = {}

    for doc_id in doc_ids:
        print(f"\n{'='*50}")
        print(f"Analyzing Document {doc_id}")
        print(f"{'='*50}")

        try:
            result = analyze_document_specific_neurons(
                model, tokenizer, result_dict, stats, doc_id,
                layer_indices, frequency_threshold, other_queries_sample, verbose
            )
            results[str(doc_id)] = result
        except Exception as e:
            print(f"Error analyzing document {doc_id}: {e}")
            results[str(doc_id)] = None

    # Summary comparison
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")

    print(f"{'Doc ID':<8} {'Queries':<8} {'Neurons':<8} {'Target Drop':<12} {'Other Drop':<11} {'Specificity'}")
    print("-" * 70)

    for doc_id, result in results.items():
        if result is None:
            continue

        n_queries = len(result['neuron_data']['target_queries'])
        n_neurons = sum(len(neurons) for neurons in result['neuron_data']['filtered_neurons'].values())
        target_drop = result['test_results']['summary']['target_accuracy_drop']
        other_drop = result['test_results']['summary']['other_accuracy_drop']

        # Calculate specificity score (high target drop, low other drop = high specificity)
        if target_drop > 0:
            specificity = target_drop / (other_drop + 1e-6)  # Add small epsilon to avoid division by zero
        else:
            specificity = 0

        print(f"{doc_id:<8} {n_queries:<8} {n_neurons:<8} {target_drop:<11.1f}% {other_drop:<10.1f}% {specificity:<10.2f}")

    return results