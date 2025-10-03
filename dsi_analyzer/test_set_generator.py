"""
Test set generator for DSI neuron analysis.

This module handles filtering test queries to only include those the model
answers correctly, then converts them to the evaluation format.
"""

import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

from .utils import safe_file_read, safe_file_write, get_model_device, batch_list, extract_doc_ids_from_output


def load_test_queries(test_queries_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load test queries from JSON file.

    Args:
        test_queries_path: Path to test_queries_trivia_qa.json

    Returns:
        List of test query dictionaries
    """
    test_queries = safe_file_read(test_queries_path)
    return test_queries


def filter_correctly_answered_queries(model,
                                    tokenizer,
                                    test_queries: List[Dict[str, Any]],
                                    batch_size: int = 32,
                                    verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Filter test queries to only include those the model answers correctly.
    Uses the same batched inference approach as the analysis pipeline for consistency.

    Args:
        model: The DSI model
        tokenizer: Model tokenizer
        test_queries: List of test query dictionaries
        batch_size: Batch size for inference
        verbose: Whether to print progress

    Returns:
        List of correctly answered test queries
    """
    if verbose:
        print(f" Filtering {len(test_queries)} test queries to find correctly answered ones...")
        print(f" Using batched inference (batch_size={batch_size}) for consistency with analysis pipeline")

    correctly_answered = []
    total_queries = len(test_queries)

    # Extract just the query texts for batched inference
    query_texts = [item['query'] for item in test_queries]

    # Run batched inference using the same approach as analysis pipeline
    query_batches = batch_list(query_texts, batch_size)
    all_predictions = []

    if verbose:
        progress_bar = tqdm(query_batches, desc="Running batched inference")
    else:
        progress_bar = query_batches

    for batch_queries in progress_bar:
        try:
            # Run inference with direct string input (same as analysis pipeline)
            with torch.no_grad():
                logits = model(batch_queries)

            # Get predictions (handle full sequences like analysis pipeline)
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

        except Exception as e:
            if verbose:
                print(f" Error processing batch: {e}")
            # Add None predictions for failed batch
            all_predictions.extend([None] * len(batch_queries))
            continue

    # Now process predictions to filter correctly answered queries
    if verbose:
        print(f"Processing {len(all_predictions)} predictions to find correct answers...")

    for i, (query_item, prediction_text) in enumerate(zip(test_queries, all_predictions)):
        if prediction_text is None:
            continue

        relevant_docs = query_item['relevant_docs']

        # Extract document ID from model output
        extracted_doc_ids = extract_doc_ids_from_output([prediction_text])
        extracted_doc_id = extracted_doc_ids[0] if extracted_doc_ids[0] is not None else None

        if verbose and i < 5:  # Debug output for first few predictions
            print(f"  Query {i}: {query_item['query'][:50]}...")
            print(f"    Raw output: {repr(prediction_text)}")
            print(f"    Extracted doc ID: {extracted_doc_id}")
            print(f"    Relevant docs: {relevant_docs}")

        # Check if extracted document ID matches any relevant document
        # Note: Test queries can have multiple correct answers in relevant_docs
        # If model predicts ANY of them, it's considered correct
        if extracted_doc_id and extracted_doc_id in relevant_docs:
            # Model answered correctly - store the specific doc ID the model predicted
            query_item['prediction'] = extracted_doc_id  # Store the actual prediction
            correctly_answered.append(query_item)

            if verbose and i < 5:
                print(f"    Match found!")
        elif verbose and i < 5:
            print(f"    No match")

    if verbose:
        print(f" Found {len(correctly_answered)}/{total_queries} correctly answered queries ({len(correctly_answered)/total_queries*100:.1f}%)")

    return correctly_answered


def convert_to_evaluation_format(correctly_answered_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert correctly answered test queries to evaluation format.

    Args:
        correctly_answered_queries: List of correctly answered test query dicts

    Returns:
        Dictionary in the format expected by the analysis pipeline (matching training data format)
    """
    evaluation_dict = {}

    for query_item in correctly_answered_queries:
        query_id = query_item['id']  # Use query ID as key (e.g., "QTest0")
        query_text = query_item['query']
        correct_doc_id = query_item['prediction']  # The extracted document ID from model prediction
        relevant_docs = query_item['relevant_docs']  # Original relevant docs from test file

        # Format to match training data structure: key = query_id, value contains input field
        evaluation_dict[query_id] = {
            'input': query_text,  # Query text goes under 'input' field
            'correct_doc_id': correct_doc_id,  # The specific doc ID model predicted correctly
            'relevant_docs': relevant_docs,  # All possible correct answers
            'activated_neurons': {}  # Empty - not needed for test evaluation
        }

    return evaluation_dict


def generate_test_evaluation_data(model,
                                tokenizer,
                                test_queries_path: Union[str, Path],
                                output_path: Optional[Union[str, Path]] = None,
                                batch_size: int = 128,
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Complete pipeline to generate test evaluation data.

    Args:
        model: The DSI model
        tokenizer: Model tokenizer
        test_queries_path: Path to test_queries_trivia_qa.json
        output_path: Path to save the filtered test evaluation data
        batch_size: Batch size for inference
        verbose: Whether to print progress

    Returns:
        Dictionary with test evaluation data
    """
    # Load test queries
    test_queries = load_test_queries(test_queries_path)

    # Filter to correctly answered queries only
    correctly_answered = filter_correctly_answered_queries(
        model, tokenizer, test_queries, batch_size, verbose
    )

    # Convert to evaluation format
    evaluation_data = convert_to_evaluation_format(correctly_answered)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_file_write(output_path, evaluation_data, overwrite=True)

        if verbose:
            print(f" Saved test evaluation data to: {output_path}")

    return evaluation_data


def check_test_evaluation_data_exists(stats_path: Union[str, Path]) -> bool:
    """
    Check if test evaluation data already exists.

    Args:
        stats_path: Path to statistics directory

    Returns:
        True if activated_neurons_test.json exists
    """
    test_file = Path(stats_path) / "activated_neurons_test.json"
    return test_file.exists()


def get_test_data_info(stats_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about existing test evaluation data.

    Args:
        stats_path: Path to statistics directory

    Returns:
        Dictionary with test data information
    """
    test_file = Path(stats_path) / "activated_neurons_test.json"

    if not test_file.exists():
        return {
            'exists': False,
            'path': str(test_file),
            'num_queries': 0
        }

    try:
        test_data = safe_file_read(test_file)
        return {
            'exists': True,
            'path': str(test_file),
            'num_queries': len(test_data),
            'sample_queries': list(test_data.keys())[:5]
        }
    except Exception as e:
        return {
            'exists': True,
            'path': str(test_file),
            'num_queries': 0,
            'error': str(e)
        }