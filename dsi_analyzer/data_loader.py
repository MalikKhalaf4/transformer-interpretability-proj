"""
Data loading utilities for DSI document analysis.

This module provides classes and functions for loading and managing TriviaQA
datasets, including train, validation, and test data.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from collections import defaultdict

from .utils import safe_file_read, create_default_paths


class QuestionsDataset(Dataset):
    """
    PyTorch Dataset for TriviaQA questions and relevant documents.

    This dataset handles question-answer pairs with their corresponding relevant
    document IDs for training and evaluation of DSI models.
    """

    def __init__(self, inputs: List[str], targets: List[List[str]], ids: List[str]):
        """
        Initialize the dataset with questions, targets, and IDs.

        Args:
            inputs: List of input questions/queries
            targets: List of relevant document lists for each question
            ids: List of unique identifiers for each question
        """
        if not (len(inputs) == len(targets) == len(ids)):
            raise ValueError("Inputs, targets, and ids must have the same length")

        self.inputs = inputs
        self.targets = targets
        self.ids = ids

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[str, List[str], str]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (input_text, target_docs, query_id)
        """
        return self.inputs[idx], self.targets[idx], self.ids[idx]

    def collate_fn(self, batch: List[Tuple[str, List[str], str]]) -> Tuple[List[str], List[List[str]], List[str]]:
        """
        Collate function for DataLoader to handle batching.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Tuple of (input_texts, target_texts, ids)
        """
        input_texts, target_texts, ids = zip(*batch)
        return list(input_texts), list(target_texts), list(ids)


class TriviaQADataLoader:
    """
    Data loader and manager for TriviaQA datasets.

    Handles loading train, validation, and test data, creating datasets and
    data loaders with appropriate configurations.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing TriviaQA data files
        """
        if data_dir is None:
            paths = create_default_paths()
            data_dir = paths['trivia_qa_path']

        self.data_dir = Path(data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._combined_data = None

    def load_train_data(self, file_name: str = "train_queries_trivia_qa.json") -> List[Dict[str, Any]]:
        """
        Load training data from JSON file.

        Args:
            file_name: Name of the training data file

        Returns:
            List of training data entries

        Raises:
            FileNotFoundError: If training data file is not found
        """
        file_path = self.data_dir / file_name
        try:
            self.train_data = safe_file_read(file_path)
            return self.train_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data file not found: {file_path}")

    def load_val_data(self, file_name: str = "val_queries_trivia_qa.json") -> List[Dict[str, Any]]:
        """
        Load validation data from JSON file.

        Args:
            file_name: Name of the validation data file

        Returns:
            List of validation data entries

        Raises:
            FileNotFoundError: If validation data file is not found
        """
        file_path = self.data_dir / file_name
        try:
            self.val_data = safe_file_read(file_path)
            return self.val_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Validation data file not found: {file_path}")

    def load_test_data(self, file_name: str = "test_queries_trivia_qa.json") -> List[Dict[str, Any]]:
        """
        Load test data from JSON file.

        Args:
            file_name: Name of the test data file

        Returns:
            List of test data entries

        Raises:
            FileNotFoundError: If test data file is not found
        """
        file_path = self.data_dir / file_name
        try:
            self.test_data = safe_file_read(file_path)
            return self.test_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Test data file not found: {file_path}")

    def load_all_data(self,
                     train_file: str = "train_queries_trivia_qa.json",
                     val_file: str = "val_queries_trivia_qa.json",
                     test_file: str = "test_queries_trivia_qa.json",
                     load_test: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all available data splits.

        Args:
            train_file: Training data filename
            val_file: Validation data filename
            test_file: Test data filename
            load_test: Whether to load test data

        Returns:
            Dictionary with loaded data splits
        """
        data_splits = {}

        # Load train and validation data
        data_splits['train'] = self.load_train_data(train_file)
        data_splits['val'] = self.load_val_data(val_file)

        # Optionally load test data
        if load_test:
            try:
                data_splits['test'] = self.load_test_data(test_file)
            except FileNotFoundError:
                print(f"Warning: Test data file {test_file} not found, skipping test data")
                data_splits['test'] = []

        return data_splits

    def get_combined_data(self, include_test: bool = False) -> List[Dict[str, Any]]:
        """
        Get combined train and validation data (and optionally test data).

        Args:
            include_test: Whether to include test data in combination

        Returns:
            Combined data list

        Raises:
            ValueError: If required data is not loaded
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Train and validation data must be loaded first")

        combined = self.train_data + self.val_data

        if include_test and self.test_data is not None:
            combined.extend(self.test_data)

        self._combined_data = combined
        return combined

    def create_dataset(self,
                      data: Optional[List[Dict[str, Any]]] = None,
                      use_combined: bool = True) -> QuestionsDataset:
        """
        Create a QuestionsDataset from loaded data.

        Args:
            data: Specific data to use, or None to use combined data
            use_combined: Whether to use combined train+val data if data is None

        Returns:
            QuestionsDataset instance

        Raises:
            ValueError: If no data is available
        """
        if data is None:
            if use_combined:
                if self._combined_data is None:
                    data = self.get_combined_data()
                else:
                    data = self._combined_data
            else:
                raise ValueError("No data provided and use_combined=False")

        # Extract queries, relevant docs, and IDs
        queries = [entry['query'] for entry in data]
        ground_truths = [entry['relevant_docs'] for entry in data]
        query_ids = [entry['id'] for entry in data]

        return QuestionsDataset(queries, ground_truths, query_ids)

    def create_dataloader(self,
                         dataset: Optional[QuestionsDataset] = None,
                         batch_size: int = 16,
                         shuffle: bool = False,
                         num_workers: int = 0,
                         **kwargs) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset.

        Args:
            dataset: QuestionsDataset to use, or None to create from combined data
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            **kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader instance
        """
        if dataset is None:
            dataset = self.create_dataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            **kwargs
        )

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dictionary with data statistics
        """
        stats = {}

        if self.train_data is not None:
            stats['train_size'] = len(self.train_data)

        if self.val_data is not None:
            stats['val_size'] = len(self.val_data)

        if self.test_data is not None:
            stats['test_size'] = len(self.test_data)

        if self._combined_data is not None:
            stats['combined_size'] = len(self._combined_data)

            # Analyze document frequency
            doc_counts = defaultdict(int)
            for entry in self._combined_data:
                for doc_id in entry.get('relevant_docs', []):
                    doc_counts[str(doc_id)] += 1

            stats['unique_documents'] = len(doc_counts)
            stats['total_doc_references'] = sum(doc_counts.values())
            stats['avg_docs_per_query'] = stats['total_doc_references'] / len(self._combined_data)

            # Most popular documents
            popular_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
            stats['top_10_documents'] = popular_docs[:10]

        return stats

    def create_result_dict(self, data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Create a result dictionary mapping queries to their information.

        This creates the format expected by the analysis functions, mapping
        query text to document IDs and other metadata.

        Args:
            data: Data to use, or None for combined data

        Returns:
            Dictionary mapping queries to their information
        """
        if data is None:
            if self._combined_data is None:
                data = self.get_combined_data()
            else:
                data = self._combined_data

        result_dict = defaultdict(lambda: {})

        for entry in data:
            query = entry['query']
            relevant_docs = entry.get('relevant_docs', [])

            # For DSI analysis, we typically focus on the first relevant document
            # as the "correct" document ID
            correct_doc_id = relevant_docs[0] if relevant_docs else None

            result_dict[query] = {
                'query_id': entry.get('id'),
                'relevant_docs': relevant_docs,
                'correct_doc_id': correct_doc_id,
                'activated_neurons': {}  # Will be populated during analysis
            }

        return dict(result_dict)


def load_trivia_qa_data(data_dir: Optional[Union[str, Path]] = None,
                       batch_size: int = 16,
                       include_test: bool = False,
                       shuffle: bool = False) -> Tuple[DataLoader, Dict[str, Any], TriviaQADataLoader]:
    """
    Convenience function to load TriviaQA data and create a DataLoader.

    Args:
        data_dir: Directory containing TriviaQA data files
        batch_size: Batch size for DataLoader
        include_test: Whether to include test data
        shuffle: Whether to shuffle the data

    Returns:
        Tuple of (DataLoader, data_statistics, TriviaQADataLoader instance)
    """
    loader = TriviaQADataLoader(data_dir)

    # Load data
    loader.load_all_data(load_test=include_test)
    loader.get_combined_data(include_test=include_test)

    # Create dataset and dataloader
    dataset = loader.create_dataset()
    dataloader = loader.create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Get statistics
    statistics = loader.get_data_statistics()

    return dataloader, statistics, loader


def create_popular_docs_list(result_dict: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
    """
    Create a list of popular documents sorted by query frequency.

    Args:
        result_dict: Dictionary mapping queries to document information

    Returns:
        List of (document_id, query_count) tuples sorted by frequency
    """
    doc_id_counts = defaultdict(int)

    for query, entry in result_dict.items():
        doc_id = entry.get("correct_doc_id")
        if doc_id is not None:
            doc_id_counts[str(doc_id)] += 1

    # Sort by frequency (descending)
    popular_docs = sorted(doc_id_counts.items(), key=lambda x: x[1], reverse=True)

    return popular_docs