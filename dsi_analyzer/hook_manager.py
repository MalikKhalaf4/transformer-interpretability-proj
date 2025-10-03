"""
Hook management for activation extraction and neuron manipulation.

This module provides classes and functions for creating and managing hooks
that extract activations and manipulate neuron values in DSI models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import defaultdict

from .utils import validate_layer_indices, batch_list, get_model_device


class ActivationExtractor:
    """
    Class for extracting neuron activations from transformer models.

    Uses hooks to capture intermediate activations during forward passes.
    """

    def __init__(self, layer_indices: Optional[List[int]] = None):
        """
        Initialize the activation extractor.

        Args:
            layer_indices: List of layer indices to extract from, or None for default (17-23)
        """
        self.layer_indices = validate_layer_indices(layer_indices)
        self.hooks = {}
        self.activations = {}

    def create_extraction_hooks(self, model) -> Dict[str, Any]:
        """
        Create hooks for extracting MLP activations.

        Args:
            model: The transformer model to hook

        Returns:
            Dictionary of hook handles
        """
        hooks = {}

        for layer_id in self.layer_indices:
            hook_name = f'decoder.{layer_id}.mlp.hook_post'

            def make_hook(layer_id=layer_id):
                def hook_fn(activation, hook):
                    self.activations[f'layer_{layer_id}'] = activation.clone()
                return hook_fn

            hooks[hook_name] = model.add_hook(hook_name, make_hook())

        self.hooks = hooks
        return hooks

    def extract_activated_neurons(self) -> Dict[str, torch.Tensor]:
        """
        Extract indices of activated neurons from stored activations.

        Returns:
            Dictionary mapping layer names to activated neuron indices
        """
        result_dict = {}

        for layer_key, activation in self.activations.items():
            # Find neurons with positive activations (ReLU activated)
            activated_neuron_indices = (activation > 0).nonzero(as_tuple=False)
            result_dict[layer_key] = activated_neuron_indices

        return result_dict

    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for hook in self.hooks.values():
            if hook is not None:
                hook.remove()
        self.hooks.clear()


class NeuronManipulator:
    """
    Class for manipulating neuron activations during inference.

    Supports different manipulation strategies like zeroing out or replacing
    with mean values.
    """

    def __init__(self,
                 replacement_type: str = 'zero_out',
                 detailed_stats: Optional[Dict[str, Any]] = None):
        """
        Initialize the neuron manipulator.

        Args:
            replacement_type: Strategy for manipulation ('zero_out' or 'mean_value')
            detailed_stats: Statistics for mean_value replacement (required if using mean_value)

        Raises:
            ValueError: If mean_value is used without detailed_stats
        """
        if replacement_type not in ['zero_out', 'mean_value']:
            raise ValueError("replacement_type must be 'zero_out' or 'mean_value'")

        if replacement_type == 'mean_value' and detailed_stats is None:
            raise ValueError("detailed_stats is required when replacement_type='mean_value'")

        self.replacement_type = replacement_type
        self.detailed_stats = detailed_stats
        self.hooks = {}

    def create_manipulation_hooks(self,
                                neurons_by_layer: Dict[str, List[int]]) -> List[Tuple[str, Any]]:
        """
        Create hooks for manipulating neuron activations in context manager format.

        Args:
            neurons_by_layer: Dictionary mapping layer names to neuron indices to manipulate

        Returns:
            List of (hook_name, hook_function) tuples for use with model.hooks() context manager
        """
        hooks = []

        for layer_key, neuron_indices in neurons_by_layer.items():
            if not neuron_indices:  # Skip empty lists
                continue

            # Extract layer ID from layer_key (e.g., 'layer_17' -> 17)
            layer_id = int(layer_key.split('_')[1])
            hook_name = f'decoder.{layer_id}.mlp.hook_post'

            def make_manipulation_hook(neuron_indices=neuron_indices, layer_key=layer_key):
                def hook_fn(activation, hook):
                    return self._manipulate_activations(activation, neuron_indices, layer_key)
                return hook_fn

            hooks.append((hook_name, make_manipulation_hook()))

        return hooks

    def _manipulate_activations(self,
                              activation: torch.Tensor,
                              neuron_indices: List[int],
                              layer_key: str) -> torch.Tensor:
        """
        Manipulate activations according to the replacement strategy.

        Args:
            activation: Input activation tensor
            neuron_indices: List of neuron indices to manipulate
            layer_key: Layer identifier for mean value lookup

        Returns:
            Modified activation tensor
        """
        modified_activation = activation.clone()

        if self.replacement_type == 'zero_out':
            # Set specified neurons to zero
            # Handle both 2D [batch, d_model] and 3D [batch, seq_len, d_model] tensors
            for neuron_idx in neuron_indices:
                if len(modified_activation.shape) == 3:
                    modified_activation[:, :, neuron_idx] = 0.0
                else:
                    modified_activation[:, neuron_idx] = 0.0

        elif self.replacement_type == 'mean_value':
            # Replace with mean values from statistics
            if self.detailed_stats is None:
                raise ValueError("detailed_stats required for mean_value replacement")

            layer_stats = self.detailed_stats.get(layer_key, {})
            mean_values = layer_stats.get('mean', {})

            for neuron_idx in neuron_indices:
                mean_value = mean_values.get(str(neuron_idx), 0.0)
                if len(modified_activation.shape) == 3:
                    modified_activation[:, :, neuron_idx] = mean_value
                else:
                    modified_activation[:, neuron_idx] = mean_value

        return modified_activation

    def remove_hooks(self):
        """Remove all manipulation hooks from the model."""
        for hook in self.hooks.values():
            if hook is not None:
                hook.remove()
        self.hooks.clear()


class MultiNeuronHookManager:
    """
    Manager for handling multiple types of hooks simultaneously.

    Can manage both extraction and manipulation hooks together.
    """

    def __init__(self):
        """Initialize the multi-hook manager."""
        self.extractors = {}
        self.manipulators = {}
        self.active_hooks = {}

    def add_extractor(self,
                     name: str,
                     model,
                     layer_indices: Optional[List[int]] = None) -> ActivationExtractor:
        """
        Add an activation extractor.

        Args:
            name: Name for this extractor
            model: Model to extract from
            layer_indices: Layers to extract from

        Returns:
            Created ActivationExtractor instance
        """
        extractor = ActivationExtractor(layer_indices)
        hooks = extractor.create_extraction_hooks(model)

        self.extractors[name] = extractor
        self.active_hooks[f'extractor_{name}'] = hooks

        return extractor

    def add_manipulator(self,
                       name: str,
                       model,
                       neurons_by_layer: Dict[str, List[int]],
                       replacement_type: str = 'zero_out',
                       detailed_stats: Optional[Dict[str, Any]] = None) -> NeuronManipulator:
        """
        Add a neuron manipulator.

        Args:
            name: Name for this manipulator
            model: Model to manipulate
            neurons_by_layer: Neurons to manipulate by layer
            replacement_type: Type of manipulation
            detailed_stats: Statistics for mean_value replacement

        Returns:
            Created NeuronManipulator instance
        """
        manipulator = NeuronManipulator(replacement_type, detailed_stats)
        hooks = manipulator.create_manipulation_hooks(model, neurons_by_layer)

        self.manipulators[name] = manipulator
        self.active_hooks[f'manipulator_{name}'] = hooks

        return manipulator

    def remove_all_hooks(self):
        """Remove all active hooks."""
        for extractor in self.extractors.values():
            extractor.remove_hooks()

        for manipulator in self.manipulators.values():
            manipulator.remove_hooks()

        self.active_hooks.clear()

    def get_extractor(self, name: str) -> Optional[ActivationExtractor]:
        """Get an extractor by name."""
        return self.extractors.get(name)

    def get_manipulator(self, name: str) -> Optional[NeuronManipulator]:
        """Get a manipulator by name."""
        return self.manipulators.get(name)


def extract_activated_neurons(hooks: Dict[str, torch.Tensor],
                            layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """
    Extract activated neurons from hook results.

    This is a standalone function that mimics the notebook implementation.

    Args:
        hooks: Dictionary of hook results with activation tensors
        layer_indices: List of layer indices to process

    Returns:
        Dictionary mapping layer names to activated neuron indices
    """
    if layer_indices is None:
        layer_indices = list(range(17, 24))  # Default layers 17-23

    result_dict = {}

    for layer_id in layer_indices:
        hook_key = f'decoder.{layer_id}.mlp.hook_post'
        if hook_key in hooks:
            hook_post = hooks[hook_key]
            activated_neuron_indices = (hook_post > 0).nonzero(as_tuple=False)
            result_dict[f'layer_{layer_id}'] = activated_neuron_indices

    return result_dict


def create_batched_inference_hooks(model,
                                 layer_indices: Optional[List[int]] = None) -> Tuple[Dict[str, Any], ActivationExtractor]:
    """
    Create hooks optimized for batched inference.

    Args:
        model: Model to hook
        layer_indices: Layers to extract from

    Returns:
        Tuple of (hook_handles, extractor_instance)
    """
    extractor = ActivationExtractor(layer_indices)
    hooks = extractor.create_extraction_hooks(model)

    return hooks, extractor


def run_inference_with_hooks(model,
                           tokenizer,
                           queries: List[str],
                           hook_manager: Optional[MultiNeuronHookManager] = None,
                           batch_size: int = 256,
                           return_activations: bool = False) -> Dict[str, Any]:
    """
    Run inference on queries with optional hooks.

    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        queries: List of input queries
        hook_manager: Optional hook manager for activation extraction/manipulation
        batch_size: Batch size for inference
        return_activations: Whether to return extracted activations

    Returns:
        Dictionary with inference results and optionally activations
    """
    all_predictions = []
    all_activations = [] if return_activations else None

    # Process queries in batches
    query_batches = batch_list(queries, batch_size)

    for batch_queries in query_batches:
        # Tokenize batch
        input_tokens = tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_tokens['input_ids'].to(get_model_device(model))

        # Create decoder input (typically start token)
        decoder_input = torch.zeros((len(batch_queries), 1), dtype=torch.long, device=get_model_device(model))

        # Run inference
        with torch.no_grad():
            logits, cache = model.run_with_cache(input_ids, decoder_input, remove_batch_dim=False)

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        decoded_predictions = [tokenizer.decode(pred[0]) for pred in predictions]
        all_predictions.extend(decoded_predictions)

        # Extract activations if requested
        if return_activations and hook_manager:
            for extractor in hook_manager.extractors.values():
                batch_activations = {k: v.cpu() for k, v in extractor.activations.items()}
                all_activations.append(batch_activations)
                extractor.clear_activations()

    results = {
        'predictions': all_predictions,
        'total_queries': len(queries)
    }

    if return_activations:
        results['activations'] = all_activations

    return results


def calculate_activation_statistics(activations: List[Dict[str, torch.Tensor]],
                                  layer_indices: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate mean and variance statistics for activations.

    Args:
        activations: List of activation dictionaries from multiple batches
        layer_indices: Layers to calculate statistics for

    Returns:
        Dictionary with mean and variance statistics by layer
    """
    if layer_indices is None:
        layer_indices = list(range(17, 24))

    layer_stats = {}

    for layer_id in layer_indices:
        layer_key = f'layer_{layer_id}'
        layer_activations = []

        # Collect all activations for this layer
        for batch_activations in activations:
            if layer_key in batch_activations:
                layer_activations.append(batch_activations[layer_key])

        if layer_activations:
            # Concatenate all activations
            all_layer_activations = torch.cat(layer_activations, dim=0)

            # Calculate statistics
            mean_activations = torch.mean(all_layer_activations, dim=0)
            var_activations = torch.var(all_layer_activations, dim=0)
            std_activations = torch.std(all_layer_activations, dim=0)

            layer_stats[layer_key] = {
                'mean': mean_activations.numpy(),
                'variance': var_activations.numpy(),
                'std': std_activations.numpy(),
                'shape': list(all_layer_activations.shape),
                'num_samples': all_layer_activations.shape[0]
            }

    return layer_stats