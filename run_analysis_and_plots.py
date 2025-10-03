#!/usr/bin/env python3
"""
Comprehensive Document Frequency Sweeping Analysis Script

This script runs the complete analysis pipeline from the Main_TransformerLens_TriviaQA.ipynb notebook,
including multi-document frequency sweeping, visualization generation, and results storage.

Usage:
    python run_analysis_and_plots.py [options]

Examples:
    # Use default settings with popular documents
    python run_analysis_and_plots.py

    # Custom document analysis
    python run_analysis_and_plots.py --target_docs "47788,26530,12345" --min_freq 0.05 --max_freq 0.3 --freq_step 0.05

    # Quick analysis with limited scope
    python run_analysis_and_plots.py --target_docs "47788" --replacement_types "zero_out" --other_queries_sample 1000

    # Visualization-only mode from existing results
    python run_analysis_and_plots.py --visualize_from data/results/neuron_analysis_20231201_143052/complete_analysis_results.json

    # Test set validation of training findings
    python run_analysis_and_plots.py --target_docs "47788" --use_test_set --replacement_types "zero_out"

    # Generate test evaluation data first, then run test validation
    python run_analysis_and_plots.py --generate_test_data --test_queries_path "TriviaQAData/test_queries_trivia_qa.json"

    # Generate popular documents JSON from training data (fast - no model loading)
    python run_analysis_and_plots.py --generate_popular_docs

    # Generate popular documents JSON from test data (top 50)
    python run_analysis_and_plots.py --generate_popular_docs --use_test_set --popular_docs_top_n 50

Environment Requirements:
    - Requires conda environment 236004 for ML dependencies
    - Activate with: source ~/miniconda3/etc/profile.d/conda.sh && conda activate 236004
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Add the local package to path
sys.path.insert(0, str(Path(__file__).parent))

from dsi_analyzer import (
    DSIDocumentAnalyzer,
    create_default_paths,
    get_memory_usage,
    safe_file_write,
    analyze_and_visualize_sweep_results,
    save_incorrect_queries_analysis
)
from dsi_analyzer.test_set_generator import (
    generate_test_evaluation_data,
    check_test_evaluation_data_exists,
    get_test_data_info
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive document frequency sweeping analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Document selection
    parser.add_argument(
        '--target_docs',
        type=str,
        default=None,
        help='Comma-separated document IDs to analyze (e.g., "47788,26530"). If not provided, uses popular documents.'
    )

    parser.add_argument(
        '--num_popular_docs',
        type=int,
        default=5,
        help='Number of popular documents to analyze if --target_docs not provided (default: 5)'
    )

    # Frequency parameters
    parser.add_argument(
        '--min_freq',
        type=float,
        default=0.01,
        help='Minimum frequency threshold (default: 0.01)'
    )

    parser.add_argument(
        '--max_freq',
        type=float,
        default=1.0,
        help='Maximum frequency threshold (default: 1.0)'
    )

    parser.add_argument(
        '--freq_step',
        type=float,
        default=0.1,
        help='Step size between frequency thresholds (default: 0.1)'
    )

    # Analysis parameters
    parser.add_argument(
        '--replacement_types',
        type=str,
        default="zero_out,mean_value",
        help='Comma-separated replacement types to test (default: "zero_out,mean_value")'
    )

    parser.add_argument(
        '--other_queries_sample',
        type=int,
        default=None,
        help='Number of other queries to test for side effects (default: all queries)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for model inference (default: 256)'
    )

    # File paths
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to DSI model (default: auto-detect)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to TriviaQA data (default: auto-detect)'
    )

    parser.add_argument(
        '--stats_path',
        type=str,
        default=None,
        help='Path to statistics files (default: auto-detect)'
    )

    parser.add_argument(
        '--visualize_from',
        type=str,
        default=None,
        help='Path to complete_analysis_results.json file for visualization-only mode'
    )

    parser.add_argument(
        '--use_test_set',
        action='store_true',
        help='Use test set queries instead of training queries for validation'
    )

    parser.add_argument(
        '--test_data_path',
        type=str,
        default=None,
        help='Path to test set data file (default: auto-detect based on stats_path)'
    )

    parser.add_argument(
        '--generate_test_data',
        action='store_true',
        help='Generate test evaluation data from test_queries_trivia_qa.json'
    )

    parser.add_argument(
        '--generate_popular_docs',
        action='store_true',
        help='Generate popular documents JSON ranked by query frequency (fast - no model loading required)'
    )

    parser.add_argument(
        '--popular_docs_top_n',
        type=int,
        default=100,
        help='Number of top documents to include in popular docs JSON (default: 100)'
    )

    parser.add_argument(
        '--test_queries_path',
        type=str,
        default=None,
        help='Path to test_queries_trivia_qa.json file (default: auto-detect in TriviaQAData/)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: data/results/neuron_analysis)'
    )

    # Execution options
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device for computation (auto/cuda/cpu, default: auto)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable detailed logging (default: False)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show parameters and exit without running analysis'
    )

    parser.add_argument(
        '--no_confirm',
        action='store_true',
        help='Skip confirmation prompt and proceed automatically'
    )

    parser.add_argument(
        '--generate_detailed_stats',
        action='store_true',
        help='Generate detailed activation statistics if not available'
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate and process command line arguments."""
    # Validate frequency parameters
    if args.min_freq > args.max_freq:
        raise ValueError(f"min_freq ({args.min_freq}) must be less than max_freq ({args.max_freq})")

    if args.freq_step <= 0:
        raise ValueError(f"freq_step ({args.freq_step}) must be positive")

    if args.min_freq <= 0 or args.max_freq > 1:
        raise ValueError(f"Frequency thresholds must be in range (0, 1]")

    # Process document IDs
    if args.target_docs:
        try:
            doc_ids = [doc.strip() for doc in args.target_docs.split(',')]
            # Try to convert to int, keep as string if it fails
            processed_docs = []
            for doc in doc_ids:
                try:
                    processed_docs.append(int(doc))
                except ValueError:
                    processed_docs.append(doc)
            args.target_docs = processed_docs
        except Exception as e:
            raise ValueError(f"Invalid target_docs format: {e}")

    # Process replacement types
    args.replacement_types = [rt.strip() for rt in args.replacement_types.split(',')]
    valid_types = ['zero_out', 'mean_value']
    for rt in args.replacement_types:
        if rt not in valid_types:
            raise ValueError(f"Invalid replacement type '{rt}'. Valid types: {valid_types}")

    return args


def print_analysis_parameters(args, analyzer, doc_ids, frequency_thresholds):
    """Print comprehensive analysis parameters before execution."""
    print("🔬 DSI Document Frequency Sweeping Analysis")
    print("=" * 60)

    # System information
    memory_info = get_memory_usage()
    print(f"💾 Memory: {memory_info['used_gb']:.1f}GB used, {memory_info['available_gb']:.1f}GB available")
    print(f"🖥️  Device: {args.device}")
    print()

    # Document analysis
    print(f"📊 Document Analysis:")
    print(f"   Target documents: {len(doc_ids)} documents")
    for i, doc_id in enumerate(doc_ids[:10]):  # Show first 10
        print(f"     {i+1}. Document {doc_id}")
    if len(doc_ids) > 10:
        print(f"     ... and {len(doc_ids) - 10} more")
    print()

    # Frequency parameters
    print(f"📈 Frequency Analysis:")
    print(f"   Range: {args.min_freq:.3f} to {args.max_freq:.3f}")
    print(f"   Step size: {args.freq_step:.3f}")
    print(f"   Thresholds: {len(frequency_thresholds)} points")
    print(f"   Values: {[f'{t:.3f}' for t in frequency_thresholds[:5]]}{'...' if len(frequency_thresholds) > 5 else ''}")
    print()

    # Analysis parameters
    print(f"🔧 Analysis Parameters:")
    print(f"   Replacement types: {args.replacement_types}")
    print(f"   Other queries sample: {args.other_queries_sample if args.other_queries_sample else 'all'}")
    print(f"   Batch size: {args.batch_size}")
    print()

    # Estimation
    total_runs = len(doc_ids) * len(frequency_thresholds) * len(args.replacement_types)
    print(f"⏱️  Estimated Analysis:")
    print(f"   Total analysis runs: {total_runs}")
    print(f"   Estimated time: {total_runs * 2:.0f}-{total_runs * 5:.0f} minutes")
    print()

    # Output
    print(f"📁 Output:")
    print(f"   Directory: {args.output_dir}")
    print(f"   Detailed logging: {args.verbose}")
    print()


def create_frequency_thresholds(min_freq: float, max_freq: float, freq_step: float) -> np.ndarray:
    """Create frequency threshold array."""
    # Use numpy linspace for more precise control
    n_points = int((max_freq - min_freq) / freq_step) + 1
    thresholds = np.linspace(min_freq, max_freq, n_points)
    return np.round(thresholds, 6)  # Round to avoid floating point issues


def get_target_documents(analyzer: DSIDocumentAnalyzer,
                        target_docs: Optional[List],
                        num_popular: int) -> List:
    """Get list of target documents for analysis."""
    if target_docs:
        print(f"📋 Using specified documents: {target_docs}")
        return target_docs

    # Get popular documents
    popular_docs = analyzer.get_popular_documents(num_popular * 2)  # Get extra in case some fail
    doc_ids = [doc[0] for doc in popular_docs[:num_popular]]

    print(f"📋 Using top {num_popular} popular documents:")
    for i, (doc_id, count) in enumerate(popular_docs[:num_popular]):
        print(f"   {i+1}. Document {doc_id}: {count} queries")

    return doc_ids


def run_multi_document_sweep(analyzer: DSIDocumentAnalyzer,
                           doc_ids: List,
                           frequency_thresholds: np.ndarray,
                           replacement_types: List[str],
                           other_queries_sample: Optional[int],
                           output_dir: Path,
                           verbose: bool) -> Dict[str, Any]:
    """Run frequency sweeping analysis across multiple documents."""
    print("🔄 Starting multi-document frequency sweep analysis...")

    all_results = {
        'documents': {},
        'frequencies': frequency_thresholds.tolist(),
        'replacement_types': replacement_types,
        'total_queries': len(analyzer.result_dict),
        'analysis_timestamp': datetime.now().isoformat(),
        'parameters': {
            'frequency_range': [float(frequency_thresholds[0]), float(frequency_thresholds[-1])],
            'num_thresholds': len(frequency_thresholds),
            'replacement_types': replacement_types,
            'other_queries_sample': other_queries_sample
        }
    }

    total_runs = len(doc_ids) * len(frequency_thresholds) * len(replacement_types)
    current_run = 0

    for doc_id in doc_ids:
        print(f"\n📖 Analyzing Document {doc_id}")
        print("-" * 40)

        doc_results = {
            'document_id': doc_id,
            'frequencies': {}
        }

        # Get document query count
        try:
            popular_docs = analyzer.get_popular_documents(100)  # Get many to find this one
            doc_query_count = next((count for did, count in popular_docs if str(did) == str(doc_id)), 0)
            doc_results['query_count'] = doc_query_count
            print(f"📊 Document {doc_id} has {doc_query_count} queries")
        except:
            doc_results['query_count'] = 0
            print(f"⚠️  Could not determine query count for document {doc_id}")

        for freq in frequency_thresholds:
            print(f"\n  🎯 Frequency threshold: {freq:.3f}")
            doc_results['frequencies'][float(freq)] = {}

            for replacement_type in replacement_types:
                current_run += 1
                progress = (current_run / total_runs) * 100

                print(f"    🔧 {replacement_type} ({current_run}/{total_runs}, {progress:.1f}%)")

                try:
                    start_time = time.time()

                    # Check if we can use this replacement type
                    if replacement_type == 'mean_value' and analyzer.detailed_activation_stats is None:
                        print(f"    ⚠️  Skipping mean_value - detailed stats not available")
                        doc_results['frequencies'][float(freq)][replacement_type] = {
                            'error': 'detailed_activation_stats not available for mean_value replacement'
                        }
                        continue

                    # Run analysis
                    result = analyzer.analyze_document(
                        target_doc_id=doc_id,
                        frequency_threshold=freq,
                        replacement_type=replacement_type,
                        other_queries_sample=other_queries_sample,
                        verbose=verbose,  # Use the verbose parameter from function
                        use_test_queries=hasattr(analyzer, 'test_result_dict') and analyzer.test_result_dict is not None
                    )

                    elapsed_time = time.time() - start_time

                    # Save incorrect queries analysis if there are any changed queries
                    if result and 'test_results' in result and 'queries_changed' in result['test_results']:
                        target_changed = result['test_results']['queries_changed'].get('target_queries_became_incorrect', [])
                        other_changed = result['test_results']['queries_changed'].get('other_queries_became_incorrect', [])

                        if target_changed or other_changed:
                            # Create filename for incorrect queries
                            incorrect_queries_filename = f"incorrect_queries_{doc_id}_{freq:.3f}_{replacement_type}.json"
                            incorrect_queries_path = output_dir / incorrect_queries_filename

                            # Save the analysis
                            save_incorrect_queries_analysis(
                                target_changed_queries=target_changed,
                                other_changed_queries=other_changed,
                                target_doc_id=doc_id,
                                frequency_threshold=freq,
                                replacement_type=replacement_type,
                                output_path=incorrect_queries_path,
                                verbose=verbose
                            )

                    # Extract key metrics
                    test_results = result['test_results']
                    summary = test_results['summary']

                    # Store simplified results
                    doc_results['frequencies'][float(freq)][replacement_type] = {
                        'target_queries': {
                            'total': test_results['target_results']['total'],
                            'correct_before': test_results['target_results']['correct_before'],
                            'correct_after': test_results['target_results']['correct_after'],
                            'accuracy_before': summary['target_accuracy_before'],
                            'accuracy_after': summary['target_accuracy_after'],
                            'accuracy_drop': summary['target_accuracy_drop']
                        },
                        'other_queries': {
                            'total': test_results['other_results']['total'],
                            'correct_before': test_results['other_results']['correct_before'],
                            'correct_after': test_results['other_results']['correct_after'],
                            'accuracy_before': summary['other_accuracy_before'],
                            'accuracy_after': summary['other_accuracy_after'],
                            'accuracy_drop': summary['other_accuracy_drop']
                        },
                        'overall': {
                            'total': test_results['overall_results']['total'],
                            'correct_before': test_results['overall_results']['correct_before'],
                            'correct_after': test_results['overall_results']['correct_after'],
                            'accuracy_before': summary.get('overall_accuracy_before', 0),
                            'accuracy_after': summary.get('overall_accuracy_after', 0),
                            'accuracy_drop': summary.get('overall_accuracy_drop', 0)
                        },
                        'neurons_filtered': sum(len(neurons) for neurons in result['neuron_data']['filtered_neurons'].values()),
                        'target_relevant_queries': {
                            'total': test_results['target_relevant_results']['total'],
                            'correct_before': test_results['target_relevant_results']['correct_before'],
                            'correct_after': test_results['target_relevant_results']['correct_after'],
                            'accuracy_before': summary['target_relevant_accuracy_before'],
                            'accuracy_after': summary['target_relevant_accuracy_after'],
                            'accuracy_drop': summary['target_relevant_accuracy_drop']
                        },
                        'other_relevant_queries': {
                            'total': test_results['other_relevant_results']['total'],
                            'correct_before': test_results['other_relevant_results']['correct_before'],
                            'correct_after': test_results['other_relevant_results']['correct_after'],
                            'accuracy_before': summary['other_relevant_accuracy_before'],
                            'accuracy_after': summary['other_relevant_accuracy_after'],
                            'accuracy_drop': summary['other_relevant_accuracy_drop']
                        },
                        'queries_changed': {
                            'target_changed': test_results['queries_changed']['n_target_changed'],
                            'other_changed': test_results['queries_changed']['n_other_changed']
                        },
                        'execution_time': elapsed_time
                    }

                    print(f"    ✅ Target: {summary['target_accuracy_drop']:.1f}% drop, "
                          f"Other: {summary['other_accuracy_drop']:.1f}% drop, "
                          f"Time: {elapsed_time:.1f}s")
                    print(f"       Relevant-docs accuracy - Target: {summary['target_relevant_accuracy_drop']:.1f}% drop, "
                          f"Other: {summary['other_relevant_accuracy_drop']:.1f}% drop")

                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    doc_results['frequencies'][float(freq)][replacement_type] = {
                        'error': str(e),
                        'execution_time': 0
                    }

        all_results['documents'][str(doc_id)] = doc_results

    print(f"\n✅ Multi-document sweep analysis completed!")
    print(f"📊 Analyzed {len(doc_ids)} documents across {len(frequency_thresholds)} thresholds")

    # Print detailed results summary
    print(f"\n📋 Results Summary:")
    print(f"{'='*80}")

    for doc_id in doc_ids:
        doc_results = all_results['documents'][str(doc_id)]
        print(f"\n📄 Document {doc_id}:")

        for freq in frequency_thresholds:
            freq_key = float(freq)
            if freq_key in doc_results['frequencies']:
                print(f"\n  🎯 Frequency {freq:.3f}:")

                for replacement_type in doc_results['frequencies'][freq_key]:
                    result = doc_results['frequencies'][freq_key][replacement_type]

                    if 'error' in result:
                        print(f"    {replacement_type}: ❌ {result['error']}")
                        continue

                    # Exact accuracy metrics
                    target_drop = result['target_queries']['accuracy_drop']
                    other_drop = result['other_queries']['accuracy_drop']

                    # Relevant docs accuracy metrics
                    target_rel_drop = result.get('target_relevant_queries', {}).get('accuracy_drop', 0)
                    other_rel_drop = result.get('other_relevant_queries', {}).get('accuracy_drop', 0)

                    print(f"    {replacement_type}:")
                    print(f"      📊 Exact accuracy - Target: {target_drop:.1f}% drop, Other: {other_drop:.1f}% drop")
                    print(f"      📈 Relevant-docs accuracy - Target: {target_rel_drop:.1f}% drop, Other: {other_rel_drop:.1f}% drop")

    print(f"\n{'='*80}")

    return all_results


def handle_test_data_generation(analyzer: DSIDocumentAnalyzer, args) -> bool:
    """
    Handle test data generation if requested or needed.

    Args:
        analyzer: Initialized DSI analyzer with loaded model
        args: Command line arguments

    Returns:
        True if successful, False otherwise
    """
    # Auto-detect test queries path if not provided
    if args.test_queries_path is None:
        possible_paths = [
            Path("TriviaQAData/test_queries_trivia_qa.json"),
            Path("../TriviaQAData/test_queries_trivia_qa.json"),
            analyzer.data_path.parent / "test_queries_trivia_qa.json"
        ]

        for path in possible_paths:
            if path.exists():
                args.test_queries_path = str(path)
                break

        if args.test_queries_path is None:
            print("❌ Could not find test_queries_trivia_qa.json. Please specify --test_queries_path")
            return False

    # Check if test evaluation data already exists
    test_info = get_test_data_info(analyzer.stats_path)

    if test_info['exists'] and not args.generate_test_data:
        print(f"✅ Test evaluation data already exists: {test_info['path']}")
        print(f"📊 Contains {test_info['num_queries']} correctly answered test queries")
        return True

    if args.generate_test_data or not test_info['exists']:
        print("🧪 Generating test evaluation data...")
        print(f"📁 Using test queries from: {args.test_queries_path}")

        try:
            # Generate test evaluation data
            output_path = Path(analyzer.stats_path) / "activated_neurons_test.json"

            test_evaluation_data = generate_test_evaluation_data(
                model=analyzer.model,
                tokenizer=analyzer.tokenizer,
                test_queries_path=args.test_queries_path,
                output_path=output_path,
                batch_size=args.batch_size,  # Use same batch size as analysis
                verbose=True
            )

            print(f"✅ Generated test evaluation data with {len(test_evaluation_data)} queries")
            return True

        except Exception as e:
            print(f"❌ Error generating test evaluation data: {e}")
            return False

    return True


def handle_popular_docs_generation(analyzer: DSIDocumentAnalyzer, args) -> bool:
    """
    Handle popular documents generation if requested.

    Args:
        analyzer: Initialized DSI analyzer with loaded statistics
        args: Command line arguments

    Returns:
        bool: True if successful or not needed, False if failed
    """
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(analyzer.stats_path) if analyzer.stats_path else Path("data")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file names based on test set usage
    if args.use_test_set:
        popular_docs_file = output_dir / "popular_documents_test.json"
        data_type = "test"
        source_file = "activated_neurons_test.json"
    else:
        popular_docs_file = output_dir / "popular_documents.json"
        data_type = "training"
        source_file = "activated_neurons.json"

    # Check if popular docs file already exists and we're not forcing regeneration
    if popular_docs_file.exists() and not args.generate_popular_docs:
        try:
            with open(popular_docs_file, 'r') as f:
                existing_data = json.load(f)
            print(f"✅ Popular documents file already exists: {popular_docs_file}")
            print(f"📊 Contains {len(existing_data.get('popular_documents', []))} documents")
            return True
        except Exception as e:
            print(f"⚠️  Error reading existing popular docs file: {e}")
            print("🔄 Will regenerate...")

    if args.generate_popular_docs or not popular_docs_file.exists():
        print("📊 Generating popular documents JSON...")
        print(f"📁 Data type: {data_type}")
        print(f"🔢 Top N documents: {args.popular_docs_top_n}")

        try:
            # Get popular documents using the analyzer
            popular_docs = analyzer.get_popular_documents(n=args.popular_docs_top_n)

            # Get total query count for percentage calculations
            total_queries = len(analyzer.result_dict) if analyzer.result_dict else 0

            # Create comprehensive JSON structure
            from datetime import datetime
            popular_docs_data = {
                "metadata": {
                    "source_file": source_file,
                    "generated_at": datetime.now().isoformat(),
                    "total_unique_documents": len(set(entry.get("correct_doc_id") for entry in analyzer.result_dict.values() if entry.get("correct_doc_id") is not None)) if analyzer.result_dict else 0,
                    "total_queries": total_queries,
                    "top_n_returned": len(popular_docs),
                    "data_type": data_type,
                    "generation_parameters": {
                        "top_n_requested": args.popular_docs_top_n,
                        "use_test_set": args.use_test_set
                    }
                },
                "popular_documents": [
                    {
                        "rank": i + 1,
                        "document_id": doc_id,
                        "query_count": count,
                        "percentage": round((count / total_queries) * 100, 3) if total_queries > 0 else 0
                    }
                    for i, (doc_id, count) in enumerate(popular_docs)
                ]
            }

            # Save to file
            with open(popular_docs_file, 'w') as f:
                json.dump(popular_docs_data, f, indent=2)

            print(f"✅ Generated popular documents JSON: {popular_docs_file}")
            print(f"📊 Top 5 documents ({data_type} data):")
            for i, (doc_id, count) in enumerate(popular_docs[:5]):
                percentage = (count / total_queries) * 100 if total_queries > 0 else 0
                print(f"  {i+1}. Doc {doc_id}: {count} queries ({percentage:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Error generating popular documents JSON: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def run_visualization_only_mode(args):
    """
    Run visualization-only mode from existing analysis results.

    Args:
        args: Parsed command line arguments with visualize_from path

    Returns:
        bool: True if successful, False otherwise
    """
    import json
    from pathlib import Path
    from dsi_analyzer.visualization import analyze_and_visualize_sweep_results

    print("📊 Running Visualization-Only Mode")
    print("=" * 50)

    # Validate input file
    results_file = Path(args.visualize_from)
    if not results_file.exists():
        print(f"❌ Error: Results file not found: {results_file}")
        return False

    if not results_file.suffix == '.json':
        print(f"❌ Error: Expected JSON file, got: {results_file.suffix}")
        return False

    print(f"📁 Loading results from: {results_file}")

    try:
        # Load the complete analysis results
        with open(results_file, 'r') as f:
            complete_results = json.load(f)

        print(f"✅ Loaded results for {len(complete_results.get('documents', {}))} documents")

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Use same directory as input file, or create new timestamped directory
            input_dir = results_file.parent
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = input_dir / f"visualizations_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        # Create visualizations using the existing function
        print("🎨 Generating visualizations...")
        df, fig = analyze_and_visualize_sweep_results(complete_results, output_dir=output_dir)

        print(f"✅ Visualizations saved to: {output_dir}")
        print(f"📊 Main plot: {output_dir / 'sweep_analysis_visualization.png'}")
        print(f"📋 Data table: {output_dir / 'sweep_results_detailed.csv'}")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    start_time = time.time()

    try:
        # Parse and validate arguments
        args = parse_arguments()
        args = validate_arguments(args)

        # Handle visualization-only mode
        if args.visualize_from:
            return run_visualization_only_mode(args)

        # Create frequency thresholds
        frequency_thresholds = create_frequency_thresholds(args.min_freq, args.max_freq, args.freq_step)

        # Initialize analyzer
        print("🚀 Initializing DSI Document Analyzer...")
        analyzer = DSIDocumentAnalyzer(
            model_path=args.model_path,
            data_path=args.data_path,
            stats_path=args.stats_path,
            device=args.device
        )

        # Check if we only need to generate popular docs (no model loading needed)
        popular_docs_only = (args.generate_popular_docs and
                            not args.target_docs and
                            not args.visualize_from and
                            not args.generate_test_data and
                            not args.use_test_set)

        if popular_docs_only:
            print("📊 Popular docs generation mode - skipping model loading...")
        else:
            # Load model and data for full analysis
            print("🤖 Loading model...")
            analyzer.load_model()

            print("📚 Loading data...")
            analyzer.load_data()

            # Handle test data generation if needed
            if args.use_test_set or args.generate_test_data:
                print("🧪 Handling test data generation...")
                if not handle_test_data_generation(analyzer, args):
                    print("❌ Failed to generate or find test evaluation data")
                    return 1

            # If only generating test data, exit here
            if args.generate_test_data and not args.use_test_set and not args.target_docs and not args.generate_popular_docs:
                print("✅ Test data generation completed. Use --use_test_set to run analysis with test data.")
                return 0

        print("📈 Loading statistics...")
        if args.use_test_set:
            print("🧪 Using test set data for validation")
        try:
            analyzer.load_statistics(
                use_test_set=args.use_test_set,
                test_data_path=args.test_data_path
            )
        except FileNotFoundError as e:
            print(f"⚠️  Statistics not found: {e}")
            if args.generate_detailed_stats:
                print("🔬 Generating statistics...")
                analyzer.generate_statistics(generate_detailed=True)
            else:
                print("❌ Use --generate_detailed_stats to generate missing statistics")
                return 1

        # Handle popular documents generation
        if args.generate_popular_docs:
            print("📊 Handling popular documents generation...")
            if not handle_popular_docs_generation(analyzer, args):
                print("❌ Failed to generate popular documents JSON")
                return 1

        # If only generating popular docs, exit here
        if popular_docs_only:
            print("✅ Popular documents generation completed.")
            return 0

        # Get target documents
        doc_ids = get_target_documents(analyzer, args.target_docs, args.num_popular_docs)

        # Set up output directory
        if args.output_dir is None:
            paths = create_default_paths()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = paths['results_dir'] / f"neuron_analysis_{timestamp}"
        else:
            args.output_dir = Path(args.output_dir)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Print parameters
        print_analysis_parameters(args, analyzer, doc_ids, frequency_thresholds)

        # Dry run check
        if args.dry_run:
            print("🏃 Dry run mode - exiting without analysis")
            return 0

        # Confirm execution
        if not args.no_confirm:
            print("❓ Proceed with analysis? (Press Enter to continue, Ctrl+C to abort)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n❌ Analysis aborted by user")
                return 1
            except EOFError:
                print("\n⚠️  No input detected, proceeding automatically...")
        else:
            print("🚀 Proceeding automatically (--no_confirm flag set)")

        # Filter replacement types based on available data
        if 'mean_value' in args.replacement_types and analyzer.detailed_activation_stats is None:
            print("⚠️  Removing 'mean_value' from replacement types - detailed stats not available")
            args.replacement_types = [rt for rt in args.replacement_types if rt != 'mean_value']

        if not args.replacement_types:
            print("❌ No valid replacement types available")
            return 1

        # Run analysis
        results = run_multi_document_sweep(
            analyzer=analyzer,
            doc_ids=doc_ids,
            frequency_thresholds=frequency_thresholds,
            replacement_types=args.replacement_types,
            other_queries_sample=args.other_queries_sample,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

        # Save results
        print(f"\n💾 Saving results to {args.output_dir}")

        # Save complete results
        results_file = args.output_dir / "complete_analysis_results.json"
        safe_file_write(results_file, results, overwrite=True)
        print(f"📄 Complete results: {results_file}")

        # Generate visualizations
        print(f"\n📈 Generating visualizations...")
        try:
            df, fig = analyze_and_visualize_sweep_results(results, output_dir=args.output_dir)
            print(f"📊 Visualizations saved to {args.output_dir}")
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")

        # Save summary
        total_time = time.time() - start_time

        # Convert args to JSON-serializable format
        args_dict = vars(args).copy()
        for key, value in args_dict.items():
            if hasattr(value, '__fspath__') or isinstance(value, Path):
                args_dict[key] = str(value)

        summary = {
            'analysis_completed': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'documents_analyzed': len(doc_ids),
            'frequency_thresholds': len(frequency_thresholds),
            'replacement_types': args.replacement_types,
            'total_analysis_runs': len(doc_ids) * len(frequency_thresholds) * len(args.replacement_types),
            'output_directory': str(args.output_dir),
            'parameters': args_dict
        }

        summary_file = args.output_dir / "analysis_summary.json"
        safe_file_write(summary_file, summary, overwrite=True)

        print(f"\n🎉 Analysis completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())