#!/usr/bin/env python3
"""
Prepare 100-query dataset for GRASP experiments
This script processes existing adversarial results and BEIR datasets
to create the required data files for GRASP pipeline.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_json, save_json, load_beir_datasets


def prepare_100_queries(
    dataset_name: str = 'nq',
    split: str = 'test',
    seed: int = 42,
    output_dir: str = 'data'
):
    """
    Prepare 100 queries with ground truth and adversarial samples
    
    Args:
        dataset_name: BEIR dataset name (nq, msmarco, hotpotqa)
        split: Dataset split
        seed: Random seed
        output_dir: Output directory for data files
    """
    print("=" * 60)
    print("GRASP Dataset Preparation: 100 Queries")
    print("=" * 60)
    
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing adversarial results
    adv_results_path = f'results/adv_targeted_results/{dataset_name}.json'
    
    if not os.path.exists(adv_results_path):
        print(f"Error: Adversarial results not found at {adv_results_path}")
        print("Please ensure the adversarial results are available.")
        return
    
    print(f"\n1. Loading adversarial results from {adv_results_path}")
    all_adv_data = load_json(adv_results_path)
    print(f"   Found {len(all_adv_data)} adversarial samples")
    
    # Load BEIR dataset
    print(f"\n2. Loading BEIR dataset: {dataset_name} ({split})")
    corpus, queries, qrels = load_beir_datasets(dataset_name, split)
    print(f"   Loaded {len(queries)} queries, {len(corpus)} documents")
    
    # Select 100 queries that have both adversarial data and ground truth
    print(f"\n3. Selecting 100 queries with complete data")
    
    valid_queries = []
    for qid, adv_data in all_adv_data.items():
        if qid in queries and qid in qrels and len(qrels[qid]) > 0:
            valid_queries.append((qid, adv_data))
    
    print(f"   Found {len(valid_queries)} valid queries")
    
    # Sample 100 queries
    if len(valid_queries) > 100:
        selected_queries = random.sample(valid_queries, 100)
    else:
        selected_queries = valid_queries[:100]
        print(f"   Warning: Only {len(selected_queries)} queries available")
    
    # Prepare queries_groundtruth.json
    print(f"\n4. Preparing queries_groundtruth.json")
    
    queries_data = {
        "dataset": dataset_name,
        "split": split,
        "num_queries": len(selected_queries),
        "queries": []
    }
    
    for idx, (qid, adv_data) in enumerate(selected_queries):
        query_text = queries[qid]
        
        # Get ground truth documents
        gt_doc_ids = list(qrels[qid].keys())
        ground_truth_texts = [corpus[doc_id]["text"] for doc_id in gt_doc_ids]
        
        query_entry = {
            "id": idx + 1,
            "original_id": qid,
            "query": query_text,
            "ground_truth": ground_truth_texts[0] if ground_truth_texts else "",
            "all_ground_truth_ids": gt_doc_ids,
            "category": dataset_name,
            "correct_answer": adv_data.get("correct answer", ""),
            "incorrect_answer": adv_data.get("incorrect answer", "")
        }
        
        queries_data["queries"].append(query_entry)
    
    # Save queries_groundtruth.json
    queries_output_path = os.path.join(output_dir, 'queries_groundtruth.json')
    save_json(queries_data, queries_output_path)
    print(f"   Saved to {queries_output_path}")
    
    # Prepare adversarial_samples.json
    print(f"\n5. Preparing adversarial_samples.json")
    
    adversarial_data = {
        "dataset": dataset_name,
        "num_samples": len(selected_queries),
        "adv_per_query": 5,
        "adversarial_texts": []
    }
    
    for idx, (qid, adv_data) in enumerate(selected_queries):
        adv_sample = {
            "query_id": idx + 1,
            "original_id": qid,
            "original_query": queries[qid],
            "adversarial_texts": adv_data.get("adv_texts", [])[:5],
            "target_response": adv_data.get("incorrect answer", ""),
            "correct_answer": adv_data.get("correct answer", "")
        }
        
        adversarial_data["adversarial_texts"].append(adv_sample)
    
    # Save adversarial_samples.json
    adv_output_path = os.path.join(output_dir, 'adversarial_samples.json')
    save_json(adversarial_data, adv_output_path)
    print(f"   Saved to {adv_output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Number of queries: {len(selected_queries)}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - {queries_output_path}")
    print(f"  - {adv_output_path}")
    print("\nNext steps:")
    print("  1. Review the generated files")
    print("  2. Run GRASP pipeline:")
    print(f"     python grasp_kaggle.py --queries_file {queries_output_path} --adversarial_file {adv_output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare 100-query dataset for GRASP experiments'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='nq',
        choices=['nq', 'msmarco', 'hotpotqa'],
        help='BEIR dataset name'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split (train, test, dev)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for data files'
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_100_queries(
        dataset_name=args.dataset,
        split=args.split,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
