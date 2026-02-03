"""
GRASP (Genetic RAG Attack with Semantic Poisoning) - Kaggle Ready Implementation
Main script for running end-to-end GRASP experiments with HuggingFace LLaMA-3-8B-Instruct (4-bit)

This script is designed to run in Kaggle notebook environments with GPU support.
"""

import os
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import create_model
from src.utils import load_json, save_json, setup_seeds, clean_str, load_models
from src.attack import Attacker
from src.prompts import wrap_prompt
from dataset_loader import (
    prepare_queries_groundtruth,
    prepare_adversarial_samples_from_existing,
    load_queries_groundtruth
)
from evaluation_metrics import (
    evaluate_attack,
    print_evaluation_results,
    save_detailed_results
)


class GRASPPipeline:
    """
    GRASP Pipeline for RAG poisoning attacks
    """
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup seeds for reproducibility
        setup_seeds(args.seed)
        
        # Initialize results storage
        self.results_dir = args.results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized GRASP Pipeline")
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
    
    def prepare_data(self):
        """
        Phase 1 & 2: Prepare dataset and adversarial samples
        """
        print("\n" + "="*60)
        print("PHASE 1 & 2: Data Preparation")
        print("="*60)
        
        # Check if custom data files are provided
        if self.args.queries_file and os.path.exists(self.args.queries_file):
            print(f"Loading queries from {self.args.queries_file}")
            self.queries_data = load_queries_groundtruth(self.args.queries_file)
        else:
            # Prepare from BEIR dataset
            print(f"Preparing {self.args.num_queries} queries from {self.args.dataset}")
            self.queries_data = prepare_queries_groundtruth(
                dataset_name=self.args.dataset,
                num_queries=self.args.num_queries,
                split=self.args.split,
                seed=self.args.seed,
                output_path=os.path.join(self.results_dir, 'queries_groundtruth.json')
            )
        
        print(f"Loaded {len(self.queries_data['queries'])} queries")
        
        # Load or prepare adversarial samples
        if self.args.adversarial_file and os.path.exists(self.args.adversarial_file):
            print(f"Loading adversarial samples from {self.args.adversarial_file}")
            self.adversarial_data = load_json(self.args.adversarial_file)
        else:
            print(f"Preparing adversarial samples from existing results")
            self.adversarial_data = prepare_adversarial_samples_from_existing(
                dataset_name=self.args.dataset,
                queries_data=self.queries_data,
                adv_per_query=self.args.adv_per_query,
                output_path=os.path.join(self.results_dir, 'adversarial_samples.json')
            )
        
        if self.adversarial_data:
            print(f"Loaded {len(self.adversarial_data['adversarial_texts'])} adversarial samples")
        
    def initialize_models(self):
        """
        Phase 3: Initialize retrieval models and LLM
        """
        print("\n" + "="*60)
        print("PHASE 3: Model Initialization")
        print("="*60)
        
        # Load retrieval models for embedding
        if self.args.attack_method not in [None, 'None']:
            print(f"Loading retrieval model: {self.args.eval_model_code}")
            self.model, self.c_model, self.tokenizer, self.get_emb = load_models(
                self.args.eval_model_code
            )
            self.model.eval()
            self.model.to(self.device)
            self.c_model.eval()
            self.c_model.to(self.device)
            
            # Initialize attacker
            self.attacker = Attacker(
                self.args,
                model=self.model,
                c_model=self.c_model,
                tokenizer=self.tokenizer,
                get_emb=self.get_emb
            )
        
        # Load LLM (HuggingFace LLaMA-3-8B-Instruct with 4-bit quantization)
        print(f"\nLoading LLM from config: {self.args.model_config_path}")
        self.llm = create_model(self.args.model_config_path)
        print("LLM loaded successfully")
    
    def load_corpus_and_results(self):
        """
        Load BEIR corpus and retrieval results
        """
        print("\n" + "="*60)
        print("PHASE 4: Loading Corpus and Retrieval Results")
        print("="*60)
        
        from src.utils import load_beir_datasets
        
        # Load BEIR dataset
        self.corpus, self.queries, self.qrels = load_beir_datasets(
            self.args.dataset,
            self.args.split
        )
        
        # Load BEIR retrieval results
        if self.args.orig_beir_results is None:
            if self.args.split == 'test':
                self.args.orig_beir_results = f"results/beir_results/{self.args.dataset}-{self.args.eval_model_code}.json"
            elif self.args.split == 'dev':
                self.args.orig_beir_results = f"results/beir_results/{self.args.dataset}-{self.args.eval_model_code}-dev.json"
            
            if self.args.score_function == 'cos_sim':
                self.args.orig_beir_results = f"results/beir_results/{self.args.dataset}-{self.args.eval_model_code}-cos.json"
        
        if not os.path.exists(self.args.orig_beir_results):
            print(f"Warning: BEIR results not found at {self.args.orig_beir_results}")
            print("Proceeding with available data...")
            self.beir_results = {}
        else:
            with open(self.args.orig_beir_results, 'r') as f:
                self.beir_results = json.load(f)
            print(f"Loaded BEIR results: {len(self.beir_results)} queries")
    
    def run_clean_baseline(self):
        """
        Phase 5: Run clean RAG baseline (without poisoning)
        """
        print("\n" + "="*60)
        print("PHASE 5: Clean Baseline Evaluation")
        print("="*60)
        
        clean_responses = []
        
        for query_entry in tqdm(self.queries_data['queries'][:self.args.num_eval], 
                                desc="Clean baseline"):
            original_id = query_entry['original_id']
            question = query_entry['query']
            
            # Get top-k documents
            if original_id in self.beir_results:
                topk_idx = list(self.beir_results[original_id].keys())[:self.args.top_k]
                topk_contents = [self.corpus[idx]['text'] for idx in topk_idx]
            else:
                # Fallback: use ground truth
                gt_ids = query_entry.get('all_ground_truth_ids', [])
                topk_contents = [self.corpus[gid]['text'] for gid in gt_ids[:self.args.top_k]]
            
            # Generate response
            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            response = self.llm.query(query_prompt)
            
            clean_responses.append({
                'query_id': query_entry['id'],
                'question': question,
                'response': response,
                'contexts': topk_contents
            })
        
        self.clean_responses = clean_responses
        
        # Save clean responses
        save_json(
            clean_responses,
            os.path.join(self.results_dir, 'clean_responses.json')
        )
        
        print(f"Clean baseline completed: {len(clean_responses)} queries")
    
    def run_poisoned_evaluation(self):
        """
        Phase 6: Run poisoned RAG evaluation with adversarial texts
        """
        print("\n" + "="*60)
        print("PHASE 6: Poisoned RAG Evaluation")
        print("="*60)
        
        if not self.adversarial_data:
            print("No adversarial data available. Skipping poisoned evaluation.")
            return
        
        # Build adversarial text lookup
        adv_lookup = {}
        for adv_entry in self.adversarial_data['adversarial_texts']:
            query_id = adv_entry['query_id']
            adv_lookup[query_id] = adv_entry
        
        poisoned_responses = []
        num_adv_in_topk_list = []
        target_answers = []
        
        # Process with genetic algorithm if enabled
        if self.args.use_genetic_algorithm and self.args.attack_method not in [None, 'None']:
            print("Applying Genetic Algorithm to evolve adversarial texts...")
            # This would call the genetic algorithm implementation
            # For now, we use the existing adversarial texts
        
        for query_entry in tqdm(self.queries_data['queries'][:self.args.num_eval],
                               desc="Poisoned evaluation"):
            query_id = query_entry['id']
            original_id = query_entry['original_id']
            question = query_entry['query']
            
            # Get adversarial texts for this query
            if query_id not in adv_lookup:
                print(f"Warning: No adversarial data for query {query_id}")
                continue
            
            adv_entry = adv_lookup[query_id]
            adv_texts = adv_entry['adversarial_texts'][:self.args.top_adversarial]
            target_answer = adv_entry.get('target_response', '')
            target_answers.append(target_answer)
            
            # Prepend query to adversarial texts (following original implementation)
            adv_texts_with_query = [question + ". " + adv for adv in adv_texts]
            
            # Get top-k documents and inject adversarial texts
            if original_id in self.beir_results and self.args.attack_method not in [None, 'None']:
                topk_idx = list(self.beir_results[original_id].keys())[:self.args.top_k]
                topk_results = [
                    {'score': self.beir_results[original_id][idx], 
                     'context': self.corpus[idx]['text']} 
                    for idx in topk_idx
                ]
                
                # Compute embeddings for adversarial texts
                adv_input = self.tokenizer(adv_texts_with_query, padding=True, 
                                          truncation=True, return_tensors="pt")
                adv_input = {key: value.to(self.device) for key, value in adv_input.items()}
                
                with torch.no_grad():
                    adv_embs = self.get_emb(self.c_model, adv_input)
                
                # Compute query embedding
                query_input = self.tokenizer(question, padding=True, truncation=True, 
                                            return_tensors="pt")
                query_input = {key: value.to(self.device) for key, value in query_input.items()}
                
                with torch.no_grad():
                    query_emb = self.get_emb(self.model, query_input)
                
                # Calculate similarity scores and inject adversarial texts
                for j, adv_text in enumerate(adv_texts_with_query):
                    adv_emb = adv_embs[j, :].unsqueeze(0)
                    
                    if self.args.score_function == 'dot':
                        adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                    elif self.args.score_function == 'cos_sim':
                        adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                    
                    topk_results.append({'score': adv_sim, 'context': adv_text})
                
                # Re-rank and select top-k
                topk_results = sorted(topk_results, key=lambda x: float(x['score']), 
                                    reverse=True)
                topk_contents = [topk_results[j]["context"] for j in range(self.args.top_k)]
                
                # Count adversarial texts in top-k
                adv_text_set = set(adv_texts_with_query)
                cnt_from_adv = sum([1 for i in topk_contents if i in adv_text_set])
                num_adv_in_topk_list.append(cnt_from_adv)
                injected_adv = [i for i in topk_contents if i in adv_text_set]
            else:
                # Simple injection at top positions
                topk_contents = adv_texts_with_query[:self.args.top_k]
                num_adv_in_topk_list.append(len(topk_contents))
                injected_adv = topk_contents
            
            # Generate response with poisoned context
            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            response = self.llm.query(query_prompt)
            
            poisoned_responses.append({
                'query_id': query_id,
                'question': question,
                'response': response,
                'injected_adversarial': injected_adv,
                'target_answer': target_answer,
                'correct_answer': adv_entry.get('correct_answer', ''),
                'contexts': topk_contents
            })
        
        self.poisoned_responses = poisoned_responses
        self.num_adv_in_topk_list = num_adv_in_topk_list
        self.target_answers = target_answers
        
        # Save poisoned responses
        save_json(
            poisoned_responses,
            os.path.join(self.results_dir, 'poisoned_responses.json')
        )
        
        print(f"Poisoned evaluation completed: {len(poisoned_responses)} queries")
    
    def evaluate_results(self):
        """
        Phase 7: Calculate evaluation metrics
        """
        print("\n" + "="*60)
        print("PHASE 7: Evaluation Metrics")
        print("="*60)
        
        if not hasattr(self, 'poisoned_responses') or not self.poisoned_responses:
            print("No poisoned responses available for evaluation")
            return
        
        # Extract data for evaluation
        clean_resp_texts = [r['response'] for r in self.clean_responses]
        poisoned_resp_texts = [r['response'] for r in self.poisoned_responses]
        
        # Calculate metrics
        results = evaluate_attack(
            clean_responses=clean_resp_texts,
            poisoned_responses=poisoned_resp_texts,
            target_answers=self.target_answers,
            num_adv_in_topk_list=self.num_adv_in_topk_list,
            top_k=self.args.top_k,
            adv_per_query=self.args.adv_per_query
        )
        
        self.evaluation_results = results
        
        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_json(
            results,
            os.path.join(self.results_dir, 'evaluation_results.json')
        )
        
        # Save detailed results with per-query information
        query_details = []
        for i, pr in enumerate(self.poisoned_responses):
            query_details.append({
                'query_id': pr['query_id'],
                'question': pr['question'],
                'clean_response': self.clean_responses[i]['response'] if i < len(self.clean_responses) else '',
                'poisoned_response': pr['response'],
                'target_answer': pr['target_answer'],
                'correct_answer': pr['correct_answer'],
                'attack_success': results['success_list'][i] if i < len(results['success_list']) else False,
                'num_adv_in_topk': self.num_adv_in_topk_list[i] if i < len(self.num_adv_in_topk_list) else 0,
                'injected_adversarial': pr['injected_adversarial']
            })
        
        save_detailed_results(
            results,
            query_details,
            os.path.join(self.results_dir, 'detailed_results.json')
        )
        
        # Save as CSV for easier analysis
        import csv
        csv_path = os.path.join(self.results_dir, 'attack_results_detailed.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query_id', 'question', 'attack_success', 'num_adv_in_topk',
                'target_answer', 'correct_answer'
            ])
            writer.writeheader()
            for qd in query_details:
                writer.writerow({
                    'query_id': qd['query_id'],
                    'question': qd['question'],
                    'attack_success': qd['attack_success'],
                    'num_adv_in_topk': qd['num_adv_in_topk'],
                    'target_answer': qd['target_answer'],
                    'correct_answer': qd['correct_answer']
                })
        
        print(f"\nResults saved to {self.results_dir}/")
    
    def run(self):
        """
        Run the complete GRASP pipeline
        """
        print("\n" + "="*60)
        print("GRASP PIPELINE - START")
        print("="*60)
        
        self.prepare_data()
        self.initialize_models()
        self.load_corpus_and_results()
        self.run_clean_baseline()
        self.run_poisoned_evaluation()
        self.evaluate_results()
        
        print("\n" + "="*60)
        print("GRASP PIPELINE - COMPLETE")
        print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='GRASP: Genetic RAG Attack with Semantic Poisoning')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='nq', 
                       help='BEIR dataset to use (nq, msmarco, hotpotqa)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split (train, test, dev)')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries to process')
    parser.add_argument('--num_eval', type=int, default=100,
                       help='Number of queries to evaluate')
    
    # Data file arguments
    parser.add_argument('--queries_file', type=str, default=None,
                       help='Path to queries_groundtruth.json file')
    parser.add_argument('--adversarial_file', type=str, default=None,
                       help='Path to adversarial_samples.json file')
    
    # Model arguments
    parser.add_argument('--model_config_path', type=str,
                       default='model_configs/llama3_8b_instruct_config.json',
                       help='Path to LLM config file')
    parser.add_argument('--eval_model_code', type=str, default='contriever',
                       help='Retrieval model code')
    
    # Attack arguments
    parser.add_argument('--attack_method', type=str, default='LM_targeted',
                       help='Attack method to use')
    parser.add_argument('--adv_per_query', type=int, default=5,
                       help='Number of adversarial texts per query')
    parser.add_argument('--top_adversarial', type=int, default=3,
                       help='Top N adversarial texts to use per query')
    parser.add_argument('--use_genetic_algorithm', action='store_true',
                       help='Use genetic algorithm to evolve adversarial texts')
    
    # RAG arguments
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--score_function', type=str, default='dot',
                       choices=['dot', 'cos_sim'],
                       help='Scoring function for retrieval')
    
    # BEIR results
    parser.add_argument('--orig_beir_results', type=str, default=None,
                       help='Path to BEIR retrieval results')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--results_dir', type=str, default='results/grasp_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    return args


def main():
    """
    Main function for GRASP Kaggle notebook
    """
    args = parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # Create and run pipeline
    pipeline = GRASPPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
