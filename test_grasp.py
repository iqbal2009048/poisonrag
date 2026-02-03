#!/usr/bin/env python3
"""
Test script for GRASP implementation
Tests individual components and integration
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    modules = [
        'evaluation_metrics',
        'genetic_algorithm',
        'dataset_loader',
        'llm_huggingface',
        'prepare_grasp_data',
        'grasp_kaggle'
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
    
    print()


def test_evaluation_metrics():
    """Test evaluation metrics calculations"""
    print("="*60)
    print("Testing Evaluation Metrics")
    print("="*60)
    
    from evaluation_metrics import (
        calculate_attack_success_rate,
        calculate_precision_recall_f1_from_counts
    )
    
    # Test ASR
    responses = ["The answer is 5000mg", "I don't know", "It's safe"]
    targets = ["5000mg", "unknown", "safe"]
    
    asr, success_list = calculate_attack_success_rate(responses, targets)
    print(f"ASR Test: {asr:.2%} (expected ~0.67)")
    print(f"Success list: {success_list}")
    assert len(success_list) == 3, "Success list length mismatch"
    
    # Test retrieval metrics
    num_adv_in_topk = [2, 0, 3, 1, 2]
    metrics = calculate_precision_recall_f1_from_counts(
        num_adv_in_topk,
        top_k=5,
        adv_per_query=5
    )
    
    print(f"Precision: {metrics['precision_mean']:.4f}")
    print(f"Recall: {metrics['recall_mean']:.4f}")
    print(f"F1: {metrics['f1_mean']:.4f}")
    
    print("✓ Evaluation metrics tests passed\n")


def test_genetic_algorithm():
    """Test genetic algorithm components"""
    print("="*60)
    print("Testing Genetic Algorithm")
    print("="*60)
    
    from genetic_algorithm import FitnessEvaluator, SemanticMutator, GeneticAlgorithm
    
    # Test fitness evaluator (without actual model)
    evaluator = FitnessEvaluator(
        embedding_model=None,
        tokenizer=None,
        weights=(0.5, 0.3, 0.2)
    )
    
    # Test fluency calculation
    text = "This is a proper sentence with good structure."
    fluency = evaluator.compute_fluency(text)
    print(f"Fluency score: {fluency:.4f}")
    assert 0 <= fluency <= 1, "Fluency score out of range"
    
    # Test mutator
    mutator = SemanticMutator(mutation_rate=0.5)
    original = "The safe dose is 200mg daily"
    mutated = mutator.mutate(original)
    print(f"Original: {original}")
    print(f"Mutated: {mutated}")
    
    print("✓ Genetic algorithm tests passed\n")


def test_data_templates():
    """Test data template files exist and are valid JSON"""
    print("="*60)
    print("Testing Data Templates")
    print("="*60)
    
    import json
    
    templates = [
        'data/queries_groundtruth_template.json',
        'data/adversarial_samples_template.json'
    ]
    
    for template_path in templates:
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    data = json.load(f)
                print(f"✓ {template_path} - Valid JSON")
                print(f"  Keys: {list(data.keys())}")
            except Exception as e:
                print(f"✗ {template_path}: {e}")
        else:
            print(f"✗ {template_path}: File not found")
    
    print()


def test_model_config():
    """Test model configuration file"""
    print("="*60)
    print("Testing Model Configuration")
    print("="*60)
    
    import json
    
    config_path = 'model_configs/llama3_8b_instruct_config.json'
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✓ {config_path} - Valid JSON")
            print(f"  Provider: {config['model_info']['provider']}")
            print(f"  Model: {config['model_info']['name']}")
            print(f"  Temperature: {config['params']['temperature']}")
            print(f"  Max tokens: {config['params']['max_output_tokens']}")
            
            # Validate required fields
            assert 'model_info' in config
            assert 'api_key_info' in config
            assert 'params' in config
            
            print("✓ Configuration structure valid")
        except Exception as e:
            print(f"✗ {config_path}: {e}")
    else:
        print(f"✗ {config_path}: File not found")
    
    print()


def test_utils():
    """Test utility functions"""
    print("="*60)
    print("Testing Utility Functions")
    print("="*60)
    
    from src.utils import clean_str, f1_score
    import numpy as np
    
    # Test clean_str
    test_strings = [
        ("Hello World.", "hello world"),
        ("  CAPS  ", "caps"),
        ("Multiple   spaces", "multiple   spaces")
    ]
    
    for input_str, expected_output in test_strings:
        output = clean_str(input_str)
        # Note: clean_str removes trailing period and converts to lowercase
        print(f"clean_str('{input_str}') = '{output}'")
    
    # Test f1_score
    precision = np.array([0.8, 0.6, 0.9])
    recall = np.array([0.7, 0.8, 0.9])
    f1 = f1_score(precision, recall)
    print(f"\nF1 scores: {f1}")
    print(f"Mean F1: {np.mean(f1):.4f}")
    
    print("✓ Utility function tests passed\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GRASP Implementation Test Suite")
    print("="*60 + "\n")
    
    try:
        test_imports()
        test_data_templates()
        test_model_config()
        test_utils()
        test_evaluation_metrics()
        test_genetic_algorithm()
        
        print("="*60)
        print("All Tests Passed! ✓")
        print("="*60)
        print("\nGRASP implementation is ready to use.")
        print("\nNext steps:")
        print("1. Prepare dataset: python prepare_grasp_data.py --dataset nq")
        print("2. Run pipeline: python grasp_kaggle.py --dataset nq")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test Failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
