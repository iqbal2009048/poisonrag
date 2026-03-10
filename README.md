# maliciousdoc
# poisonrag
# poisonrag

## GRASP Implementation

This repository now includes **GRASP (Genetic RAG Attack with Semantic Poisoning)** - a comprehensive framework for evaluating RAG system robustness with HuggingFace LLaMA-3-8B-Instruct (4-bit quantized).

### Quick Start

```bash
# Prepare dataset
python prepare_grasp_data.py --dataset nq --seed 42

# Run GRASP pipeline
python grasp_kaggle.py \
    --dataset nq \
    --num_queries 100 \
    --model_config_path model_configs/llama3_8b_instruct_config.json \
    --results_dir results/grasp_results
```

### Documentation

- **[GRASP_README.md](GRASP_README.md)** - Complete documentation
- **[GRASP_Kaggle_Notebook.ipynb](GRASP_Kaggle_Notebook.ipynb)** - Kaggle notebook template
- **[examples_grasp.py](examples_grasp.py)** - Usage examples
- **[test_grasp.py](test_grasp.py)** - Test suite

### Key Features

- ✅ HuggingFace LLaMA-3-8B-Instruct with 4-bit quantization
- ✅ Genetic Algorithm for adversarial text evolution
- ✅ Comprehensive evaluation (ASR, F1, Precision, Recall)
- ✅ Kaggle-ready with GPU support
- ✅ Batch processing of 100+ queries

### Files Added

- `llm_huggingface.py` - HuggingFace LLaMA integration
- `grasp_kaggle.py` - Main pipeline
- `genetic_algorithm.py` - GA for text evolution
- `evaluation_metrics.py` - Metrics calculation
- `dataset_loader.py` - Data utilities
- `prepare_grasp_data.py` - Dataset preparation
- `requirements_grasp.txt` - Dependencies

See [GRASP_README.md](GRASP_README.md) for detailed instructions.

