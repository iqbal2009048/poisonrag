# PoisonedRAG

A framework for studying adversarial attacks on Retrieval-Augmented Generation (RAG) systems.

## 🚀 Kaggle Quick Start (Llama3 8B Instruct)

The easiest way to run PoisonedRAG is using our Kaggle-ready notebook with Llama3 8B Instruct.

### Option 1: Jupyter Notebook
Upload `kaggle_poisonedrag_llama3_8b.ipynb` to Kaggle and run all cells.

### Option 2: Python Script
```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Install dependencies
pip install transformers accelerate torch beir sentence-transformers huggingface_hub

# Run the attack
python kaggle_run_llama3.py --dataset nq --repeat_times 2 --M 5
```

### Requirements for Kaggle
1. **GPU**: Enable GPU T4 x2 or P100 in Kaggle settings
2. **Hugging Face Token**: Add as Kaggle secret named `HF_TOKEN`
3. **Model Access**: Request access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### Available Datasets
- `nq` - Natural Questions
- `hotpotqa` - HotpotQA  
- `msmarco` - MS MARCO

## 📁 Project Structure

```
poisonrag/
├── kaggle_poisonedrag_llama3_8b.ipynb  # Kaggle notebook
├── kaggle_run_llama3.py                 # Standalone Python script
├── main.py                              # Main experiment script
├── run.py                               # Batch runner
├── gen_adv.py                           # Adversarial text generation
├── evaluate_beir.py                     # BEIR evaluation
├── model_configs/                       # LLM configurations
│   ├── llama3_8b_instruct_config.json   # Llama3 8B config
│   ├── llama7b_config.json
│   └── ...
├── src/
│   ├── attack.py                        # Attack implementation
│   ├── prompts.py                       # Prompt templates
│   ├── utils.py                         # Utilities
│   └── models/                          # LLM wrappers
└── results/                             # Pre-computed results
    ├── adv_targeted_results/            # Adversarial texts
    └── beir_results/                    # BEIR retrieval results
```

## 🔧 Advanced Usage

### Using Original Main Script
```bash
python main.py \
    --eval_model_code contriever \
    --eval_dataset nq \
    --model_name llama3_8b_instruct \
    --top_k 5 \
    --attack_method LM_targeted \
    --adv_per_query 5 \
    --repeat_times 10 \
    --M 10
```

### Configuration Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset (nq, hotpotqa, msmarco) | nq |
| `--top_k` | Number of retrieved documents | 5 |
| `--adv_per_query` | Adversarial texts per query | 5 |
| `--attack_method` | Attack method | LM_targeted |
| `--repeat_times` | Number of iterations | 10 |
| `--M` | Queries per iteration | 10 |

## 📊 Metrics

- **ASR (Attack Success Rate)**: Percentage of queries where the LLM outputs the incorrect answer
- **Precision**: Ratio of injected adversarial texts in top-K
- **Recall**: Ratio of adversarial texts successfully retrieved
- **F1 Score**: Harmonic mean of precision and recall

## 📜 License

See [LICENSE](LICENSE) for details.
