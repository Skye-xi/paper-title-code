# Capital Cultural Knowledge Graph — Experiment Code

This repository contains all experiment code for the paper **"Capital Cultural Knowledge Graph Construction Based on LLM Agent Workflow and Tensor Decomposition"** (submitted to IJGI).

## File Overview

| File | Description |
|------|-------------|
| `marginal_frequency_dissolution.py` | Core experiment: Marginal Frequency Dissolution for CP tensor decomposition validation |
| `tensor_stability_analysis.py` | Tensor decomposition stability analysis (rank selection, FMS, regularization sensitivity) |
| `save_qwen_predictions.py` | Save Qwen3 per-sample predictions for McNemar test |
| `baseline_comprehensive.py` | BERT vs Qwen3 comprehensive baseline comparison |
| `bert_evaluate_fast.py` | Optimized BERT vs Qwen3 comparison (subsampled, CPU-friendly) |
| `bert_evaluate_local.py` | Full BERT vs Qwen3 evaluation on 3 tasks |
| `bert_colab_complete.py` | Colab script for training BERT + RoBERTa across tasks |
| `train_bert_colab.py` | Original BERT training script (Colab) |
| `speed_test.py` | BERT inference speed benchmark |
| `quick_test.py` | Quick model loading sanity check |
| `convert_safetensors.py` | Convert safetensors to pytorch_model.bin |
| `generate_response_docx.js` | Generate Response to Reviewers docx |

## Before Running

### 1. Data Files

Place the following CSV files in the same directory as the scripts:

- `张量分解新_GIS标准化数据.csv` — Tensor decomposition input (columns: c_idx, a_idx, s_idx, v_idx)
- `首都文化_情感得分_抽样15%.csv` — 15% sampled Weibo data with culture, aspect, sentiment labels

### 2. Model Weights

- **BERT model**: Train using `train_bert_colab.py` or `bert_colab_complete.py` on Google Colab. Place trained weights in `./bert_culture_classifier/`.
- **Qwen3 API**: Replace `<YOUR_VLLM_API_ENDPOINT>` with your vLLM/Ollama API endpoint.
  - Example for local vLLM: `http://localhost:8000/v1`
  - Example for Ollama: `http://localhost:11434/v1`

### 3. Sensitive Placeholders

The following placeholders must be replaced with actual values:

| Placeholder | Where Used | Description |
|------------|-----------|-------------|
| `<YOUR_VLLM_API_ENDPOINT>` | `save_qwen_predictions.py`, `bert_evaluate_fast.py`, `bert_evaluate_local.py` | vLLM API base URL |
| `<YOUR_VLLM_API_KEY>` | `save_qwen_predictions.py`, `bert_evaluate_fast.py`, `bert_evaluate_local.py` | vLLM API key |
| `<YOUR_PROJECT_DATA_DIR>/张量分解新_GIS标准化数据.csv` | `tensor_stability_analysis.py` | Path to tensor data CSV |

### 4. Environment

```bash
pip install numpy pandas scipy scikit-learn torch transformers aiohttp matplotlib
```

For BERT inference, use a recent PyTorch version (2.0+) with safetensors support for faster loading.

## Citation

If you use this code, please cite our paper (to be updated upon publication).

## License

MIT License
