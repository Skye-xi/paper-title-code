Fine-Grained Cultural Perception and Evaluation of Beijing’s Capital Culture Integrating Large Language Models with Higher-Order Tensor Decomposition

Shihao Xi¹,², Zhiyuan Ou¹,², Bin Meng²,³,*, Xiaohang Li¹,²

¹ College of Applied Arts and Sciences, Beijing Union University, Beijing 100191, China
² Urban Cultural Perception and Computing Laboratory, Beijing Union University, Beijing 100191, China
³ Institute of Beijing Studies, Beijing Union University, Beijing 100191, China

License: MIT License

Abstract

Fine-grained urban cultural perception is critical for GIScience, yet traditional social media studies struggle with complex cultural semantics and heterogeneous factor integration. Addressing Beijing's "Capital Culture," this study couples LLM agents with higher-order tensor decomposition. Using 2019 full-sample geotagged Sina Weibo data, we developed a four-agent collaborative architecture with Chain-of-Thought prompting and human-in-the-loop mechanisms via a locally deployed Qwen3-32B model. A four-way tensor ("Cultural Type–Evaluation Aspect–Sentiment Polarity–Spatial Carrier") was constructed and integrated with kernel density estimation to characterize spatial differentiation. The agentic workflow achieves over 90% accuracy in cultural and sentiment classification, and the tensor decomposition attains a 94.87% goodness-of-fit, successfully identifying latent patterns. Spatially, Beijing's capital culture exhibits an unbalanced hierarchical structure—"high coupling in the core area with differentiated expansion at the periphery."

Keywords: cultural sensing; large language models; agentic workflow; aspect-based sentiment analysis; tensor decomposition; cultural heritage

Repository Structure

Core Pipeline: Four-Agent System

agent1_culture_classification.py : Agent 1. Binary classification of Weibo posts into 4 capital culture types (Ancient Capital, Red, Beijing-Flavor, Innovation) using LLM with max_tokens=1.

agent3_culture_carrier_extraction.py : Agent 3.0. Extract material culture carriers from culture-labeled posts.

agent2_absa.py : Agent 2.1. Aspect-level sentiment analysis (ABSA) — extracts (carrier, aspect, evaluation, sentiment) quadruples.

sentiment_scoring.py : Agent 2.2. Compute 17-dimension weighted sentiment scores from ABSA results using entropy weight method.

agent3_1_place_detection.py : Agent 3.1. Classify carriers as geographic places (1) or non-places (0).

agent3_2_spatial_alignment.py : Agent 3.2. Toponym normalization and alignment (hybrid: high-frequency mapping + LLM fallback).

agent4_self_reflection.py : Agent 4. Five structural consistency checks (A–E) with closed-loop correction (max_retry=2, HITL sampling every 5 batches).

agent4_decision_tree.py : Standalone decision-tree implementation of Agent 4's five checks (for paper algorithm reference).

orchestrator.py : Pipeline pseudocode illustrating the complete four-agent workflow (for methodology section).

Tensor Decomposition Experiments

marginal_frequency_dissolution.py : Core experiment: Marginal Frequency Dissolution for CP tensor decomposition validation (Section 4.3).

tensor_stability_analysis.py : Tensor decomposition stability analysis (rank selection, FMS, regularization sensitivity).

cramers_v_calculation.py : Cramér's V effect size calculation for Sections 3.1.2 and 3.2.1 (with bootstrap 95% CI).

Baseline Comparison (BERT / RoBERTa vs Qwen3)

bert_baseline_comparison.py : Full baseline comparison: fine-tune BERT-base-Chinese and RoBERTa-wwm-ext vs Qwen3-32B zero-shot on 3 tasks (culture type / aspect / polarity).

bert_colab_complete.py : Colab script for training BERT + RoBERTa across all tasks.

train_bert_colab.py : Original BERT training script (Google Colab).

bert_evaluate_fast.py : Optimized BERT vs Qwen3 comparison (subsampled, CPU-friendly).

bert_evaluate_local.py : Full BERT vs Qwen3 evaluation on 3 tasks.

baseline_comprehensive.py : Comprehensive BERT vs Qwen3 baseline comparison.

save_qwen_predictions.py : Save Qwen3 per-sample predictions for McNemar statistical test.

speed_test.py : BERT inference speed benchmark.

quick_test.py : Quick model loading sanity check.

Utilities

convert_safetensors.py : Convert safetensors format to pytorch_model.bin.

generate_response_docx.js : Generate Response to Reviewers document (.docx).

Before Running

Data Files

Place the following CSV files in the same directory as the scripts:

张量分解新_GIS标准化数据.csv : Tensor decomposition input (columns: c_idx, a_idx, s_idx, v_idx).

首都文化_情感得分_抽样15%.csv : 15% sampled Weibo data with culture, aspect, and sentiment labels.

Model Weights and APIs

BERT / RoBERTa: Train using train_bert_colab.py or bert_colab_complete.py on Google Colab. Place the trained weights in ./bert_culture_classifier/.

Qwen3 API: Replace the <YOUR_VLLM_SERVER_IP> and <YOUR_API_KEY> placeholders in all scripts with your actual vLLM/Ollama endpoint. For example, local vLLM: http://localhost:8000/v1 ; Ollama: http://localhost:11434/v1.

Database Configuration (for the Agent Pipeline)

The four-agent system requires a PostgreSQL database. Update the DB_CONFIG dictionary in each script with your own credentials:

DB_CONFIG = {
"host": "<YOUR_DB_HOST>",
"port": 5432,
"user": "<YOUR_DB_USER>",
"password": "<YOUR_DB_PASSWORD>",
"database": "<YOUR_DB_NAME>",
}

Prompt Templates

Agents 1–4 rely on external prompt files (referenced via <YOUR_PROMPT_DIR>). Place your prompt .txt files in the specified directory and update the path in each script. The complete prompt templates are provided in Appendix A of the paper.

Environment Setup

Install the required Python packages:

pip install numpy pandas scipy scikit-learn torch transformers aiohttp matplotlib asyncpg psycopg2-binary tqdm openai rich

For BERT inference, PyTorch 2.0+ with safetensors support is recommended for faster loading.

Running the Experiments

Option A: Run the Full Four-Agent Pipeline

Execute each agent script in order (refer to orchestrator.py for the correct sequence):

python agent1_culture_classification.py
python agent3_culture_carrier_extraction.py
python agent2_absa.py
python sentiment_scoring.py
python agent3_1_place_detection.py
python agent3_2_spatial_alignment.py
python agent4_self_reflection.py

Option B: Run Baseline Comparison

python bert_baseline_comparison.py

Option C: Run Tensor Decomposition Experiments

python marginal_frequency_dissolution.py
python tensor_stability_analysis.py
python cramers_v_calculation.py

Outputs

bj2019_culture_agent4_verified : Final verified table after Agent 4 self-reflection (PostgreSQL).

bj2019_agent4_hitl_samples : HITL sampling table for human annotation (PostgreSQL).

baseline_results/baseline_comparison.csv : BERT/RoBERTa vs Qwen3 performance metrics.

cramers_v_results.json : Cramér's V effect sizes with bootstrap 95% CI.

Key Findings

Based on approximately 11.76 million geotagged Sina Weibo posts from Beijing (2019), the study yielded the following key results:

Cultural perception hierarchy: Ancient Capital Culture (53.60%) and Beijing-Flavor Culture (28.56%) dominate public perception, while Red Culture (12.38%) and Innovative Culture (5.45%) remain relatively marginalized.

Classification accuracy: The agentic workflow achieved over 90% accuracy across all four cultural types.

Cross-model consistency: Agreement rate with DeepSeek-V3 on joint (aspect, polarity) prediction reached 86.20% (Cohen's kappa = 0.6827).

Tensor decomposition: CP decomposition at rank R = 9 achieved a 94.87% goodness-of-fit, identifying nine latent cultural perception patterns with clear semantic interpretation and spatial directionality.

Spatial pattern: Beijing's capital culture exhibits an unbalanced hierarchical structure—"high coupling in the core area with differentiated expansion at the periphery."

Citation

If you use this code in your research, please cite our paper:

@article{xi2026finegrained,
title={Fine-Grained Cultural Perception and Evaluation of Beijing's Capital Culture Integrating Large Language Models with Higher-Order Tensor Decomposition},
author={Xi, Shihao and Ou, Zhiyuan and Meng, Bin and Li, Xiaohang},
journal={},
year={2026},
note={Submitted}
}

License

This project is licensed under the MIT License – see the LICENSE file for details.

Acknowledgments

This research was funded by the National Natural Science Foundation of China (General Program), grant number 42471272.

Qwen3-32B for LLM-powered classification and ABSA.

DeepSeek-V3 for cross-model validation.

Hugging Face Transformers for BERT/RoBERTa fine-tuning.

PostgreSQL for data storage and pipeline orchestration.

Contact

Correspondence: Bin Meng — mengbin@buu.edu.cn

Repository: https://github.com/Skye-xi/paper-title-code
