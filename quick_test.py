"""
BERT 模型加载快速测试
=====================
验证 bert_culture_classifier 模型能否在本地环境中正确加载。
"""
import sys, os, time
print("starting...", flush=True)
t0 = time.time()
print(f"t={time.time()-t0:.1f}s importing...", flush=True)
import torch
print(f"t={time.time()-t0:.1f}s torch loaded ({torch.__version__})", flush=True)
from transformers import AutoTokenizer
print(f"t={time.time()-t0:.1f}s transformers loaded", flush=True)

torch.set_num_threads(8)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert_culture_classifier')
print(f"t={time.time()-t0:.1f}s loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(path)
print(f"t={time.time()-t0:.1f}s tokenizer loaded", flush=True)
print("DONE", flush=True)
