"""
模型格式转换工具
================
将 safetensors 格式转换为 pytorch_model.bin，加速 conda 环境加载。
"""
import torch, os, sys
from safetensors.torch import load_file

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert_culture_classifier')
st_path = os.path.join(model_dir, 'model.safetensors')
bin_path = os.path.join(model_dir, 'pytorch_model.bin')

print(f"Loading safetensors: {st_path}")
print(f"File size: {os.path.getsize(st_path)/1024/1024:.0f} MB")
t0 = __import__('time').time()
state_dict = load_file(st_path, device='cpu')
print(f"Loaded in {__import__('time').time()-t0:.1f}s")

print(f"Saving pytorch_model.bin...")
torch.save(state_dict, bin_path)
print(f"Saved: {os.path.getsize(bin_path)/1024/1024:.0f} MB")
print("Done! Model can now be loaded with from_pretrained() using .bin format.")
