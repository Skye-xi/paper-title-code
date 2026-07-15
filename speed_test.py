"""
BERT 推理速度基准测试
====================
测试 BERT-base-Chinese 微调模型在 CPU 上的推理吞吐量。
"""
import time, torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(8)
path = 'bert_culture_classifier'
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval()
print(f'模型加载: {time.time()-t0:.1f}s')

texts = ['测试文本内容' + str(i) for i in range(100)]
t0 = time.time()
with torch.no_grad():
    for i in range(0, 100, 32):
        batch = texts[i:i+32]
        inputs = tokenizer(batch, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        outputs = model(**inputs)
elapsed = time.time() - t0
speed = 100 / elapsed
print(f'100条: {elapsed:.1f}s ({speed:.1f} 条/s)')
print(f'8306条预估: {8306/speed:.0f}s = {8306/speed/60:.1f}分钟')
