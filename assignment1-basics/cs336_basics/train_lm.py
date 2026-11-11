from transformer import TransformerLM, AdamW, learning_rate_schedule, gradient_clipping, save_checkpoint, load_checkpoint, data_loading, cross_entropy
from tokenizer import Tokenizer
from tqdm import tqdm
from dataclasses import dataclass
import torch
import os
import numpy as np
import time
import json
torch.backends.cuda.matmul.allow_tf32 = True
import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        
set_seed(3407)  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
merge_path = os.path.join(BASE_DIR, 'tns.merge.txt')
vocab_path = os.path.join(BASE_DIR, 'tns.vocab.json')
special_tokens = ["<|endoftext|>"]
src = os.path.join(BASE_DIR, '..', 'data', 'TinyStoriesV2-GPT4-train.txt')
bin_path = os.path.join(BASE_DIR, '..', 'train.bin')
eval_path = os.path.join(BASE_DIR, '..', 'eval.bin')
l2_max = 1.0
vocab_size = 10000
num_layers = 4
num_heads = 16
d_model = 512
d_ff = 1344
batch_size = 64
context_length = 256
beta1 = 0.9
beta2 = 0.95
lr_max = 15e-4
lr_min = lr_max / 10
t_w = 300
t_c = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
λ = 0.01
training_loop = 20000
tlm = TransformerLM(vocab_size, num_layers, d_model, num_heads, d_ff, context_length=context_length, device=device)
adamw = AdamW(tlm.parameters(), lr_max, lr_min, t_w, t_c, beta1, beta2, λ)
tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)
t0 = time.perf_counter()
ids = np.memmap(bin_path, dtype=np.uint16, mode='r')
ids_eval = np.memmap(eval_path, dtype=np.uint16, mode='r')
t1 = time.perf_counter()
print(f'token序列加载完成，用时{(t1-t0) * 1000:.4f}ms')
train_loss_history = []
val_loss_history = []
for t in range(training_loop):
    tlm.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # t2 = time.perf_counter()
        x, y = data_loading(ids, batch_size, context_length, device=device)
        # if device == 'cuda': torch.cuda.synchronize()
        # t3 = time.perf_counter()
        logits = tlm(x)
        loss = cross_entropy(logits, y)
        # if device == 'cuda': torch.cuda.synchronize()
        # t4 = time.perf_counter()
    adamw.zero_grad(set_to_none=True)
    loss.backward()
    gradient_clipping(l2_max, tlm.parameters())
    adamw.step()
    # if device == 'cuda': torch.cuda.synchronize()
    # t5 = time.perf_counter()
    if (t+1) % 100 == 0:
        current_lr = learning_rate_schedule(t+1, lr_max, lr_min, t_w, t_c)
        train_loss_history.append({
            "step": t + 1,
            "loss": loss.item(),
            "lr": current_lr
        })
        print(f'step{t+1}:')
        """ print(f'数据加载用时{(t3-t2) * 1000:.4f}ms')
        print(f'前向传播用时{(t4-t3) * 1000:.4f}ms')
        print(f'反向传播用时{(t5-t4) * 1000:.4f}ms') """
        print(f'学习率 = {learning_rate_schedule(t+1, lr_max, lr_min, t_w, t_c):.5f}')
        print(f'training loss = {loss.item():.4f}')
    if (t+1) % 5000 == 0:
        save_seq = str(t + 1)
        save_path = 'ckpt03' + save_seq + '.pt'
        save_checkpoint(tlm, adamw, t+1, save_path)
    if (t+1) % 2000 == 0:
        tlm.eval()
        val_loss = 0.0
        val_steps = 20  # 跑 20 个 batch 算平均
        with torch.no_grad():
            for i in range(val_steps):
                x, y = data_loading(ids_eval, batch_size, context_length, device=device)
                logits = tlm(x)
                val_loss += cross_entropy(logits, y).item()
            avg_val_loss = val_loss / val_steps
        val_loss_history.append({
            "step": t + 1,
            "loss": avg_val_loss
        })
        print(f'step{t+1}, 验证集loss = {avg_val_loss:.4f}')
        history_data = {
        "train": train_loss_history,
        "val": val_loss_history
        }
        with open(f'history_lr_{lr_max}.json', 'w') as f:
            json.dump(history_data, f)
        print("训练历史已保存！")