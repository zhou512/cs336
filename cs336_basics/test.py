from transformer import TransformerLM, AdamW, learning_rate_schedule, gradient_clipping, save_checkpoint, load_checkpoint, data_loading, cross_entropy, decoding, fixed_data_loading
from tokenizer import Tokenizer
from tqdm import tqdm
from dataclasses import dataclass
import torch
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(BASE_DIR, '..', 'eval.bin')
merge_path = os.path.join(BASE_DIR, 'tns.merge.txt')
vocab_path = os.path.join(BASE_DIR, 'tns.vocab.json')
special_tokens = ["<|endoftext|>"]
check_point_path = os.path.join(BASE_DIR, '..', 'ckpt0320000.pt')
src = os.path.join(BASE_DIR, '..', 'data', 'TinyStoriesV2-GPT4-valid.txt')
l2_max = 1.0
vocab_size = 10000
num_layers = 4
num_heads = 16
d_model = 512
d_ff =  1344
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
training_loop = 10000
tlm = TransformerLM(vocab_size, num_layers, d_model, num_heads, d_ff, context_length=context_length, device=device)
adamw = AdamW(tlm.parameters(), lr_max, lr_min, t_w, t_c, beta1, beta2, λ)
tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)
text = 'one day'
load_checkpoint(check_point_path, tlm, adamw)
generating = decoding(tlm, tokenizer, text, 200, 0.7, 30)
print(generating)
tlm.eval()
ids_eval = np.memmap(bin_path, dtype=np.uint16, mode='r')
val_loss = 0
start_index = 0
val_loop = 333
""" with torch.no_grad():
    for i in range(val_loop):
        x, y = fixed_data_loading(start_index, ids_eval, batch_size, context_length, device=device)
        logits = tlm(x)
        val_loss += cross_entropy(logits, y).item()
        start_index += context_length
print(f'验证集loss = {(val_loss / val_loop):.4f}') """