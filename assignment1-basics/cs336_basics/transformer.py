import torch
import torch.nn as nn
from einops import rearrange, einsum
from collections.abc import Callable, Iterable
from typing import Optional
from tokenizer import Tokenizer
import math
import random
import numpy as np

L2_MAX = 1e5
class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features,device=device, dtype=dtype))
        self.init_weight()

    def init_weight(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0.0, std, -3*std, 3*std)

    def forward(self, x):
        x = einsum(x, self.weight, "... din, dout din -> ... dout")
        return x

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.trunc_normal_(self.embeddings, 0.0, 1.0, -3.0, 3.0)

    def forward(self, x:torch.Tensor):
        """x为(batch_size, sequence_length)， 每个元素都对应一个token id，在self.embeddings中找到token_id对应的d_model向量，赋值给这个元素"""
        if x.dtype != torch.long:
            raise TypeError(f"Embedding expects int64/long, got {x.dtype}")
        return self.embeddings[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        y = x * rms * self.g
        return y.to(in_dtype)

class SiLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_in = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_in:
            self.d_in = d_in
        else:
            self.d_in = self.verify_din()
        self.w1 = Linear(self.d_model, self.d_in, device=device, dtype = dtype)
        self.w3 = Linear(self.d_model, self.d_in, device=device, dtype = dtype)
        self.w2 = Linear(self.d_in, self.d_model, device=device, dtype = dtype)

    def verify_din(self):
        ori_din = self.d_model * 8 // 3
        return (ori_din + 63) // 64 * 64

    def forward(self, x):
        gate = self.w1(x)
        activated = (gate * torch.sigmoid(gate) * self.w3(x))
        return self.w2(activated)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        f = 1.0 /  theta ** (torch.arange(0, d_k, 2, device=device)[:d_k // 2] / d_k).float()
        t_position = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t_position, f)
        #freqs.shape = (max_seq_len, d_k // 2)
        emb = freqs.repeat_interleave(2,-1)
        #emb = cos_buf = sin_buf .shape = (max_seq_len, d_k)
        self.register_buffer('cos_buf', emb.cos(), persistent=False)
        self.register_buffer('sin_buf', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos = self.cos_buf[token_positions].to(x.dtype)
        sin = self.sin_buf[token_positions].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotate_x = torch.stack((-x2, x1), dim=-1).flatten(-2) 
        return x * cos + rotate_x * sin

def softmax(x: torch.Tensor, dim: int):
    y = x - x.max(dim=dim, keepdim=True).values
    ex_x = y.exp()
    ex_sum = ex_x.sum(dim=dim, keepdim=True)
    return ex_x / ex_sum

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor, mask:torch.Tensor = None):
    d_k = k.shape[-1]
    d_v = v.shape[-1]
    scores = einsum(q, k, '... q_len d_k, ... k_len d_k -> ... q_len k_len') * d_k ** -0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = softmax(scores, -1)
    output = einsum(attn, v, '... q_len k_len, ... k_len d_v -> ... q_len d_v')
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, theta=10000, context_length = 256, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        assert self.d_k % 2 == 0
        self.device = device
        self.context_length = context_length
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, self.context_length, self.device)

    def forward(self, x, token_positions=None):
        #x.shape = (batch_size,seq_length,d_model) = q.shape = k.shape = v.shape
        B, S, D = x.shape
        assert S <= self.context_length, f"Input seq_len {S} exceeds max {self.context_length}"
        #(d,d) @ (b, s ,d) --> (b,s,d)  4 * 2 * bsd^2 + 2 * s * d / 2 + 2 * (bs^2d * 2) 
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        mask = torch.tril(torch.ones(S, S, dtype=bool, device=x.device))
        if token_positions is None:
            token_positions = torch.arange(0, S, device=x.device)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        attn = scaled_dot_product_attention(q, k, v, mask)
        #attn.shape = (b,h,s,k)
        attn = rearrange(attn, 'b h s d -> b s (h d)')
        output = self.W_O(attn)
        return output
class MultiHeadSelfAttentionNope(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        assert self.d_k % 2 == 0
        self.device = device
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        # self.rope = RotaryPositionalEmbedding(theta, self.d_k, self.context_length, self.device)

    def forward(self, x, token_positions=None):
        #x.shape = (batch_size,seq_length,d_model) = q.shape = k.shape = v.shape
        B, S, D = x.shape
        #(d,d) @ (b, s ,d) --> (b,s,d)  4 * 2 * bsd^2 + 2 * s * d / 2 + 2 * (bs^2d * 2) 
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        mask = torch.tril(torch.ones(S, S, dtype=bool, device=x.device))
        attn = scaled_dot_product_attention(q, k, v, mask)
        #attn.shape = (b,h,s,k)
        attn = rearrange(attn, 'b h s d -> b s (h d)')
        output = self.W_O(attn)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000, context_length=None, device=None, dtype=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, theta, context_length, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        y = x + self.mhsa(self.rmsnorm1(x))
        z = y + self.swiglu(self.rmsnorm2(y))
        return z

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff,theta=10000, context_length=None, device=None, dtype=None):
        super().__init__()
        if dtype is not None and not dtype.is_floating_point:
            dtype = torch.float32
        self.embed = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            theta=theta,
            context_length=context_length,
            device=device,
            dtype=dtype
            ) for _ in range(num_layers)])
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.lm_head.weight = self.embed.embeddings
        nn.init.trunc_normal_(self.embed.embeddings, 0.0, 0.02, -3*0.02, 3*0.02)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.rmsnorm(x)
        #(b,s,d) @ (d,vocab_size) ->(b,s,vocab_size)
        logits = self.lm_head(x)
        return logits

def cross_entropy(logits: torch.Tensor, x: torch.Tensor):
    #logits.shape = (b,s,vocab_size)
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_exp_sum = torch.log(torch.exp(shifted_logits).sum(dim=-1))
    targets = shifted_logits.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    loss = log_exp_sum - targets
    return loss.mean()


""" tlm = TransformerLM(100, 10, 16, 4, 32)
print(type(tlm.parameters())) """

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr_max, lr_min, t_w, t_c, β1, β2, λ, eps=1e-8):
        self.t_w = t_w
        self.t_c = t_c
        self.lr_max = lr_max
        self.lr_min = lr_min
        defaults = dict(betas=(β1, β2), lmd=λ, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure:Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lmd = group['lmd']
            eps = group['eps']
            for p in group['params']:
                if p.dtype == torch.bfloat16 and eps <= 1e-8:
                    eps = 1e-6
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p)
                    state['t'] = 1
                    state['m'] = m
                    state['v'] = v
                lr = learning_rate_schedule(state['t'], self.lr_max, self.lr_min, self.t_w, self.t_c)
                lr_t = lr * (1 - beta2 ** state['t']) ** 0.5 / (1 - beta1 ** state['t'])
                p.data.mul(1 - lr * lmd)
                state['m'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                p.data.addcdiv_(state['m'], torch.sqrt(state['v']) + eps, value=-lr_t)
                state['t'] += 1

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    if t < t_w:
        return t / t_w * lr_max
    elif t > t_c:
        return lr_min
    else:
        x = 1 + math.cos((t - t_w) / (t_c - t_w) * torch.pi)
        return lr_min + 0.5 * x * (lr_max -  lr_min)

def gradient_clipping(l2_max, params, eps = 1e-6):
    parameters = [p for p in params if p.grad is not None]
    total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2)
    if total_norm.item() > l2_max:
        x = l2_max / (total_norm.item() + eps)
        for p in parameters:
            p.grad.detach().mul_(x)

def data_loading(ids: np.ndarray, batch_size, context_length, device: torch.device):
    chunk_list = []
    target_list = []
    
    for _ in range(batch_size):
        # 独立随机找一个起点
        start_index = random.randint(0, len(ids) - context_length - 2)
        
        # 截取数据并加入列表
        chunk_list.append(ids[start_index : start_index + context_length])
        target_list.append(ids[start_index + 1 : start_index + context_length + 1])
    
    # 将 list 转成 numpy 数组，形状直接变成 (batch_size, context_length)
    chunk = np.stack(chunk_list)
    target = np.stack(target_list)
    
    # 转成 int64 (torch.long)，因为 PyTorch 的 Embedding 和 CrossEntropy 需要 long 类型
    # .astype 会触发一次内存拷贝，避免 memmap 共享内存导致的潜在报错
    input_x = torch.from_numpy(chunk.astype(np.int64)).to(device)
    input_y = torch.from_numpy(target.astype(np.int64)).to(device)
    
    return input_x, input_y

def fixed_data_loading(start_index, ids: np.ndarray, batch_size, context_length, device: torch.device):
    chunk_list = []
    target_list = []
    
    for _ in range(batch_size):
        if(start_index + context_length >= len(ids)):
            return
        # 截取数据并加入列表
        chunk_list.append(ids[start_index : start_index + context_length])
        target_list.append(ids[start_index + 1 : start_index + context_length + 1])
    
    # 将 list 转成 numpy 数组，形状直接变成 (batch_size, context_length)
    chunk = np.stack(chunk_list)
    target = np.stack(target_list)
    
    # 转成 int64 (torch.long)，因为 PyTorch 的 Embedding 和 CrossEntropy 需要 long 类型
    # .astype 会触发一次内存拷贝，避免 memmap 共享内存导致的潜在报错
    input_x = torch.from_numpy(chunk.astype(np.int64)).to(device)
    input_y = torch.from_numpy(target.astype(np.int64)).to(device)
    
    return input_x, input_y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration, out):
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    check_point = {
        'model_dict': model_dict,
        'opt_dict': opt_dict,
        'iteration': iteration
    }
    torch.save(check_point, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    check_point = torch.load(src)
    model_dict = check_point['model_dict']
    opt_dict = check_point['opt_dict']
    iteration = check_point['iteration']
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(opt_dict)
    return iteration

@torch.no_grad()
def decoding(model:TransformerLM, tokenizer:Tokenizer, text:str, max_tokens, temperature, k):
    eos_id = tokenizer.special_str_to_id.get("<|endoftext|>", None)
    tokens = tokenizer.encode(text)
    x_gen = torch.tensor(tokens, dtype=torch.long, device='cuda').unsqueeze(0)
    while x_gen.shape[1] - len(tokens) < max_tokens:
        logits = model(x_gen)
        #x_gen.shape=(1,seq_len)
        last_logit = logits[:, -1, :]
        last_logit.div_(temperature)
        probs = softmax(last_logit, dim=-1)
        k = min(k, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(topk_probs, num_samples=1)
        next_token = torch.gather(topk_indices, -1, next_token)
        if eos_id is not None and next_token.item() == eos_id:
            break
        x_gen = torch.cat((x_gen, next_token), dim=-1)
    x_gen = x_gen[0].tolist()
    return tokenizer.decode(x_gen)
