import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d # 用于平滑曲线（如果没有安装可以 pip install scipy）

# 存放你生成的所有 json 文件路径
json_files = [
    "history_lr_0.0001.json",
    "history_lr_0.0003.json",
    "history_lr_0.0005.json",
    "history_lr_0.0007.json",
    "history_lr_0.001.json",
    "history_lr_0.0015.json",
    "history_lr_0.002.json",
    "history_lr_0.003.json",
]

plt.figure(figsize=(14, 6))

# --- 画图 1：Training Loss (平滑后) ---
plt.subplot(1, 2, 1)
for file in json_files:
    if not os.path.exists(file):
        continue
    
    with open(file, 'r') as f:
        data = json.load(f)
    
    # 提取数据
    steps = [item['step'] for item in data['train']]
    losses = [item['loss'] for item in data['train']]
    
    # 提取文件名里的学习率（用于图例）
    lr_label = file.replace('history_lr_', '').replace('.json', '')
    
    # 因为训练 Loss 噪声很大，我们用高斯滤波平滑一下（sigma越大越平滑）
    smoothed_loss = gaussian_filter1d(losses, sigma=5)
    
    # 画图
    plt.plot(steps, smoothed_loss, label=f'lr={lr_label}', linewidth=1.5)

plt.title('Training Loss vs. Steps')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.legend() # 显示图例
plt.grid(True, linestyle='--', alpha=0.7)

# --- 画图 2：Validation Loss ---
plt.subplot(1, 2, 2)
for file in json_files:
    if not os.path.exists(file):
        continue
    with open(file, 'r') as f:
        data = json.load(f)
    
    steps = [item['step'] for item in data['val']]
    losses = [item['loss'] for item in data['val']]
    lr_label = file.replace('history_lr_', '').replace('.json', '')
    
    # 验证集本身点数少，且波动大，可以不做平滑或用较小的 sigma
    plt.plot(steps, losses, label=f'lr={lr_label}', marker='o', markersize=4)

plt.title('Validation Loss vs. Steps')
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=300)
plt.show()