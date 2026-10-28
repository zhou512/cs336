# pretokenize_worker.py
import regex
from collections import Counter

# 原始正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_chunk(args):
    """args = (path, start, end, special_tokens)"""
    path, start, end, special_tokens = args
    
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    
    # 解码后立刻释放原 bytes
    text = chunk.decode("utf-8", errors="ignore")
    del chunk
    
    counter = Counter()
    
    if not special_tokens:
        # 没有特殊 token 时，直接全局扫描
        for m in regex.finditer(PAT, text):
            counter[m.group().encode("utf-8")] += 1
        return counter

    # 构造匹配特殊 token 的正则
    split_part = '|'.join(regex.escape(t) for t in special_tokens)
    
    last_end = 0
    
    # 利用 finditer 流式寻找特殊 token，不产生 parts 列表
    for m in regex.finditer(split_part, text):
        # 1. 处理特殊 token 前面的普通文本片段
        if m.start() > last_end:
            part = text[last_end:m.start()]
            for match in regex.finditer(PAT, part):
                counter[match.group().encode("utf-8")] += 1
                
        # 2. 处理特殊 token 本身
        part = text[m.start():m.end()]
        for match in regex.finditer(PAT, part):
            continue
            
        last_end = m.end()
        
    # 3. 处理最后一个特殊 token 之后的剩余文本
    if last_end < len(text):
        part = text[last_end:]
        for match in regex.finditer(PAT, part):
            counter[match.group().encode("utf-8")] += 1

    return counter