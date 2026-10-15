# pretokenize_worker.py
import regex
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_chunk(args):
    """args = (path, start, end)"""
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8", errors="ignore")

    counter = Counter()
    for m in regex.finditer(PAT, text):
        counter[m.group().encode("utf-8")] += 1
    return counter