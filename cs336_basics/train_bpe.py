from collections import Counter
from pretokenization_example import find_chunk_boundaries
from pre_tokenizer_worker import pretokenize_chunk
from multiprocessing import Pool
import heapq
import regex
import os
import time
BASEDIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASEDIR,'..','tests','fixtures','tinystories_sample_5M.txt')
def vocab_init():
    return {i:bytes([i]) for i in range(256)}

def build_pre_freqs(path: str, num_workers: int, special_token: bytes):
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, special_token)

    tasks = [(path, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    # chunksize=1 即可，因为每个 task 本身就很重
    with Pool(processes=num_workers) as pool:
        counters = pool.map(pretokenize_chunk, tasks, chunksize=1)

    # 合并
    pre_freqs = Counter()
    for c in counters:
        pre_freqs.update(c)
    return pre_freqs


def train_bpe(input_path, vocab_size, special_tokens):
    #O(M*N*L)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = vocab_init()
    pre_freqs = build_pre_freqs(input_path, min(os.cpu_count() or 1, 16), special_tokens)
    #pre_freqs: bytes -> freq
    heap = []
    freqs = {}
    words = {}
    merges = {}
    ptw = {}
    w_id = 0

    for bs, pf in pre_freqs.items():
        ids = tuple(bs)
        words[w_id] = (pf, ids)
        for pair in zip(ids,ids[1:]):
            freqs[pair] = freqs.get(pair,0) + pf
            ptw.setdefault(pair,set()).add(w_id)
        w_id += 1
    for pair,count in freqs.items():
        heapq.heappush(heap,(-count,pair))
    del pre_freqs


    merge_time = vocab_size - 256

    for iter in range(merge_time):
        idx = 256 + iter
        #不重新创建freqs和ptw，而是只维护增量
        #M个单词，单词平均长度为N O(M*N)
        if not freqs:
            break
        max_key = None
        # max_key = max(freqs,key=freqs.get)
        while heap:
            neg_cnt,pair = heap[0]
            real_cnt = freqs.get(pair,0)
            if real_cnt == 0 or -neg_cnt != real_cnt:
                heapq.heappop(heap)
            else:
                max_key = pair
                break
        if max_key == None:
            break
        affected = ptw.pop(max_key,set())
        for word_id in list(affected):
        #每个word都是受到影响的单词，要维护ptw：1.删除[max_key]键值对 2.把所有值列表中的受影响单词(包含max_key的)用新词替换(只能遍历？)
        #words = word_id ->(ids,freq)， ptw = pair -> [word_id]
        #把words中受影响的单词去掉,添加新单词，在ptw中找到受影响的序号，用新的换掉
            i = 0
            new_word = []
            freq,word= words.pop(word_id)
            old_pairs = Counter(zip(word,word[1:]))
            while i < len(word):
                if i < len(word) - 1 and word[i] == max_key[0] and word[i+1] == max_key[1]:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_pairs = Counter(zip(new_word,new_word[1:]))
            new_word = tuple(new_word)
            for p,c in old_pairs.items():
                freqs[p] -= freq * c
                if freqs[p] <= 0:
                    freqs.pop(p)
                s = ptw.get(p)
                if s is not None:
                    s.discard(word_id)
                    if not s:
                        del ptw[p]
            for p,c in new_pairs.items():
                freqs[p] = freqs.get(p,0) + freq * c
                ptw.setdefault(p,set()).add(word_id)
                heapq.heappush(heap,(-freqs[p],p))
            words[word_id] = (freq, new_word)
        merges[idx] = max_key
        vocab[idx] = vocab[max_key[0]] + vocab[max_key[1]]
    return merges,vocab

if __name__ == '__main__':
    t0 = time.time()                     # ✅ 关键
    merges, vocab = train_bpe(input_path, 10000, b'<|endoftext|>')
    t1 = time.time()
    print(f'used time:{(t1-t0) * 1000}ms')
    # print(merges)
    # print(vocab)