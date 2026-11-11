from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from .pre_tokenizer_worker import pretokenize_chunk
from multiprocessing import Pool
import heapq
import regex
import os
import time
import json
import pickle
import tracemalloc
import psutil

REBUILD_INTERVAL = 1000
SAVE_INTERVAL = 15000
BASEDIR = os.path.dirname(os.path.abspath(__file__))
# input_path = os.path.join(BASEDIR,'..','data','owt_train.txt')
input_path = os.path.join(BASEDIR,'..','data','TinyStoriesV2-GPT4-train.txt')

class _RevKey:
    """包装一个可比较的值，反转比较方向（用于让字节内容更大的优先）。"""
    __slots__ = ('data',)
    def __init__(self, data):
        self.data = data
    def __lt__(self, other):
        return self.data > other.data   # 反转
    def __eq__(self, other):
        return self.data == other.data
    
def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def rebuild(ckpt):
    words = ckpt["words"]
    vocab = ckpt["vocab"]
    freqs, ptw = {}, {}
    for wid, (f, ids) in words.items():
        for a, b in zip(ids, ids[1:]):
            p = (a << 20) | b
            freqs[p] = freqs.get(p, 0) + f
            ptw.setdefault(p, set()).add(wid)
    heap = []
    for pair, count in freqs.items():
        a, b = pair >> 20, pair & ((1 << 20) - 1)
        heap.append((-count, _RevKey((vocab[a], vocab[b])), pair))
    heapq.heapify(heap)
    return freqs, ptw, heap

def bytes_to_unicode():
    """
    返回一个映射：0~255 的每个字节 -> 一个可见的 unicode 字符。
    可逆，用于把任意 bytes token 安全地放进 JSON / 字符串。
    """
    # 可见 ASCII 段 + 拉丁补充段，直接一一对应
    bs = (
        list(range(ord("!"), ord("~") + 1))      # 33..126
        + list(range(ord("¡"), ord("¬") + 1))    # 161..172
        + list(range(ord("®"), ord("ÿ") + 1))    # 174..255
    )
    cs = bs[:]
    n = 0
    # 剩下的字节（控制字符、0x80~0xA0 等）映射到 256, 257, ... 对应的字符
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def save_checkpoint(path, words, merges, vocab, it, w_id):
    with open(path, "wb") as f:
        pickle.dump({
            "words": words,       # word_id -> (freq, tuple_ids)
            "merges": merges,
            "vocab": vocab,
            "iter": it,
            "w_id": w_id,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def vocab_init():
    return {i:bytes([i]) for i in range(256)}

def build_pre_freqs(path: str, num_workers: int, special_tokens: list[str]):
    boundry_token = special_tokens[0].encode('utf-8')
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 64, boundry_token)

    tasks = [(path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    pre_freqs = Counter()
    # chunksize=1 即可，因为每个 task 本身就很重
    with Pool(processes=num_workers) as pool:
        for counter in pool.imap_unordered(pretokenize_chunk, tasks, chunksize=1):
            pre_freqs.update(counter)
            # 显式释放子进程传来的 Counter 对象（可选，但有助于 GC）
            del counter 

    # 合并
    return pre_freqs


def train_bpe(input_path, vocab_size, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = vocab_init()
    t_pretokenizer0 = time.time()
    pre_freqs = build_pre_freqs(input_path, 4, special_tokens)
    t_pretokenizer1 = time.time()
    print(f'预分词用时：{(t_pretokenizer1 - t_pretokenizer0) *1000:.4f}ms')
    #pre_freqs: bytes -> freq
    heap = []
    freqs = {}
    words = {}
    merges = []
    ptw = {}
    w_id = 0
    # tracemalloc.start()
    t_pretokenizer2 = time.time()
    for bs, pf in pre_freqs.items():
        ids = tuple(bs)
        words[w_id] = (pf, ids)
        for a, b in zip(ids,ids[1:]):
            pair = (a << 20) | b
            freqs[pair] = freqs.get(pair,0) + pf
            ptw.setdefault(pair,set()).add(w_id)
        w_id += 1
    for pair,count in freqs.items():
        pa, pb = pair >> 20, pair & ((1 << 20) - 1)
        heapq.heappush(heap, (-count, _RevKey((vocab[pa], vocab[pb])), pair))
    t_pretokenizer3 = time.time()
    print(f'转换用时：{(t_pretokenizer3 - t_pretokenizer2) *1000:.4f}ms')
    del pre_freqs


    merge_time = vocab_size - 256 - len(special_tokens)

    for iter in range(merge_time):
        if iter % REBUILD_INTERVAL == 0 and iter != 0:
            t_rebuild0 = time.time()
            heap = []
            for pair,count in freqs.items():
                pa, pb = pair >> 20, pair & ((1 << 20) - 1)
                heap.append((-count, _RevKey((vocab[pa], vocab[pb])), pair))
            heapq.heapify(heap)
            t_rebuild1 = time.time()
            print(f'重建堆用时:{(t_rebuild1 - t_rebuild0) * 1000}ms')
        """ if iter % SAVE_INTERVAL == 0 and iter != 0:
            t_save0 = time.time()
            save_checkpoint(os.path.join(BASEDIR,'checkpoint.pkl'),words,merges,vocab,iter,w_id)
            t_save1 = time.time()
            print(f'保存检查点用时:{(t_save1 - t_save0) * 1000:.4f}ms') """
        idx = 256 + iter
        #不重新创建freqs和ptw，而是只维护增量
        t_eit0 = time.time()
        if not freqs:
            break
        max_key = None
        # max_key = max(freqs,key=freqs.get)
        while heap:
            neg_cnt, _, pair = heap[0]
            real_cnt = freqs.get(pair,0)
            if real_cnt == 0 or -neg_cnt != real_cnt:
                heapq.heappop(heap)
            else:
                max_key = pair
                break
        if max_key == None:
            break
        affected = ptw.pop(max_key,set())
        a = max_key >> 20
        b = max_key & ((1 << 20) - 1)
        changed = set()
        for word_id in affected:
        #每个word都是受到影响的单词，要维护ptw：1.删除[max_key]键值对 2.把所有值列表中的受影响单词(包含max_key的)用新词替换(只能遍历？)
        #words = word_id ->(ids,freq)， ptw = pair -> [word_id]
        #把words中受影响的单词去掉,添加新单词，在ptw中找到受影响的序号，用新的换掉
            i = 0
            new_word = []
            freq,word= words.pop(word_id)
            old_pairs = Counter((x << 20) | y for x,y in zip(word,word[1:]))
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_pairs = Counter((x << 20) | y for x,y in zip(new_word,new_word[1:]))
            for p in set(old_pairs) | set(new_pairs):
                oc = old_pairs.get(p,0)
                nc = new_pairs.get(p,0)
                if(oc == nc):
                    continue
                else:
                    changed.add(p)
                    #nc = 0, oc != 0-->pair在原来的单词里出现过，现在没有了，要减,nc != 0,oc = 0，原来的单词中没出现过这个对，要加
                    freqs[p] = freqs.get(p,0) + freq * (nc - oc)
                    if freqs[p] == 0:
                        freqs.pop(p)
                    if nc > 0:
                        ptw.setdefault(p,set()).add(word_id)
                    else:
                        s = ptw.get(p)
                        if s is not None:
                            s.discard(word_id)
                            if not s:
                                del ptw[p]
            new_word = tuple(new_word)
            words[word_id] = (freq, new_word)
        merges.append((vocab[a], vocab[b]))
        vocab[idx] = vocab[a] + vocab[b]
        for p in changed:
            if p in freqs:
                pa, pb = p >> 20, p & ((1 << 20) - 1)
                heapq.heappush(heap, (-freqs[p], _RevKey((vocab[pa], vocab[pb])), p))
        t_eit1 = time.time()
        if((iter+1) % 50 == 0):
            print(f'step{iter+1}: 用时{(t_eit1 - t_eit0) * 1000:.4f}ms')
    # cur, peak = tracemalloc.get_traced_memory()
    # print(f"peak = {peak / 1024**3:.2f} GB")
    for i, tok in enumerate(special_tokens):
        vocab[vocab_size - len(special_tokens) + i] = tok.encode("utf-8")
    return vocab, merges

def save_merge(merge,path):
    with open(path + '.merge.txt','w',encoding='utf-8') as f:
        btu = bytes_to_unicode()
        for a, b in merge:
            unia, unib = '', ''
            for ca in a:
                unia += btu[ca]
            for cb in b:
                unib += btu[cb]
            f.write(f'{unia} {unib}\n')

def save_vocab(vocab, path):
    b2u = bytes_to_unicode()
    out = {
        "".join(b2u[b] for b in bs): tid
        for tid, bs in vocab.items()
    }
    with open(path + ".vocab.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    t0 = time.time()
   
    vocab, merges = train_bpe(input_path, 10000, ['<|endoftext|>'])
    t1 = time.time()
    print(f'used time:{(t1-t0) * 1000:.4f}ms')
    save_merge(merges,os.path.join(BASEDIR,'tns'))
    save_vocab(vocab,os.path.join(BASEDIR,'tns'))
    #p = psutil.Process(os.getpid())
    #print(f"peak RSS = {p.memory_info().peak_wset / 1024**3:.2f} GB")
    # print(merges)
    # print(vocab)