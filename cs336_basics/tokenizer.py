from collections.abc import Iterable, Iterator
from json import JSONDecodeError
import json
import heapq
import os
class _Node:
    __slots__ = ("token", "prev", "next", "alive")
    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None
        self.alive = True

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}

class Tokenizer:
    def __init__(self, merges, vocab, special_tokens=None):
        self.merges = merges
        self.vocab = vocab
        self.merge_ranks = {}

    @classmethod
    def from_files(cls, merges_path, vocab_path, special_tokens=None):
        start = 256
        merges = {}
        vocab = {}
        with open(merges_path,'r',encoding='utf-8') as f:
            for line_no, line in enumerate(f,1):
                line = line.strip()
                if not line:
                    continue
                part = line.split()
                if len(part) != 2:
                    print(f'格式错误: 第{line_no}行')
                a, b = int(part[0]), int(part[1])
                pair = (a,b)
                merges[pair] = start
                start += 1
        print('merges解析完毕')
        with open(vocab_path,'r',encoding='utf-8') as f:
            try:
                raw = json.load(f)
                vocab = {int(v): k for k, v in raw.items()}
            except JSONDecodeError as e:
                print('json解析错误',e)
        if(special_tokens):
            vocab_size = int(max(vocab))
            for i, s in enumerate(special_tokens,vocab_size + 1):
                vocab[i] = s
        return merges, vocab

    def get_merge_ranks(self):
        for i,pair in enumerate(self.merges):
            self.merge_ranks[pair] = i

    """ def encode(self,text:str) -> list[int]:
        heap = []
        ids = list(text.encode('utf-8'))
        #建一个关于rank的最小堆，每次先弹出堆顶元素merge,还要存放pair的位置信息
        for pos,pair in enumerate(zip(ids,ids[1:])):
            if pair in self.merges:
                heapq.heappush(heap,(self.merge_ranks[pair], pair, pos))
        while heap:
            rank, pair, pos = heapq.heappop(heap)
            real_idx = rank + 256
            ids[pos:pos+2] = [real_idx]
            if pos > 0:
                pre_pair = (ids[pos-1],real_idx)
                r = self.merge_ranks.get(pre_pair)
                if r is not None:
                    heapq.heappush(heap,(self.merge_ranks[pre_pair],pre_pair,pos-1))
            if pos < len(ids) - 1:
                post_pair = (real_idx,ids[pos+1])
                r = self.merge_ranks.get(post_pair)
                if r is not None:
                    heapq.heappush(heap,(self.merge_ranks[post_pair],post_pair,pos))
        return ids """
    
    def decode(self, ids):
        buf = bytearray()
        for token_id in ids:
            for ch in self.vocab[token_id]:
                buf.append(_BYTE_DECODER[ch])
        return buf.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # chunk 是一个字符串，比如文件的一行
            for token_id in self.encode(chunk):
                yield token_id

    
    def encode(self,text:str):
        heap = []
        ids = list(text.encode('utf-8'))
        Nodes = [_Node(t) for t in ids]
        for i in range(len(Nodes) - 1):
            Nodes[i].next = Nodes[i+1]
            Nodes[i+1].prev = Nodes[i]
        counter = 0

        def try_push(left:_Node):
            nonlocal counter
            if left is None or left.alive == False:
                return
            right = left.next
            if right is None or right.alive == False:
                return
            pair = (left.token, right.token)
            r = self.merge_ranks.get(pair)
            if r is not None:
                heapq.heappush(heap, (r,counter,left))
                counter += 1

        for node in Nodes:
            try_push(node)

        while heap:
            rank, _, left = heapq.heappop(heap)
            if left is None or not left.alive:
                continue
            right = left.next
            if right is None or not right.alive:
                continue
            pair = (left.token, right.token)
            if(self.merge_ranks.get(pair) != rank):
                continue
            new_id = self.merges[pair]
            left.token = new_id
            left.next = right.next
            if right.next is not None:
                right.next.prev = left
            right.alive = False
            try_push(left)
            try_push(left.prev)

        res = []
        node = Nodes[0]
        while node is not None:
            if node.alive:
                res.append(node.token)
            node = node.next
        return res

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
merge_path = os.path.join(BASE_DIR, 'bpe.merge.txt')
vocab_path = os.path.join(BASE_DIR, 'bpe.vocab.json')
merges, vocab = Tokenizer.from_files(merges_path=merge_path, vocab_path=vocab_path)
tokenizer = Tokenizer(merges,vocab)
tokenizer.get_merge_ranks()
words = "hello ajwpeiojpawoifksamn;lfkn，你好"
encoded = tokenizer.encode(words)
print(encoded)
print(tokenizer.decode(encoded))
print(words == tokenizer.decode(encoded))
            