from collections.abc import Iterable, Iterator
from json import JSONDecodeError
import json
import heapq
import os
import regex
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
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        vocab = {
        k: (v if isinstance(v, str) else "".join(v))
        for k, v in vocab.items()
        }
        self.merges = merges
        self.vocab = vocab
        self.merges_dict = {}
        self.merge_ranks = {}
        self.special_tokens = special_tokens
        self.btu = _bytes_to_unicode()
        self.str_to_id = {s: i for i,s in self.vocab.items()}

        self.get_merges_dict()
        self.get_merge_ranks()
        self.special_str_to_id = {}
        self.special_ids = set()
        if special_tokens:
            for s in special_tokens:
                tid = self.str_to_id.get(s)
                if tid is None:
                    continue
                self.special_str_to_id[s] = tid
                self.special_ids.add(tid)
            pat = "(" + "|".join(regex.escape(s) for s in sorted(special_tokens,key=len,reverse=True)) + ")"
            self.special_re = regex.compile(pat)
        else:
            self.special_re = None


    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        start = 256
        merges = []
        merges_dict = {}
        vocab = {}
        with open(vocab_path,'r',encoding='utf-8') as f:
            try:
                raw = json.load(f)
                vocab = {int(v): k for k, v in raw.items()}
            except JSONDecodeError as e:
                print('json解析错误',e)
        """ if(special_tokens):
            existing = set(vocab.values())
            next_id = len(vocab)
            for s in special_tokens:
                if s in existing:
                    continue
                else:
                    vocab[next_id] = s
                    next_id += 1 """

        with open(merges_path,'r',encoding='utf-8') as f:
            for line_no, line in enumerate(f,1):
                line = line.strip()
                if not line:
                    continue
                part = line.split()
                if len(part) != 2:
                    print(f'格式错误: 第{line_no}行')
                    continue

                a_s, b_s = part[0], part[1]
                a_bytes = bytes(_BYTE_DECODER[c] for c in a_s)
                b_bytes = bytes(_BYTE_DECODER[c] for c in b_s)
                merges.append((a_bytes,b_bytes))
                a, b = raw[a_s], raw[b_s]
                pair = (a,b)
                merges_dict[pair] = start
                start += 1
        print('merges解析完毕')
        return cls(vocab, merges, special_tokens)
        

    def get_merge_ranks(self):
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges_dict)}

    def get_merges_dict(self):
        #self.merges:list[tuple(bytes,bytes)] self.merges_dict:dict[tuple(int,int),int]
        if self.merges_dict:
            return
        # raw = {uni_str: seq for seq, uni_str in self.vocab.items()}
        merges_dict = {}
        start = 256
        for pair in self.merges:
            uni_a, uni_b = '', ''
            a, b = pair[0], pair[1]
            for ca in a:
                uni_a += self.btu[ca]
            for cb in b:
                uni_b += self.btu[cb]
            i_pair = (self.str_to_id[uni_a], self.str_to_id[uni_b])
            merges_dict[i_pair] = start
            start += 1
        self.merges_dict = merges_dict
    
    def decode(self, ids):
        pieces = []
        buf = bytearray()

        def flush():
            if buf:
                pieces.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for token_id in ids:
            if token_id in self.special_ids:
                flush()
                pieces.append(self.vocab[token_id])
            else:
                for ch in self.vocab[token_id]:
                    buf.append(_BYTE_DECODER[ch])
        flush()
        return "".join(pieces)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # chunk 是一个字符串，比如文件的一行
            for token_id in self.encode(chunk):
                yield token_id

    def encode(self, text:str):
        if self.special_re == None:
            return self.encode_ordinary(text)
        res = []
        for part in self.special_re.split(text):
            if not part:
                continue
            tid = self.special_str_to_id.get(part)
            if tid is None:
                res.extend(self.encode_ordinary(part))
            else:
                res.append(tid)
        return res

    def encode_ordinary(self,text:str):
        res = []
        heap = []
        words = [word.group() for word in regex.finditer(PAT,text)]
        ids = [[self.str_to_id[self.btu[w]] for w in word.encode('utf-8')] for word in words]
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
        for id in ids:
            Nodes = [_Node(t) for t in id]
            for i in range(len(Nodes) - 1):
                Nodes[i].next = Nodes[i+1]
                Nodes[i+1].prev = Nodes[i]
            counter = 0

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
                new_id = self.merges_dict[pair]
                left.token = new_id
                left.next = right.next
                if right.next is not None:
                    right.next.prev = left
                right.alive = False
                try_push(left)
                try_push(left.prev)
            node = Nodes[0]
            while node is not None:
                if node.alive:
                    res.append(node.token)
                node = node.next
        return res

""" BASE_DIR = os.path.dirname(os.path.abspath(__file__))
merge_path = os.path.join(BASE_DIR, 'owt.merge.txt')
vocab_path = os.path.join(BASE_DIR, 'owt.vocab.json')
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
    vocab_path=vocab_path,
    merges_path=merge_path,
    special_tokens=special_tokens,
)

words = "Hello <|endoftext|> where are you from? 你好<|endoftext|>世界"
encoded = tokenizer.encode(words)
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
print(words == decoded) """
            