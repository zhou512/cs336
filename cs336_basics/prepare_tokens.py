from tokenizer import Tokenizer
from tqdm import tqdm
import os
import numpy as np
P_DATA = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
merge_path = os.path.join(BASE_DIR, 'tns.merge.txt')
vocab_path = os.path.join(BASE_DIR, 'tns.vocab.json')
special_tokens = ["<|endoftext|>"]
src = os.path.join(BASE_DIR, '..', 'data', 'TinyStoriesV2-GPT4-train.txt')
tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)

def prepare_data(input_path, output_path, tokenizer:Tokenizer, chunk_size=100_000_000):
    buffer = []
    print('开始处理')
    with open(output_path, 'wb') as f:
        pass 
    # 打开输入文件，同时以追加二进制模式（'ab'）打开输出文件
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'ab') as f_out:
        for token_id in tqdm(tokenizer.encode_iterable(f_in), desc='tokenizer'):
            buffer.append(token_id)
            if len(buffer) >= chunk_size:
                # 将文件对象 f_out 传给 tofile，直接追加写入
                np.array(buffer, dtype=np.uint16).tofile(f_out)
                buffer = []
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)
    print(f"处理完成！保存至 {output_path}")
bin_path = 'train.bin'

prepare_data(src, bin_path, tokenizer)