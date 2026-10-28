import os

def extract_sample(src_path, dst_path, target_bytes, special_token=b'<|endoftext|>'):
    """从头开始，一直复制到累计字节数 >= target_bytes，且正好停在一个特殊 token 之后。"""
    written = 0
    tail = b''
    with open(src_path, 'rb') as fin, open(dst_path, 'wb') as fout:
        while written < target_bytes:
            chunk = fin.read(8 << 20)   # 8MB
            if not chunk:
                break
            buf = tail + chunk
            # 找到 buf 里最后一个完整 special_token 的位置
            last = buf.rfind(special_token)
            if last == -1:
                # 本块里没有特殊 token，整体缓存下来，等下一块
                tail = buf
                continue
            cut = last + len(special_token)
            fout.write(buf[:cut])
            written += cut
            tail = buf[cut:]
        # 处理尾部残留
        if tail:
            fout.write(tail)
            written += len(tail)
    print(f"written = {written / 1024**2:.1f} MB")
    return written


if __name__ == '__main__':
    src = r'd:\cs336-2026\assignments\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt'
    dst = r'd:\cs336-2026\assignments\assignment1-basics\data\tinystories_500M.txt'
    extract_sample(src, dst, 500 * 1024**2, b'<|endoftext|>')