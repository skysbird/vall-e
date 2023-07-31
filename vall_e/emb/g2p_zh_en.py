import argparse
import random
import string
from functools import cache
from pathlib import Path

import torch
from g2p_zh_en import G2P
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process, Queue
import torch.multiprocessing as mp

@cache
def _get_model(language:str):
    return G2P(language=language)


@cache
def _get_graphs(path):
    with open(path, "r") as f:
        graphs = f.read()
    return graphs


def encode(graphs: str, language:str) -> list[str]:
    g2p = _get_model(language)
    phones = g2p(graphs)
    ignored = {" ", *string.punctuation}
    return ["_" if p in ignored else p for p in phones]

import time


def process_g2p(path, phone_path,language):
    graphs = _get_graphs(path)
    #start_time = time.time()

    phones = encode(graphs,language)
    with open(phone_path, "w") as f:
        f.write(" ".join(phones))

    #end_time = time.time()
    #print("耗时: {:.5f}豪秒".format((end_time - start_time)*1000))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default=".normalized.txt")
    parser.add_argument("--worker-size", type=int)
    parser.add_argument("--language", type=str, default='zh-cn')
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    paths = list(args.folder.rglob(f"*{args.suffix}"))
    random.shuffle(paths)


    pool = Pool(processes=args.worker_size)

    for path in tqdm(paths):
        phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
        if phone_path.exists():
            continue
        pool.apply(process_g2p,(path,phone_path,args.language,))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
