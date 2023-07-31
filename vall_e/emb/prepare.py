#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


import random
from tqdm import tqdm
from collections import defaultdict

from multiprocessing import Pool, cpu_count, Process, Queue
from multiprocessing.pool import ThreadPool


def default_dict():
    return None

def make_dict(wav_line):
    #make file to path dict
    trans_dict = defaultdict(default_dict)

    for wp in wav_line:
        trans_dict[wp.stem] = wp.parent

    return trans_dict

def process_aishell(line,trans_dict):

    #find transcript
    segs = line.split(" ")

    filename = segs[0]
    norm_txt = " ".join(segs[1:]).strip().removesuffix("\n")

    #write normalize.txt
    norm_txt_path = trans_dict[segs[0]]     
    if norm_txt_path is None:
        print(f"no transcript for {filename}")
    else:
        norm_txt_path = norm_txt_path / f"{filename}.normalized.txt"

        #print(f"write transcript for {filename} to {norm_txt_path} {norm_txt}")
        with open(norm_txt_path, "w") as f:
            f.write(norm_txt)





def load_trans(fl):
    trans = [] 
    for f in fl:
        with open(f, "r") as f:
            trans = f.readlines()
    return trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default=".wav")
    parser.add_argument("--worker-size", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="aishell")
    parser.add_argument("folder", type=Path)

    args = parser.parse_args()


    trans_path = args.folder / "transcript"
    trans_file = list(trans_path.rglob(f"*.txt"))
    wav_file = list(args.folder.rglob(f"*{args.suffix}"))

    trans = load_trans(trans_file)

    pool = ThreadPool(args.worker_size)


    trans_dict = make_dict(wav_file)
    for l in tqdm(trans):
        if args.dataset == 'aishell':
            pool.apply(process_aishell,(l,trans_dict,))


    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
