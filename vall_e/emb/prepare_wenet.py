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

def process_wenet(line,trans_dict):

    #find transcript
    segs = line.split("\t")

    filename = segs[0]
    norm_txt = " ".join(segs[1:]).strip().removesuffix("\n")

    #write normalize.txt
    norm_txt_path = trans_dict[segs[0].split("_")[0]]     
    norm_txt_path = Path(norm_txt_path)
    if norm_txt_path is None:
        print(f"no transcript for {filename}")
    else:
        norm_txt_path = norm_txt_path.parent / f"{filename}.normalized.txt"

        print(f"write transcript for {filename} to {norm_txt_path} {norm_txt}")
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
    parser.add_argument("folder", type=Path)

    args = parser.parse_args()

    #path data
    scp_paths = [*args.folder.rglob(f"wav.scp")]

    path_data = {}
    for path in scp_paths:
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                segs = l.strip().split("\t")
                path_data[segs[0]] = segs[1]

    trans_file = list(args.folder.rglob(f"text"))

    trans = load_trans(trans_file)

    pool = ThreadPool(args.worker_size)


    for l in tqdm(trans):
        process_wenet(l,path_data)
#        pool.apply(process_wenet,(l,path_data,))


    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
