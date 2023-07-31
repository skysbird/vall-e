#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


import random
from tqdm import tqdm


def process_aishell(path):
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default=".wav")
    parser.add_argument("--worker-size", type=int)
    parser.add_argument("--dataset", type=str, default="aishell")
    parser.add_argument("folder", type=Path)

    args = parser.parse_args()

    paths = list(args.folder.rglob(f"*{args.suffix}"))

    for path in tqdm(paths):
        if args.dataset == 'aishell':
            process_aishell(path)


if __name__ == "__main__":
    main()
