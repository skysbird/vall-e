import argparse
import random
from functools import cache
from pathlib import Path

import soundfile
import torch
import torchaudio
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Process, Queue
import torch.multiprocessing as mp

from ..config import cfg


@cache
def _load_model(device="cuda"):
    # Instantiate a pretrained EnCodec model
    assert cfg.sample_rate == 24_000
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    return model


def unload_model():
    return _load_model.cache_clear()


@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
    """
    Args:
        codes: (b q t)
    """
    assert codes.dim() == 3
    model = _load_model(device)
    return model.decode([(codes, None)]), model.sample_rate


def decode_to_file(resps: Tensor, path: Path):
    assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
    resps = rearrange(resps, "t q -> 1 q t")
    wavs, sr = decode(resps)
    soundfile.write(str(path), wavs.cpu()[0, 0], sr)


def _replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


@torch.inference_mode()
def encode(wav: Tensor, sr: int, device="cuda"):
    """
    Args:
        wav: (t)
        sr: int
    """
    model = _load_model(device)
    wav = wav.unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    encoded_frames = model.encode(wav)
    qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
    return qnt


def encode_from_file(path, device="cuda"):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 2:
        wav = wav[:1]
    return encode(wav, sr, device)


def getimgpath_process(path, img_path_queue):
    for img_name in os.listdir(root_path):
        img_path = osp.join(root_path, img_name)
        img_path_queue.put(img_path)



img_path_queue = Queue()




def process_qnt(path,out_path,rank):
    gpu_id = rank %2 
    torch.cuda.set_device(gpu_id)
    qnt = encode_from_file(path)
    torch.save(qnt.cpu(), out_path)

def main():
    #mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-size", type=int)
    parser.add_argument("--suffix", default=".wav")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    paths = [*args.folder.rglob(f"*{args.suffix}")]
    random.shuffle(paths)


    pool = Pool(processes=args.worker_size)

    i = 0
    for path in tqdm(paths):
        out_path = _replace_file_extension(path, ".qnt.pt")
        if out_path.exists():
            continue

        pool.apply(process_qnt,(path,out_path,i))
        i = i+1

    pool.close()
    pool.join()



if __name__ == "__main__":
    main()

