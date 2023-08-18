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
from multiprocessing.pool import ThreadPool

from vall_e.config import cfg


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


def _replace_file_extension(path, name, suffix):
    return (path.parent / name).with_suffix(suffix)


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


def encode_from_file(wav,sr, start,end,device="cuda"):
    #offset = 0L,
    #duration = -1L,
    #unit = c("samples", "time")
    sampleRate = 48000
    start = int(sampleRate * start)
    end = int(sampleRate * end)

    #print(wav.size())
    import time

    start_time = time.time()

    #print(f"cost {time.time() - start_time}")
    #, offset = start, duration = (end-start), unit="time")

    if wav.shape[0] == 2:
        wav = wav[:1]
    wav = wav[...,start:end]
    return encode(wav, sr, device)


def getimgpath_process(path, img_path_queue):
    for img_name in os.listdir(root_path):
        img_path = osp.join(root_path, img_name)
        img_path_queue.put(img_path)



img_path_queue = Queue()




def process_qnt(wav,sr,out_path,start,end):
    try:
        #gpu_id = rank %2 
        torch.cuda.set_device(0)
        qnt = encode_from_file(wav,sr,start,end)
        
        #qqq = rearrange(qnt, "1 q t -> t q")
        #decode_to_file(qqq, "/home/jovyan/work/sky/vall-e-en/test.wav")
        torch.save(qnt.cpu(), out_path)
    except Exception as e:
        print(e)
        #print(f"path={path}")

def batch(path_data, name,wav_list,i):

    path = path_data[name]

    wav, sr = torchaudio.load(str(path))
    print(i)
    #'utt_id': 'POD1000000005_S0000000', 'name': 'POD1000000005', 'start': 0.0, 'end': 4.859}
    for v in wav_list:
        print(v)
        path = Path(path)
        start = v['start']
        end = v['end']


        out_path = _replace_file_extension(path, v['utt_id'], ".qnt.pt")
        print(out_path)
        if out_path.exists():
            continue

        process_qnt(wav,sr,out_path,start,end)
    #pool.apply(process_qnt,(path,out_path,start,end,i))



def main():
    #mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-size", type=int)
    #parser.add_argument("--suffix", default=".wav")
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

    print(path_data)

    #wav data
    paths = [*args.folder.rglob(f"segments")]
    wav_data = {}

    for path in paths:
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                segs = l.strip().split("\t")
                d = {
                    'utt_id':segs[0],
                    'name':segs[1],
                    'start':float(segs[2]),
                    'end':float(segs[3])
                    }

                if wav_data.__contains__(segs[1]):
                    wav_data[segs[1]].append(d)
                else:
                    wav_data[segs[1]] = [d]

    pool = ThreadPool(args.worker_size)

    i = 0
    for (k,v) in tqdm(wav_data.items()):

        #batch(path_data,k,v)
        pool.apply_async(batch,(path_data,k,v,i))
        i = i+1

        #break

            


    #i = 0
    #for path in tqdm(paths):
    #    out_path = _replace_file_extension(path, ".qnt.pt")
    #    if out_path.exists():
    #        continue

    #    pool.apply(process_qnt,(path,out_path,i))
    #    i = i+1

    pool.close()
    pool.join()



if __name__ == "__main__":
    main()

