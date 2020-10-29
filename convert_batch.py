#!/usr/bin/env python3
"""Convert multiple pairs."""

import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import yaml
import torch
import numpy as np
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("info_path", type=str)
    parser.add_argument("output_dir", type=str, default=".")
    parser.add_argument("-c", "--ckpt_path", default="checkpoints/fragmentvc.pt")
    parser.add_argument("-w", "--wav2vec_path", default="checkpoints/wav2vec_small.pt")
    parser.add_argument("-v", "--vocoder_path", default="checkpoints/vocoder.pt")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=326)
    parser.add_argument("--win_len", type=int, default=1304)
    parser.add_argument("--n_fft", type=int, default=1304)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--audio_config", action=ActionConfigFile)

    return vars(parser.parse_args())


def main(
    info_path,
    output_dir,
    ckpt_path,
    wav2vec_path,
    vocoder_path,
    sample_rate,
    preemph,
    hop_len,
    win_len,
    n_fft,
    n_mels,
    f_min,
    **kwargs,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    print("[INFO] Wav2Vec is loaded from", wav2vec_path)

    model = torch.jit.load(ckpt_path).to(device).eval()
    print("[INFO] FragmentVC is loaded from", ckpt_path)

    vocoder = torch.jit.load(vocoder_path).to(device).eval()
    print("[INFO] Vocoder is loaded from", vocoder_path)

    path2wav = partial(load_wav, sample_rate=sample_rate)
    wav2mel = partial(
        log_mel_spectrogram,
        preemph=preemph,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        f_min=f_min,
    )

    with open(info_path) as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)

    out_mels = []
    attns = []

    for pair_name, pair in infos.items():
        src_wav = load_wav(pair["source"], sample_rate, trim=True)
        src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)

        with Pool(cpu_count()) as pool:
            tgt_wavs = pool.map(path2wav, pair["target"])
            tgt_mels = pool.map(wav2mel, tgt_wavs)

        tgt_mel = np.concatenate(tgt_mels, axis=0)
        tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(device)

        with torch.no_grad():
            src_feat = wav2vec.extract_features(src_wav, None)[0]

            out_mel, attn = model(src_feat, tgt_mel)
            out_mel = out_mel.transpose(1, 2).squeeze(0)

            out_mels.append(out_mel)
            attns.append(attn)

        print(f"[INFO] Pair {pair_name} converted")

    print("[INFO] Generating waveforms...")

    with torch.no_grad():
        out_wavs = vocoder.generate(out_mels)

    print("[INFO] Waveforms generated")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, out_mel, out_wav, attn in zip(
        infos.keys(), out_mels, out_wavs, attns
    ):
        out_wav = out_wav.cpu().numpy()
        out_path = Path(out_dir, pair_name)

        plot_mel(out_mel, filename=out_path.with_suffix(".mel.png"))
        plot_attn(attn, filename=out_path.with_suffix(".attn.png"))
        sf.write(out_path.with_suffix(".wav"), out_wav, sample_rate)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
