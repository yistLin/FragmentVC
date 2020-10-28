#!/usr/bin/env python3
"""Precompute Wav2Vec features."""

import os
import json
from pathlib import Path
from tempfile import mkstemp
from multiprocessing import cpu_count

import tqdm
import torch
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, ActionConfigFile

from models import load_pretrained_wav2vec
from data import PreprocessDataset


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("wav2vec_path", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--trim_method", choices=["librosa", "vad"], default="vad")
    parser.add_argument("--n_workers", type=int, default=cpu_count())

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
    data_dirs,
    wav2vec_path,
    out_dir,
    trim_method,
    n_workers,
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

    out_dir_path = Path(out_dir)

    if out_dir_path.exists():
        assert out_dir_path.is_dir()
    else:
        out_dir_path.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PreprocessDataset(
        data_dirs,
        trim_method,
        sample_rate,
        preemph,
        hop_len,
        win_len,
        n_fft,
        n_mels,
        f_min,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=n_workers
    )

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)

    speaker_infos = {}

    pbar = tqdm.tqdm(total=len(dataset), ncols=0)

    for speaker_name, audio_path, wav, mel in dataloader:
        if wav.size(-1) < 10:
            continue

        wav = wav.to(device)
        speaker_name = speaker_name[0]
        audio_path = audio_path[0]

        with torch.no_grad():
            feat = wav2vec.extract_features(wav, None)[0]
            feat = feat.detach().cpu().squeeze(0)
            mel = mel.squeeze(0)

        fd, temp_file = mkstemp(suffix=".tar", prefix="utterance-", dir=out_dir_path)
        torch.save({"feat": feat, "mel": mel}, temp_file)
        os.close(fd)

        if speaker_name not in speaker_infos.keys():
            speaker_infos[speaker_name] = []

        speaker_infos[speaker_name].append(
            {
                "feature_path": Path(temp_file).name,
                "audio_path": audio_path,
                "feat_len": len(feat),
                "mel_len": len(mel),
            }
        )

        pbar.update(dataloader.batch_size)

    with open(out_dir_path / "metadata.json", "w") as f:
        json.dump(speaker_infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
