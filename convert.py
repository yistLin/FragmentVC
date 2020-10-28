#!/usr/bin/env python3
"""Convert using one source utterance and multiple target utterances."""

import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

import sox

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("source_path", type=str)
    parser.add_argument("reference_paths", type=str, nargs="+")
    parser.add_argument("-w", "--wav2vec_path", type=str, required=True)
    parser.add_argument("-c", "--ckpt_path", type=str, required=True)
    parser.add_argument("-v", "--vocoder_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default="output.wav")

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
    source_path,
    reference_paths,
    wav2vec_path,
    ckpt_path,
    vocoder_path,
    output_path,
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

    begin_time = step_moment = datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    print("[INFO] Wav2Vec is loaded from", wav2vec_path)

    model = torch.jit.load(ckpt_path).to(device).eval()
    print("[INFO] FragmentVC is loaded from", ckpt_path)

    vocoder = torch.jit.load(vocoder_path).to(device).eval()
    print("[INFO] Vocoder is loaded from", vocoder_path)

    elaspe_time = datetime.now() - step_moment
    step_moment = datetime.now()
    print("[INFO] elasped time", elaspe_time.total_seconds())

    tfm = sox.Transformer()
    tfm.vad(location=1)
    tfm.vad(location=-1)

    src_wav = load_wav(source_path, sample_rate)
    src_wav = deepcopy(tfm.build_array(input_array=src_wav, sample_rate_in=sample_rate))
    src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)
    print("[INFO] source waveform shape:", src_wav.shape)

    ref_mels = []
    for ref_path in reference_paths:
        ref_wav = load_wav(ref_path, sample_rate)
        ref_wav = tfm.build_array(input_array=ref_wav, sample_rate_in=sample_rate)
        ref_wav = deepcopy(ref_wav)
        ref_mel = log_mel_spectrogram(
            ref_wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min
        )
        ref_mels.append(ref_mel)

    ref_mel = np.concatenate(ref_mels, axis=0)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    print("[INFO] target spectrograms shape:", ref_mel.shape)

    with torch.no_grad():
        src_feat = wav2vec.extract_features(src_wav, None)[0]
        print("[INFO] source Wav2Vec feature shape:", src_feat.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        out_mel, attns = model(src_feat, ref_mel)
        out_mel = out_mel.transpose(1, 2).squeeze(0)
        print("[INFO] converted spectrogram shape:", out_mel.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        out_wav = vocoder.generate([out_mel])[0]
        out_wav = out_wav.cpu().numpy()
        print("[INFO] generated waveform shape:", out_wav.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

    wav_path = Path(output_path)
    sf.write(wav_path, out_wav, sample_rate)
    print("[INFO] generated waveform is saved to", wav_path)

    mel_path = wav_path.with_suffix(".mel.png")
    plot_mel(out_mel, filename=mel_path)
    print("[INFO] mel-spectrogram plot is saved to", mel_path)

    attn_path = wav_path.with_suffix(".attn.png")
    plot_attn(attns, filename=attn_path)
    print("[INFO] attention plot is saved to", attn_path)

    elaspe_time = datetime.now() - begin_time
    print("[INFO] Overall elasped time", elaspe_time.total_seconds())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
