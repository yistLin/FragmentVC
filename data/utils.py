"""Utilities for data manipulation."""

from typing import Union
from pathlib import Path

import librosa
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import lfilter

matplotlib.use("Agg")


def load_wav(
    audio_path: Union[str, Path], sample_rate: int, trim: bool = False
) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]

    return wav


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T


def plot_mel(gt_mel, predicted_mel=None, filename="mel.png"):
    if predicted_mel is not None:
        fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))

    axes[0][0].imshow(gt_mel.detach().cpu().numpy().T, origin="lower")
    axes[0][0].set_aspect(1, adjustable="box")
    axes[0][0].set_ylim(1.0, 80)
    axes[0][0].set_title("ground-truth mel-spectrogram", fontsize="medium")
    axes[0][0].tick_params(labelsize="x-small", left=False, labelleft=False)

    if predicted_mel is not None:
        axes[1][0].imshow(predicted_mel.detach().cpu().numpy(), origin="lower")
        axes[1][0].set_aspect(1.0, adjustable="box")
        axes[1][0].set_ylim(0, 80)
        axes[1][0].set_title("predicted mel-spectrogram", fontsize="medium")
        axes[1][0].tick_params(labelsize="x-small", left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_attn(attn, filename="attn.png"):
    fig, axes = plt.subplots(len(attn), 1, squeeze=False, figsize=(10, 10))

    for i, layer_attn in enumerate(attn):
        axes[i][0].imshow(attn[i][0].detach().cpu().numpy(), origin="lower")
        axes[i][0].set_title("layer {}".format(i), fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small")
        axes[i][0].set_xlabel("target")
        axes[i][0].set_ylabel("source")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
