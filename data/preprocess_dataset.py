"""Precompute Wav2Vec features and spectrograms."""

from copy import deepcopy
from pathlib import Path

import torch
from librosa.util import find_files

import sox

from .utils import load_wav, log_mel_spectrogram


class PreprocessDataset(torch.utils.data.Dataset):
    """Prefetch audio data for preprocessing."""

    def __init__(
        self,
        data_dirs,
        trim_method,
        sample_rate,
        preemph,
        hop_len,
        win_len,
        n_fft,
        n_mels,
        f_min,
    ):

        data = []

        for data_dir in data_dirs:
            data_dir_path = Path(data_dir)
            speaker_dirs = [x for x in data_dir_path.iterdir() if x.is_dir()]

            for speaker_dir in speaker_dirs:
                audio_paths = find_files(speaker_dir)
                if len(audio_paths) == 0:
                    continue

                speaker_name = speaker_dir.name
                for audio_path in audio_paths:
                    data.append((speaker_name, audio_path))

        self.trim_method = trim_method
        self.sample_rate = sample_rate
        self.preemph = preemph
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.data = data

        if trim_method == "vad":
            tfm = sox.Transformer()
            tfm.vad(location=1)
            tfm.vad(location=-1)
            self.sox_transform = tfm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        speaker_name, audio_path = self.data[index]

        if self.trim_method == "librosa":
            wav = load_wav(audio_path, self.sample_rate, trim=True)
        elif self.trim_method == "vad":
            wav = load_wav(audio_path, self.sample_rate)
            trim_wav = self.sox_transform.build_array(
                input_array=wav, sample_rate_in=self.sample_rate
            )
            wav = deepcopy(trim_wav if len(trim_wav) > 10 else wav)

        mel = log_mel_spectrogram(
            wav,
            self.preemph,
            self.sample_rate,
            self.n_mels,
            self.n_fft,
            self.hop_len,
            self.win_len,
            self.f_min,
        )
        return speaker_name, audio_path, torch.FloatTensor(wav), torch.FloatTensor(mel)
