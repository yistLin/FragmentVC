"""Dataset for reconstruction scheme."""

import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class IntraSpeakerDataset(Dataset):
    """Dataset for reconstruction scheme.

    Returns:
        speaker_id: speaker id number.
        feat: Wav2Vec feature tensor.
        mel: log mel spectrogram tensor.
    """

    def __init__(self, data_dir, metadata_path, n_samples=5, pre_load=False):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        executor = ThreadPoolExecutor(max_workers=4)
        futures = []

        for speaker_name, utterances in metadata.items():
            for utterance in utterances:
                futures.append(
                    executor.submit(
                        _process_data,
                        speaker_name,
                        data_dir,
                        utterance["feature_path"],
                        pre_load,
                    )
                )

        self.data = []
        self.speaker_to_indices = {}
        for i, future in enumerate(tqdm(futures, ncols=0)):
            result = future.result()
            speaker_name = result[0]
            self.data.append(result)
            if speaker_name not in self.speaker_to_indices:
                self.speaker_to_indices[speaker_name] = [i]
            else:
                self.speaker_to_indices[speaker_name].append(i)

        self.data_dir = Path(data_dir)
        self.n_samples = n_samples
        self.pre_load = pre_load

    def __len__(self):
        return len(self.data)

    def _get_data(self, index):
        if self.pre_load:
            speaker_name, content_emb, target_mel = self.data[index]
        else:
            speaker_name, content_emb, target_mel = _load_data(*self.data[index])
        return speaker_name, content_emb, target_mel

    def __getitem__(self, index):
        speaker_name, content_emb, target_mel = self._get_data(index)
        utterance_indices = self.speaker_to_indices[speaker_name].copy()
        utterance_indices.remove(index)

        sampled_mels = []
        for sampled_id in random.sample(utterance_indices, self.n_samples):
            sampled_mel = self._get_data(sampled_id)[2]
            sampled_mels.append(sampled_mel)

        reference_mels = torch.cat(sampled_mels, dim=0)

        return content_emb, reference_mels, target_mel


def _process_data(speaker_name, data_dir, feature_path, load):
    if load:
        return _load_data(speaker_name, data_dir, feature_path)
    else:
        return speaker_name, data_dir, feature_path


def _load_data(speaker_name, data_dir, feature_path):
    feature = torch.load(Path(data_dir, feature_path))
    content_emb = feature["feat"]
    target_mel = feature["mel"]

    return speaker_name, content_emb, target_mel


def collate_batch(batch):
    """Collate a batch of data."""
    srcs, refs, tgts = zip(*batch)

    src_lens = [len(src) for src in srcs]
    ref_lens = [len(ref) for ref in refs]
    tgt_lens = [len(tgt) for tgt in tgts]
    overlap_lens = [
        min(src_len, tgt_len) for src_len, tgt_len in zip(src_lens, tgt_lens)
    ]

    srcs = pad_sequence(srcs, batch_first=True)  # (batch, max_src_len, wav2vec_dim)

    src_masks = [torch.arange(srcs.size(1)) >= src_len for src_len in src_lens]
    src_masks = torch.stack(src_masks)  # (batch, max_src_len)

    refs = pad_sequence(refs, batch_first=True, padding_value=-20)
    refs = refs.transpose(1, 2)  # (batch, mel_dim, max_ref_len)

    ref_masks = [torch.arange(refs.size(2)) >= ref_len for ref_len in ref_lens]
    ref_masks = torch.stack(ref_masks)  # (batch, max_ref_len)

    tgts = pad_sequence(tgts, batch_first=True, padding_value=-20)
    tgts = tgts.transpose(1, 2)  # (batch, mel_dim, max_tgt_len)

    tgt_masks = [torch.arange(tgts.size(2)) >= tgt_len for tgt_len in tgt_lens]
    tgt_masks = torch.stack(tgt_masks)  # (batch, max_tgt_len)

    return srcs, src_masks, refs, ref_masks, tgts, tgt_masks, overlap_lens
