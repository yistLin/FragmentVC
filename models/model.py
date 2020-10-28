"""FragmentVC model architecture."""

from typing import Tuple, List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convolutional_transformer import Smoother, Extractor


class FragmentVC(nn.Module):
    """
    FragmentVC uses Wav2Vec feature of the source speaker to query and attend
    on mel spectrogram of the target speaker.
    """

    def __init__(self, d_model=512):
        super().__init__()

        self.unet = UnetBlock(d_model)

        self.smoothers = nn.TransformerEncoder(Smoother(d_model, 2, 1024), num_layers=3)

        self.mel_linear = nn.Linear(d_model, 80)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # out: (src_len, batch, d_model)
        out, attns = self.unet(srcs, refs, src_masks=src_masks, ref_masks=ref_masks)

        # out: (src_len, batch, d_model)
        out = self.smoothers(out, src_key_padding_mask=src_masks)

        # out: (src_len, batch, 80)
        out = self.mel_linear(out)

        # out: (batch, 80, src_len)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out = out + refined

        # out: (batch, 80, src_len)
        return out, attns


class UnetBlock(nn.Module):
    """Hierarchically attend on references."""

    def __init__(self, d_model: int):
        super(UnetBlock, self).__init__()

        self.conv1 = nn.Conv1d(80, d_model, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")

        self.prenet = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, d_model),
        )

        self.extractor1 = Extractor(d_model, 2, 1024, no_residual=True)
        self.extractor2 = Extractor(d_model, 2, 1024)
        self.extractor3 = Extractor(d_model, 2, 1024)

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # tgt: (batch, tgt_len, d_model)
        tgt = self.prenet(srcs)
        # tgt: (tgt_len, batch, d_model)
        tgt = tgt.transpose(0, 1)

        # ref*: (batch, d_model, mel_len)
        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        # out*: (tgt_len, batch, d_model)
        out, attn1 = self.extractor1(
            tgt,
            ref3.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn2 = self.extractor2(
            out,
            ref2.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn3 = self.extractor3(
            out,
            ref1.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )

        # out: (tgt_len, batch, d_model)
        return out, [attn1, attn2, attn3]
