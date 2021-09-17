"""
Official code of "Keisuke Izumiya and Edgar Simo-Serra, Inventory Management with Attention-Based Meta Actions, IEEE Conference on Games (CoG), 2021."
    Copyright (C) 2021 Keisuke Izumiya

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn
import torch.nn.functional as F

from nle.nethack import NUM_CHARS


PAD_CHAR = 0
INF = 1e30


class MyEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, use_index_select=True, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

        self._use_index_select = use_index_select

    def forward(self, x):
        if self._use_index_select:
            out = self.weight.index_select(0, x.view(-1))
            return out.view(x.shape + (-1,))
        else:
            return super().forward(x)


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self._num_classes)


class ToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float()


def make_char_emb(emb_dim=None, emb_kind="emb", use_index_select=True):
    if emb_kind == "emb":
        return MyEmbedding(NUM_CHARS, emb_dim, use_index_select, padding_idx=PAD_CHAR)
    elif emb_kind == "onehot":
        return nn.Sequential(OneHot(NUM_CHARS), ToFloat())
    else:
        raise ValueError(f"emb_kind == {emb_kind}")
