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

import mmle.nn as mnn
import torch
import torch.nn as nn

from nle.agent.models.util import make_char_emb
from nle.nethack import NUM_CHARS


class TxtCnn(nn.Module):
    def __init__(self, emb_dim, emb_kind, hidden_dim, cnn_depth, input_len, use_index_select):
        super().__init__()
        assert cnn_depth >= 5
        assert input_len >= 8

        # emb
        self._emb = make_char_emb(emb_dim, emb_kind, use_index_select)

        # cnn
        if emb_kind == "emb":
            conv1_input_channel = emb_dim
        elif emb_kind == "onehot":
            conv1_input_channel = NUM_CHARS
        else:
            raise ValueError(f"emb_kind == {emb_kind}")

        self._convs = nn.ModuleList(
            [
                mnn.Conv1d(conv1_input_channel, hidden_dim, 7, padding=3, bn=False),
                mnn.Conv1d(hidden_dim, hidden_dim, stride=2, bn=False),
                mnn.Conv1d(hidden_dim, hidden_dim, 7, padding=3, bn=False),
                mnn.Conv1d(hidden_dim, hidden_dim, stride=2, bn=False),
            ]
        )
        for _ in range(cnn_depth - 5):
            self._convs.append(mnn.Conv1d(hidden_dim, hidden_dim, bn=False))
        self._convs.append(mnn.Conv1d(hidden_dim, hidden_dim, stride=2, bn=False))

        # fc
        self._fc = mnn.FC(input_len // 8 * hidden_dim, hidden_dim, bn=False)

    def forward(self, x):
        """
        (B, L) -> (B, H)
        """

        x = self._emb(x).transpose(1, 2)  # (B, E, L)

        for conv in self._convs:
            x = conv(x)  # (B, H, L')

        return self._fc(x.flatten(1))  # (B, H)


class TxtRnn(nn.Module):
    def __init__(self, model_kind, emb_dim, emb_kind, hidden_dim, use_index_select):
        super().__init__()
        assert hidden_dim % 2 == 0

        self._hidden_dim = hidden_dim

        self._emb = make_char_emb(emb_dim, emb_kind, use_index_select)

        if model_kind == "gru":
            model_cls = nn.GRU
        elif model_kind == "lstm":
            model_cls = nn.LSTM
        else:
            raise ValueError(f"model_kind == {model_kind}")

        self._rnn = model_cls(emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        """
        (B, L) -> (B, H)
        """

        x = self._emb(x)  # (B, L, E)
        x, _ = self._rnn(x)  # (B, L, H)

        fwd_rep = x[:, -1, : self._hidden_dim // 2]  # (B, H/2)
        bwd_rep = x[:, 0, self._hidden_dim // 2 :]  # (B, H/2)

        return torch.cat([fwd_rep, bwd_rep], dim=1)  # (B, H)
