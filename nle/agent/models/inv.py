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
import torch.nn.functional as F

from nle.agent.models.embed import make_inv_glyph_emb
from nle.agent.models.txt import TxtCnn, TxtRnn
from nle.agent.models.util import MyEmbedding, INF
from nle.nethack import MAX_GLYPH, MAXOCLASSES, INV_SIZE, INV_STRS_SHAPE, NUM_CHARS


class _ItemEmbDims:
    def __init__(self, emb_dim, glyphs=0, letters=0, oclasses=0, strs=0):
        assert glyphs + letters + oclasses + strs > 0
        assert emb_dim % (glyphs + letters + oclasses + strs) == 0

        self.sum = emb_dim

        scale = emb_dim // (glyphs + letters + oclasses + strs)
        self.glyphs = scale * glyphs
        self.letters = scale * letters
        self.oclasses = scale * oclasses
        self.strs = scale * strs


class InvBaseModel(nn.Module):
    def __init__(
        self, str_model, inv_glyphs_emb, item_emb_dims, hidden_dim, use_index_select
    ):
        super().__init__()

        self.register_buffer("_inf", torch.tensor(INF))

        # each item embedding
        self._glyphs_emb = None
        self._letters_emb = None
        self._oclasses_emb = None
        self._strs_emb = None
        if item_emb_dims.glyphs > 0:
            self._glyphs_emb = inv_glyphs_emb
        if item_emb_dims.letters > 0:
            self._letters_emb = MyEmbedding(NUM_CHARS, item_emb_dims.letters, use_index_select)
        if item_emb_dims.oclasses > 0:
            self._oclasses_emb = MyEmbedding(
                MAXOCLASSES + 1, item_emb_dims.oclasses, use_index_select
            )
        if item_emb_dims.strs > 0:
            self._strs_emb = str_model

        # each item feature
        self._fc_item = mnn.FC(item_emb_dims.sum, item_emb_dims.sum, bn=False)

        # inventory feature
        self._fc_inv = mnn.FC(item_emb_dims.sum, hidden_dim, bn=False)

        # invalid item mask
        if item_emb_dims.letters > 0:
            self.register_buffer("_invalid_val", torch.tensor(NUM_CHARS - 1))
            self._invalid_mask_maker = lambda g, l, o, s: l == self._invalid_val
        elif item_emb_dims.oclasses > 0:
            self.register_buffer("_invalid_val", torch.tensor(MAXOCLASSES))
            self._invalid_mask_maker = lambda g, l, o, s: o == self._invalid_val
        elif item_emb_dims.glyphs > 0:
            self.register_buffer("_invalid_val", torch.tensor(MAX_GLYPH))
            self._invalid_mask_maker = lambda g, l, o, s: g.flatten(0, 1) == self._invalid_val
        elif item_emb_dims.strs > 0:
            self.register_buffer("_invalid_val", torch.tensor(0))
            self._invalid_mask_maker = lambda g, l, o, s: s.sum(-1) == self._invalid_val
        else:
            raise ValueError("Impossible path.")

    def forward(self, inv_glyphs, inv_letters, inv_oclasses, inv_strs):
        """
        (T, B', N), (B, N), (B, N), (B, N, L) -> (B, H), (B, N, H'), (B, N)
        """

        # calc invalid item mask
        invalid_mask = self._invalid_mask_maker(
            inv_glyphs, inv_letters, inv_oclasses, inv_strs
        )  # (B, N)
        expanded_invalid_mask = invalid_mask[:, :, None]  # (B, N, 1)

        # calc each item feature
        embs = []
        if self._glyphs_emb:
            embs.append(
                self._glyphs_emb(self._glyphs_emb.prepare_input({"glyphs": inv_glyphs}))
            )  # (B, N, E1)
        if self._letters_emb:
            embs.append(self._letters_emb(inv_letters))  # (B, N, E2)
        if self._oclasses_emb:
            embs.append(self._oclasses_emb(inv_oclasses))  # (B, N, E3)
        if self._strs_emb:
            B, N, _ = inv_strs.shape
            embs.append(self._strs_emb(inv_strs.flatten(0, 1)).view(B, N, -1))  # (B, N, E4)

        items_emb = self._fc_item(torch.cat(embs, dim=-1)) * ~expanded_invalid_mask  # (B, N, H')

        # calc inventory feature
        inv_emb = self._fc_inv(items_emb.sum(1))  # (B, H)

        return inv_emb, items_emb, invalid_mask  # (B, H), (B, N, H'), (B, N)


def _make_inv_str_model(flags, strs_emb_dim):
    if strs_emb_dim == 0:
        return None

    inv = flags.model.inv
    txt = flags.model.txt

    model_kind = inv.str_model
    if model_kind == "cnn":
        return TxtCnn(
            txt.emb_dim,
            txt.emb_kind,
            strs_emb_dim,
            inv.cnn.depth,
            INV_STRS_SHAPE[1],
            flags.model.use_index_select,
        )
    elif model_kind in ("gru", "lstm"):
        return TxtRnn(
            model_kind, txt.emb_dim, txt.emb_kind, strs_emb_dim, flags.model.use_index_select
        )
    else:
        raise ValueError(f"inv.strs.model == {model_kind}")


class InvAttn(nn.Module):
    def __init__(self, hidden_dim, item_emb_dim, key_dim):
        super().__init__()

        self.register_buffer("_normalizer", torch.tensor(key_dim ** -0.5))

        self._query_fc = mnn.FC(hidden_dim, key_dim, bias=False, bn=False)
        self._key_fc = mnn.FC(item_emb_dim, key_dim, bias=False, bn=False)
        self._val_fc = mnn.FC(item_emb_dim, 1, bias=False, bn=False)

    def forward(self, hidden, items_emb):
        """
        (B, H), (B, N, H') -> (B, N)
        """

        query = self._query_fc(hidden)  # (B, d_k)
        key = self._key_fc(items_emb)  # (B, N, d_k)
        val = self._val_fc(items_emb).squeeze(-1)  # (B, N)

        weight = torch.einsum("bk,bnk->bn", query, key) * self._normalizer  # (B, N)
        return weight * val  # (B, N)


def _make_inv_attn(flags):
    inv = flags.model.inv

    return InvAttn(flags.model.hidden_dim, inv.item_emb_dim, inv.attn.key_dim)


def make_inv_models(flags):
    inv = flags.model.inv
    emb_ratio = inv.emb_ratio

    item_emb_dims = _ItemEmbDims(
        inv.item_emb_dim, emb_ratio.glyphs, emb_ratio.letters, emb_ratio.oclasses, emb_ratio.strs
    )

    str_model = _make_inv_str_model(flags, item_emb_dims.strs)

    inv_attn = _make_inv_attn(flags)

    inv_glyphs_emb = make_inv_glyph_emb(flags, item_emb_dims.glyphs)

    model_kind = inv.model
    if model_kind == "none":
        return None, None, 0
    elif model_kind == "baseline":
        return (
            InvBaseModel(
                str_model,
                inv_glyphs_emb,
                item_emb_dims,
                inv.hidden_dim,
                flags.model.use_index_select,
            ),
            inv_attn,
            inv.hidden_dim,
        )
    else:
        raise ValueError(f"inv.model == {model_kind}")
