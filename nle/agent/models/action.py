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

from nle.agent.models.util import MyEmbedding


class ActionBaseModel(nn.Module):
    def __init__(self, num_action, emb_dim, use_index_select):
        super().__init__()

        self._emb = MyEmbedding(num_action, emb_dim, use_index_select)

    def forward(self, x):
        """
        (B, A) -> (B, H)
        """

        return self._emb(x)


def make_action_model(flags, num_action):
    model_kind = flags.model.action.model
    if model_kind == "none":
        return None, 0
    elif model_kind == "baseline":
        return (
            ActionBaseModel(num_action, flags.model.action.emb_dim, flags.model.use_index_select),
            flags.model.action.emb_dim,
        )
    else:
        raise ValueError(f"action.model == {model_kind}")
