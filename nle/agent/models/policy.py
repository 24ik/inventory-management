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


class BasePolicy(nn.Module):
    def __init__(self, num_action, hidden_dim):
        super().__init__()

        self._fc = mnn.FC(
            hidden_dim, num_action, bn=False, activ="log_softmax", activ_kwargs={"dim": -1}
        )

    def forward(self, feature, item_score):
        """
        (B, H), _ -> (B, A)
        """

        log_prob = self._fc(feature)  # (B, A)

        return log_prob, log_prob, None  # (B, A), (B, A), _


class MetaPolicy(nn.Module):
    def __init__(self, num_virtual_action, hidden_dim):
        super().__init__()

        self._fc = mnn.FC(
            hidden_dim, num_virtual_action, bn=False, activ="log_softmax", activ_kwargs={"dim": -1}
        )

    def forward(self, feature, item_score):
        """
        (B, H), (B, N) -> (B, A)
        """

        virtual_log_prob = self._fc(feature)  # (B, a+1)
        item_log_prob = F.log_softmax(item_score, dim=1)  # (B, N)
        policy_log_prob = torch.cat(
            [
                virtual_log_prob[:, :-1],
                virtual_log_prob[:, -1:] + item_log_prob,
            ],
            dim=-1,
        )  # (B, A)

        return policy_log_prob, virtual_log_prob, item_log_prob  # (B, A), (B, a+1), (B, N)


def make_policy_model(flags, num_action, num_virtual_action):
    model_kind = flags.model.policy.model
    if model_kind == "baseline":
        return BasePolicy(num_action, flags.model.hidden_dim)
    elif model_kind == "meta":
        return MetaPolicy(num_virtual_action, flags.model.hidden_dim)
    else:
        raise ValueError(f"policy.model == {model_kind}")
