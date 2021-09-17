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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

import mmle.nn as mnn
import torch
from torch import nn
from torch.nn import functional as F

from nle.agent.models.action import make_action_model
from nle.agent.models.blstats import make_blstats_model
from nle.agent.models.embed import make_field_glyph_emb
from nle.agent.models.field import make_field_models
from nle.agent.models.inv import make_inv_models
from nle.agent.models.msg import make_msg_model
from nle.agent.models.policy import make_policy_model
from nle.agent.models.util import INF


class NetHackNet(nn.Module):
    AgentOutput = collections.namedtuple(
        "AgentOutput", "action policy_log_prob baseline virtual_log_prob item_log_prob"
    )

    def __init__(self):
        super(NetHackNet, self).__init__()

        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

    def forward(self, inputs, core_state):
        raise NotImplementedError

    def initial_state(self, batch_size=1):
        return ()

    def prepare_input(self, inputs):
        # -- [T x B x H x W]
        glyphs = inputs["glyphs"]

        # -- [T x B x F]
        features = inputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        features = features.view(T * B, -1).float()

        return glyphs, features

    def embed_state(self, inputs):
        raise NotImplementedError

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)


class BaseNet(NetHackNet):
    def __init__(self, flags, num_action, num_virtual_action):
        super().__init__()

        H = flags.model.hidden_dim

        self.register_buffer("_inf", torch.tensor(INF))

        self._field_glyph_emb = make_field_glyph_emb(flags)
        (
            self._full_field_model,
            self._crop_field_model,
            self._crop,
            field_out_dim,
        ) = make_field_models(flags)
        self._blstats_model, blstats_out_dim = make_blstats_model(flags)
        self._action_model, action_out_dim = make_action_model(flags, num_action)
        self._msg_model, msg_out_dim = make_msg_model(flags)
        self._inv_model, self._inv_attn, inv_out_dim = make_inv_models(flags)
        self._policy_model = make_policy_model(flags, num_action, num_virtual_action)

        self._mlp = nn.Sequential(
            mnn.FC(
                field_out_dim + blstats_out_dim + action_out_dim + msg_out_dim + inv_out_dim,
                H,
                bn=False,
            ),
            mnn.FC(H, H, bn=False),
        )

        if flags.model.rnn == "none":
            self._rnn = None
        elif flags.model.rnn == "gru":
            self._rnn = nn.GRU(H, H)
            self._make_initial_state = lambda b: torch.zeros(1, b, H)
            self._mask_core_state = lambda cs, nd: cs * nd[None, :, None]
        elif flags.model.rnn == "lstm":
            self._rnn = nn.LSTM(H, H)
            self._make_initial_state = lambda b: (torch.zeros(1, b, H), torch.zeros(1, b, H))
            self._mask_core_state = lambda cs, nd: (
                cs[0] * nd[None, :, None],
                cs[1] * nd[None, :, None],
            )
        else:
            raise ValueError(f"model.rnn == {flags.model.rnn}")

        self._baseline_mlp = mnn.FC(H, 1, bn=False, activ="id")

    def forward(self, inputs, core_state, greedy=False):
        T, B, *_ = inputs["glyphs"].shape

        glyphs = self._field_glyph_emb.prepare_input(inputs)
        blstats = inputs["blstats"].flatten(0, 1).float()
        coordinates = blstats[:, :2]

        reps = []
        if self._full_field_model:
            reps.append(self._full_field_model(self._field_glyph_emb(glyphs).transpose(1, 3)))
        if self._crop_field_model:
            crop = self._field_glyph_emb.GlyphTuple(*(self._crop(g, coordinates) for g in glyphs))
            reps.append(self._crop_field_model(self._field_glyph_emb(crop).transpose(1, 3)))
        if self._blstats_model:
            reps.append(self._blstats_model(blstats))
        if self._action_model:
            reps.append(self._action_model(inputs["last_action"].flatten(0, 1)))
        if self._msg_model:
            reps.append(self._msg_model(inputs["message"].long().flatten(0, 1)))
        if self._inv_model:
            inv_glyphs, inv_letters, inv_oclasses, inv_strs = None, None, None, None
            if "inv_glyphs" in inputs:
                inv_glyphs = inputs["inv_glyphs"]  # flattened in inv-model
            if "inv_letters" in inputs:
                inv_letters = inputs["inv_letters"].long().flatten(0, 1)
            if "inv_oclasses" in inputs:
                inv_oclasses = inputs["inv_oclasses"].long().flatten(0, 1)
            if "inv_strs" in inputs:
                inv_strs = inputs["inv_strs"].long().flatten(0, 1)

            inv_emb, items_emb, invalid_mask = self._inv_model(
                inv_glyphs, inv_letters, inv_oclasses, inv_strs
            )
            reps.append(inv_emb)

        feature = self._mlp(torch.cat(reps, dim=1))

        if self._rnn:
            rnn_outputs = []

            for input, not_done in zip(feature.view(T, B, -1).unbind(), (~inputs["done"]).unbind()):
                core_state = self._mask_core_state(core_state, not_done)
                output, core_state = self._rnn(input[None, ...], core_state)
                rnn_outputs.append(output)

            feature = torch.cat(rnn_outputs, dim=0).flatten(0, 1)

        score = None
        if self._inv_attn:
            score = ~invalid_mask * self._inv_attn(feature, items_emb) - invalid_mask * self._inf

        policy_log_prob, virtual_log_prob, item_log_prob = self._policy_model(feature, score)
        baseline = self._baseline_mlp(feature)

        if greedy:
            action = policy_log_prob.argmax(1)
        else:
            action = torch.multinomial(policy_log_prob.exp(), 1)

        policy_log_prob = policy_log_prob.view(T, B, -1)
        virtual_log_prob = virtual_log_prob.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        device = action.device
        if item_log_prob is None:
            item_log_prob = torch.empty(T, B, 1, device=device)
        else:
            item_log_prob = item_log_prob.view(T, B, -1)

        return (
            dict(
                policy_log_prob=policy_log_prob,
                virtual_log_prob=virtual_log_prob,
                item_log_prob=item_log_prob,
                baseline=baseline,
                action=action,
            ),
            core_state,
        )

    def initial_state(self, batch_size=1):
        return self._make_initial_state(batch_size)
