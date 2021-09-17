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

import torch
from torch.nn import functional as F


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(log_prob, coef=None):
    entropy_per_timestep = (-log_prob.exp() * log_prob).sum(-1)
    if coef is None:
        return -torch.sum(entropy_per_timestep)
    else:
        return -torch.sum(coef * entropy_per_timestep)


def compute_policy_gradient_loss(log_prob, actions, advantages):
    cross_entropy = F.nll_loss(
        log_prob.flatten(0, 1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach()
    return torch.sum(policy_gradient_loss_per_timestep)
