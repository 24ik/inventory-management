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
