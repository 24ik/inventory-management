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

from nle.agent.models.model import BaseNet
from nle.agent.polybeast_env import create_env
from nle.agent.polyhydra import get_common_flags, get_environment_flags
from nle.nethack import INV_SIZE


def create_model(flags, device):
    env = create_env(get_environment_flags(get_common_flags(flags)))
    num_action = env.action_space.n
    del env

    num_virtual_action = num_action
    if flags.model.inv.model != "none" and flags.model.policy.model == "meta":
        num_virtual_action -= INV_SIZE[0]  # not add item-use action directly
        num_virtual_action += 1  # meta-action "use item"
    model = BaseNet(flags, num_action, num_virtual_action)

    return model.to(device)
