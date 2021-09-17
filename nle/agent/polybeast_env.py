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

#
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

import json
import multiprocessing as mp
import logging
import os
import signal
import sys
import threading
import time

import torch
import libtorchbeast

from nle.env.tasks import REDUCED_ACTIONS
from nle.agent.envs import tasks


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


# Helper functions for NethackEnv.
def _format_observation(obs):
    obs = torch.from_numpy(obs)
    return obs.view((1, 1) + obs.shape)  # (...) -> (T,B,...).


def create_folders(flags):
    # Creates some of the folders that would be created by the filewriter.
    logdir = os.path.join(flags.savedir, "archives")
    if not os.path.exists(logdir):
        logging.info("Creating archive directory: %s" % logdir)
        os.makedirs(logdir, exist_ok=True)
    else:
        logging.info("Found archive directory: %s" % logdir)


def create_env(flags, env_id=0, full_obs=False, lock=threading.Lock()):
    # commenting out these options for now because they use too much disk space
    # archivefile = "nethack.%i.%%(pid)i.%%(time)s.zip" % env_id
    # if flags.single_ttyrec and env_id != 0:
    #     archivefile = None

    # logdir = os.path.join(flags.savedir, "archives")

    with lock:
        env_class = tasks.ENVS[flags.env.name]
        kwargs = dict(
            savedir=None,
            archivefile=None,
            character=flags.env.character,
            max_episode_steps=flags.env.max_num_steps,
            penalty_step=flags.env.penalty_step,
            penalty_time=flags.env.penalty_time,
            penalty_mode=flags.env.fn_penalty_step,
        )
        if not full_obs:
            kwargs.update(
                observation_keys=(
                    "glyphs",
                    "chars",
                    "colors",
                    "specials",
                    "blstats",
                    "message",
                    "inv_glyphs",
                    "inv_letters",
                    "inv_oclasses",
                    "inv_strs",
                ),
            )
        if flags.env.name in ("staircase", "pet", "oracle"):
            kwargs.update(reward_win=flags.env.reward_win, reward_lose=flags.env.reward_lose)
        elif env_id == 0:  # print warning once
            print("Ignoring flags.env.reward_win and flags.env.reward_lose")
        if flags.env.state_counter != "none":
            kwargs.update(state_counter=flags.env.state_counter)
        if flags.env.reduced_action:
            kwargs.update(actions=REDUCED_ACTIONS)
        env = env_class(**kwargs)
        if flags.seedspath is not None and len(flags.seedspath) > 0:
            json  # Unused.
            raise NotImplementedError("seedspath > 0 not implemented yet.")
        #     with open(flags.seedspath) as f:
        #         seeds = json.load(f)
        #     assert flags.num_seeds == len(seeds)
        #     env = SeedingWrapper(env, seeds=seeds)
        # elif flags.num_seeds > 0:
        #     env = SeedingWrapper(env, num_seeds=flags.num_seeds)
        return env


def serve(flags, server_address, env_id):
    env = lambda: create_env(flags, env_id)
    server = libtorchbeast.Server(env, server_address=server_address)
    server.run()


def main(flags):
    create_folders(flags)

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=serve, args=(flags, f"{flags.pipes_basename}.{i}", i), daemon=True
        )
        p.start()
        processes.append(p)

    handler = lambda s, f: sys.exit()
    signal.signal(signal.SIGTERM, handler)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
