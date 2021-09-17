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

import logging
import multiprocessing as mp
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from nle.agent import polybeast_env, polybeast_learner


if (
    torch.__version__.startswith("1.5")
    or torch.__version__.startswith("1.6")
    or torch.__version__.startswith("1.7")
    or torch.__version__.startswith("1.8")
    or torch.__version__.startswith("1.9")
):
    # pytorch 1.5.* needs this for some reason on the cluster
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
    level=0,
)


def pipes_basename():
    logdir = Path(os.getcwd())
    name = ".".join([logdir.parents[1].name, logdir.parents[0].name, logdir.name])
    return "unix:/tmp/poly.%s" % name


def get_common_flags(flags):
    flags = OmegaConf.to_container(flags)
    flags["pipes_basename"] = pipes_basename()
    flags["savedir"] = os.getcwd()
    return OmegaConf.create(flags)


def get_learner_flags(flags):
    lrn_flags = OmegaConf.to_container(flags)
    lrn_flags["checkpoint"] = os.path.join(flags["savedir"], "checkpoint.tar")
    lrn_flags["max_learner_queue_size"] = flags.batch_size
    return OmegaConf.create(lrn_flags)


def run_learner(flags: DictConfig):
    polybeast_learner.main(flags)


def get_environment_flags(flags):
    env_flags = OmegaConf.to_container(flags)
    env_flags["num_servers"] = flags.num_actors
    env_flags["max_num_steps"] = int(flags.env.max_num_steps)
    env_flags["seedspath"] = ""
    return OmegaConf.create(env_flags)


def run_env(flags):
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags)


@hydra.main(config_path=".", config_name="config")
def main(flags: DictConfig):
    OmegaConf.save(flags, "config.yaml")

    flags = get_common_flags(flags)

    env_flags = get_environment_flags(flags)
    env_process = mp.Process(target=run_env, args=(env_flags,))
    env_process.start()

    try:
        lrn_flags = get_learner_flags(flags)
        run_learner(lrn_flags)
    except KeyboardInterrupt:
        pass
    finally:
        env_process.terminate()
        env_process.join()


if __name__ == "__main__":
    main()
