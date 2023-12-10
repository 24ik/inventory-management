"""
Copyright 2021 Keisuke Izumiya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from tempfile import gettempdir, NamedTemporaryFile

import torch  # pylint: disable=E0401
import torch.distributed as distd  # pylint: disable=E0401
import torch.multiprocessing as mp  # pylint: disable=E0401
from torch.nn import SyncBatchNorm as SBN  # pylint: disable=E0401
from torch.nn.parallel import DistributedDataParallel as DDP  # pylint: disable=E0401

import mmle.utils as mut


def _spawn_wrapper(world_size, args, kwargs, rank, fn, file_name):
    distd.init_process_group(
        "nccl", init_method=f"file://{file_name}", rank=rank, world_size=world_size
    )

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    fn(*args, **kwargs)


def spawn(fn, args=None, kwargs=None, nprocs=1, join=True, daemon=False):
    mp.spawn(
        functools.partial(_spawn_wrapper, nprocs, args, kwargs),
        (fn, NamedTemporaryFile().name),
        nprocs=nprocs,
        join=join,
        daemon=daemon,
    )


def sync(module, rank=None):
    if rank is None:
        rank = distd.get_rank()

    tmp_p = mut.convert_to_path(gettempdir()) / "_mmle_tmp"

    if rank == 0:
        torch.save(module.state_dict(), tmp_p)
    distd.barrier()

    module.load_state_dict(torch.load(tmp_p, map_location={"cuda:0": f"cuda:{rank}"}))

    if rank == 0:
        tmp_p.unlink()


def distribute_module(module, rank=None, *args, **kwargs):
    if rank is None:
        rank = distd.get_rank()

    module = DDP(
        SBN.convert_sync_batchnorm(module).to(rank),
        device_ids=[rank],
        output_device=rank,
        *args,
        **kwargs,
    )
    sync(module, rank)

    return module
