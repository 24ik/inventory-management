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

import mmle.distributed as mdistd  # pylint: disable=E0401
import mmle.nn as mnn  # pylint: disable=E0401
import mmle.utils as mut  # pylint: disable=E0401
import torch  # pylint: disable=E0401
import torch.distributed as distd  # pylint: disable=E0401
import torch.nn as nn  # pylint: disable=E0401
import torch.nn.functional as F  # pylint: disable=E0401
import torch.optim as optim  # pylint: disable=E0401
from torch.utils.data import DataLoader  # pylint: disable=E0401
from torch.utils.data.distributed import DistributedSampler  # pylint: disable=E0401


BATCH_SIZE = 400
DATA_NUM = 40000
DATA_DIM = 10
EPOCH_NUM = 50
LABEL_DIM = 5
LOG_DIR = "log"
MIDDLE_DIM = 16


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data = torch.randn(DATA_NUM, DATA_DIM)
        self.label = torch.randn(DATA_NUM, LABEL_DIM)

    def __len__(self):
        return DATA_NUM

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def main():
    world_size = distd.get_world_size()
    rank = distd.get_rank()

    model = nn.Sequential(
        mnn.FC(DATA_DIM, MIDDLE_DIM), mnn.FC(MIDDLE_DIM, LABEL_DIM, bn=False, activ="id")
    )
    model = mdistd.distribute_module(model)

    optimizer = optim.Adam(model.parameters())

    dataset = Dataset()
    sampler = DistributedSampler(dataset, world_size, rank)
    loader = DataLoader(dataset, BATCH_SIZE, sampler=sampler, drop_last=True)

    if rank == 0:
        manager = mut.Manager(model, optimizer, log_dir=mut.get_time_dir(LOG_DIR))
        manager.add_graph(dataset[0][0].repeat(BATCH_SIZE, 1))
        step = 0

    for epoch in mut.range1(EPOCH_NUM):
        model.train()
        for data, label in loader:
            loss = F.mse_loss(model(data), label.to(rank))

            mnn.zero_grad(model)
            loss.backward()
            optimizer.step()
            distd.barrier()

            if rank == 0:
                step += world_size
                manager.plot("loss", "train", loss.item(), step)

        if rank == 0:
            print(f"Finish epoch {epoch}: loss={loss.item():.3f}")
            params_dict = manager.get_params_dict()
            manager.save(step=step, **params_dict)
        distd.barrier()

    if rank == 0:
        manager.close()


if __name__ == "__main__":
    mdistd.spawn(main, nprocs=torch.cuda.device_count())
