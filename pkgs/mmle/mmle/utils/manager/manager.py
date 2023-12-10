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

import torch  # pylint: disable=E0401
from torch.utils.tensorboard import SummaryWriter  # pylint: disable=E0401

import mmle.nn as mnn
import mmle.utils as mut


class Manager:
    def __init__(
        self, module=None, optimizer=None, scheduler=None, log_dir=None, mkdir=True, use_writer=True
    ):
        self._module = module
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._save_d = None
        self._writer = None
        if log_dir is not None:
            log_d = mut.convert_to_path(log_dir)
            self._save_d = log_d / "save"

            if mkdir:
                self._save_d.mkdir(parents=True, exist_ok=True)
            if use_writer:
                self._writer = SummaryWriter(log_d / "tb")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, writer):
        self._writer = writer

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def save(self, save_as_dict=True, allow_overwrite=True, **save_items):
        if save_as_dict:
            dict_ = {"data": save_items}
        else:
            dict_ = save_items

        for name, obj_ in dict_.items():
            save_p = self._save_d / f"{name}.pth"
            if not allow_overwrite and save_p.exists():
                raise ValueError(f"{save_p} already exists.")

            torch.save(obj_, save_p)

    def load(self, *names, map_location=None):
        results = []
        for name in names:
            load_p = self._save_d / f"{name}.pth"
            results.append(torch.load(load_p, map_location))

        if len(results) == 1:
            return results[0]
        else:
            return results

    def get_params_dict(self):
        params_dict = {}

        if self._module is not None:
            if mnn.is_dataparallel(self._module):
                params_dict["module"] = self._module.module.state_dict()
            else:
                params_dict["module"] = self._module.state_dict()

        if self._optimizer is not None:
            params_dict["optimizer"] = self._optimizer.state_dict()

        if self._scheduler is not None:
            params_dict["scheduler"] = self._scheduler.state_dict()

        return params_dict

    def set_params_dict(self, **params_dict):
        if self._module is not None and "module" in params_dict:
            if mnn.is_dataparallel(self._module):
                self._module.module.load_state_dict(params_dict["module"])
            else:
                self._module.load_state_dict(params_dict["module"])

        if self._optimizer is not None and "optimizer" in params_dict:
            self._optimizer.load_state_dict(params_dict["optimizer"])

        if self._scheduler is not None and "scheduler" in params_dict:
            self._scheduler.load_state_dict(params_dict["scheduler"])

    def plot(self, tag, legend, val, step=None):
        self.writer.add_scalars(tag, {legend: val}, step)

    def add_graph(self, *sample_input, device=None):
        if mnn.is_dataparallel(self._module):
            module = self._module.module
        else:
            module = self._module

        if device is None:
            device = next(self._module.parameters()).device

        with torch.no_grad():
            self.writer.add_graph(module, mut.to(sample_input, device))
