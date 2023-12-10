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

import torch  # pylint: disable=E0401
import torch.nn as nn  # pylint: disable=E0401
import torch.nn.functional as F  # pylint: disable=E0401

import mmle
import mmle.nn.functional as mF


class Module(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = None

    def set_core(self, core):
        self.core = core

    def forward(self, x):
        return self.core(x)


class ActivModule(Module):
    def __init__(self, activ="relu", activ_kwargs=None):
        super().__init__()

        if activ_kwargs is None:
            activ_kwargs = {}
        self.activ = self._get_activ_fn(activ, **activ_kwargs)

    def forward(self, x):
        return self.activ(self.core(x))

    @staticmethod
    def _get_activ_fn(str_or_fn, **kwargs):
        if hasattr(str_or_fn, "__call__"):
            fn = str_or_fn
        elif not isinstance(str_or_fn, str):
            raise ValueError(f"str_or_fn must be str or function, but got {type(str_or_fn)}.")
        elif str_or_fn in ("id", "none"):
            fn = mmle.id_fn
        elif hasattr(mmle, str_or_fn):
            fn = eval(f"mmle.{str_or_fn}")
        elif hasattr(mF, str_or_fn):
            fn = eval(f"mF.{str_or_fn}")
        elif hasattr(torch, str_or_fn):
            fn = eval(f"torch.{str_or_fn}")
        elif hasattr(F, str_or_fn):
            fn = eval(f"F.{str_or_fn}")
        else:
            raise ValueError(f"Invalid input: str_or_fn={str_or_fn}")

        return functools.partial(fn, **kwargs)


class ParamModule(ActivModule):
    def __init__(
        self,
        how_init_weight=None,
        init_weight_kwargs=None,
        how_init_bias=None,
        init_bias_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # weight
        if init_weight_kwargs is None:
            init_weight_kwargs = {}
        self._init_weight_data = (how_init_weight, init_weight_kwargs)

        # bias
        if init_bias_kwargs is None:
            init_bias_kwargs = {}
        self._init_bias_data = (how_init_bias, init_bias_kwargs)

    def set_core(self, core):
        self.core = core
        self._init_param()

    def _init_param(self):
        # initialize weight
        how_init, init_kwargs = self._init_weight_data
        if how_init is not None:
            self._get_init(how_init)(self.core.weight, **init_kwargs)

        # initialize bias
        how_init, init_kwargs = self._init_bias_data
        if how_init is not None:
            self._get_init(how_init)(self.core.bias, **init_kwargs)

    @staticmethod
    def _get_init(str_or_fn):
        if hasattr(str_or_fn, "__call__"):
            return str_or_fn
        elif not isinstance(str_or_fn, str):
            raise ValueError(f"str_or_fn must be str or function, but got {type(str_or_fn)}.")
        elif hasattr(nn.init, str_or_fn):
            return eval(f"nn.init.{str_or_fn}")
        else:
            raise ValueError(f"Invalid input: str_or_fn={str_or_fn}")


class BNModule(ParamModule):
    def __init__(self, dim=None, num_features=None, bn=True, **kwargs):
        super().__init__(**kwargs)

        if bn:
            if dim in (1, 2, 3):
                self.bn = eval(f"nn.BatchNorm{dim}d")(num_features)
            else:
                raise ValueError(f"dim must be 1, 2, or 3, but got {dim}.")
        else:
            self.bn = mmle.id_fn

    def forward(self, x):
        return self.activ(self.bn(self.core(x)))
