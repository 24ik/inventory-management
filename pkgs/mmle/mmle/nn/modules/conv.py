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

import torch.nn as nn  # pylint: disable=E0401
import torch.nn.functional as F  # pylint: disable=E0401

import mmle
from .module import BNModule


class _BaseConv(BNModule):
    def __init__(
        self,
        dim,
        c_in,
        c_out,
        kernel=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        upscale=None,
        **kwargs,
    ):
        super().__init__(dim, c_out, **kwargs)

        if bias is None:
            bias = not isinstance(self.bn, nn.Module)
        self.set_core(
            eval(f"nn.Conv{dim}d")(
                c_in, c_out, kernel, stride, padding, dilation, groups, bias, padding_mode
            )
        )

        self._upscale = mmle.id_fn
        if upscale is not None:
            if dim == 1:
                mode = "linear"
            elif dim == 2:
                mode = "bilinear"
            elif dim == 3:
                mode = "trilinear"
            else:
                raise ValueError(f"dim must be 1, 2, or 3, but got {dim}.")

            self._upscale = lambda x: F.interpolate(
                x, scale_factor=upscale, mode=mode, align_corners=False
            )

    def forward(self, x):
        return self.activ(self.bn(self.core(self._upscale(x))))


class Conv1d(_BaseConv):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class Conv2d(_BaseConv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class Conv3d(_BaseConv):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


Conv = Conv2d
