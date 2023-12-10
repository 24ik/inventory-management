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


def setup_device(idx=None, cudnn_benchmark=True):
    if torch.cuda.is_available():
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        device_str = "cuda"
        if idx is not None:
            device_str += f":{idx}"
    else:
        device_str = "cpu"

    return torch.device(device_str)


def rec_map(fn, data):
    if isinstance(data, tuple):
        return tuple([rec_map(fn, d) for d in data])
    elif isinstance(data, list):
        return [rec_map(fn, d) for d in data]
    elif isinstance(data, dict):
        return {k: rec_map(fn, v) for k, v in data.items()}
    else:
        return fn(data)


def to(data, *args, **kwargs):
    return rec_map(lambda d: d.to(*args, **kwargs), data)
