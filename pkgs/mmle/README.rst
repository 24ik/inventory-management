####
MMLE
####

MMLE (Make Machine Learning Easy) is a helper for machine learning with `PyTorch <https://pytorch.org/>`_.

*******
Install
*******

If you use `poetry <https://python-poetry.org/>`_, you can install MMLE by:
::

    $ poetry add git+https://github.com/izumiya-keisuke/mmle.git

Installing with pip is also supported:
::

    $ pip install https://github.com/izumiya-keisuke/mmle.git

********
Features
********

torch.nn.Module extension
=========================

MMLE provides additional :code:`torch.nn.Module` such as :code:`FC`, :code:`Conv` and so on.

Example:
::

    import mmle.nn as mnn

    # By default, fully-connected layer contains batch normalization and relu activation
    fc1 = mnn.FC(2, 3)  # nn.Linear(2, 3, bias=False) -> nn.BatchNorm1d(3) -> nn.ReLU()

    # You can set activation function and batch normalization use
    fc2 = mnn.FC(4, 5, activ="tanh", bn=False)  # nn.Linear(4, 5) -> nn.Tanh()

    # Convolution layer has different default parameters from torch.nn.ConvNd
    conv1 = mnn.Conv(6, 7)  # nn.Conv2d(6, 7, 3, 1, 1, bias=False) -> nn.BatchNorm2d(7) -> nn.ReLU()


Manager
=======

:code:`Manager` is learning manager, which supports saving, loading, managing model parameters and so on.

An example is shown in :code:`examples/distributed_data_parallel.py`.

Distributed training
====================

MMLE supports distributed training with some functions, such as initialization, model conversion (to :code:`torch.nn.parallel.DistributedDataParallel`), and so on.

An example is shown in :code:`examples/distributed_data_parallel.py`.

Utilities
=========

Some utility functions are also provided.

Example:
::

    import mmle.nn as mnn
    import mmle.utils as mut

    model = mnn.FC(2, 3)
    mnn.zero_grad(model)  # better than model.zero_grad()
    mnn.freeze(model)  # fix parameters (set requires_grad=False)

    time_dir = mut.get_time_dir("foo")  # pathlib.Path("foo/YYYYMMDD-hhmmss")
    for i in mut.range1(5):
        print(i)  # 1, 2, 3, 4, 5

*******
License
*******

Apache-2.0
