######################################################
Inventory Management with Attention-Based Meta Actions
######################################################

This is an official code of "Keisuke Izumiya and Edgar Simo-Serra, Inventory Management with Attention-Based Meta Actions, IEEE Conference on Games (CoG), 2021."

Our research and source codes are based on the `NLE <https://github.com/facebookresearch/nle>`_ (Küttler et al., The NetHack Learning Environment, NeurIPS, 2021), distributed in the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
The changes are shown in :code:`CHANGELOG.md`.

************
Installation
************

Requirements
============

* Ubuntu
* Python3 (:code:`3.6` or later)
* `CMake <https://cmake.org>`_ (:code:`3.14` or later)
* Some packages, installed by:
::

    $ sudo apt-get install -y build-essential autoconf libtool pkg-config git flex bison libbz2-dev

Installation with Poetry
========================

Installation with `poetry <https://python-poetry.org/>`_:
::

    $ poetry add git+ssh://git@github.com/izumiya-keisuke/inventory-management.git#main

Installation without Poetry
===========================

::

    $ pip install git+ssh://git@github.com/izumiya-keisuke/nest.git
    $ git clone --recursive git+ssh://git@github.com/izumiya-keisuke/inventory-management.git
    $ cd inventory-management
    $ pip install .

********
Training
********

If you installed this repo with poetry:
::

    $ python -m nle.agent.polyhydra

Otherwise:
::

    $ python inventory-management/nle/agent/polyhydra.py

In both cases, an output directory (contains learned model weights, a tensorboard log, etc.) is generated in the current working directory.

You can adjust configurations from CLI, like:
::

    $ python -m nle.agent.polyhydra batch_size=16 model.hidden_dim=128

The configuration list is shown in :code:`nle/agent/config.yaml`.

*******
Testing
*******

If you installed this repo with poetry:
::

    $ python -m nle.agent.test outputs/YYYY-MM-DD/hh-mm-ss

Otherwise:
::

    $ python inventory-management/nle/agent/test.py outputs/YYYY-MM-DD/hh-mm-ss

By default, the agent's behavior is not displayed; specify the :code:`--render` option to show it.
You can see other options by specifing :code:`-h`.

********
Citation
********

::

    @inproceedings{izumiya2021cog,
    author = "Keisuke Izumiya and Edgar Simo-Serra",
    title = "Inventory Management with Attention-Based Meta Actions",
    booktitle = "IEEE Conference on Games (CoG)",
    year = 2021,
    }

*******
License
*******

The unmodified files are provided under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_, and the modified files are provided under `GPLv3+ <https://www.gnu.org/licenses/gpl-3.0.html>`_.
Details of the changes are shown in the :code:`CHANGELOG.md`.
