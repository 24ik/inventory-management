# Change Log

Our codes are based on [NLE](https://github.com/facebookresearch/nle) (mostly `neurips2020release` branch, partly `master` branch).

The changes are as follows:

- Rename original README.
    - `README.md` -> [README.md](/original-README.md)
    - `README.nh` -> [README.nh](/original-README.nh)
- Rename NetHack license.
    - `LICENSE` -> [nethack-LICENSE](/nethack-LICENSE)
- Remove unused files.
    - `nle/agent/models/intrinsic.py`
    - `nle/agent/models/dynamics.py`
    - `nle/agent/models/transformer.py`
- Add [README.rst](/README.rst) and [LICENSE](/LICENSE).
- Make installation easy.
    - [setup.py](/setup.py)
- Add reduced action space.
    - [tasks.py](/nle/env/tasks.py)
- Support for item-select actions.
    - [base.py](/nle/env/base.py)
- Refactor codes of model definitions.
    - `nle/agent/models/*.py`
- Refactor the configuration file.
    - [config.yaml](/nle/agent/config.yaml)
    - [polybeast_env.py](/nle/agent/polybeast_env.py)
    - [polybeast_learner.py](/nle/agent/polybeast_learner.py)
    - [polyhydra.py](/nle/agent/polyhydra.py)
    - `nle/agent/models/*.py`
- Fixed some minor behaviors in learning (process termination, logging using tensorboard, etc.)
    - [polybeast_env.py](/nle/agent/polybeast_env.py)
    - [polybeast_learner.py](/nle/agent/polybeast_learner.py)
    - [polyhydra.py](/nle/agent/polyhydra.py)
- Change loss calculation.
    - [losses.py](/nle/agent/models/losses.py)
    - [polybeast_learner.py](/nle/agent/polybeast_learner.py)
    - [vtrace.py](/nle/agent/core/vtrace.py)
- Rewrite [test.py](/nle/agent/test.py).
