import logging
import os
import time
from pathlib import Path

import mmle.nn as mnn
import torch
from omegaconf import OmegaConf

from nle.agent import polybeast_env, polybeast_learner
from nle.agent.polyhydra import get_common_flags, get_environment_flags, get_learner_flags


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

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
    level=0,
)


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("log_dir", help="log directory.")
    parser.add_argument("--greedy", action="store_true", help="act greedily.")
    parser.add_argument("--render", action="store_true", help="Render.")
    parser.add_argument("--num-episodes", default=1, type=int, help="Number of episode[s].")
    parser.add_argument(
        "--sleep-sec", default=0.3, type=float, help="sleep sec betweeen rendering."
    )
    parser.add_argument("--seed-core", default=123, type=int, help="seed for core.")
    parser.add_argument("--seed-disp", default=456, type=int, help="seed for disp.")

    return parser.parse_args()


class Env:
    def __init__(self, gym_env):
        self.env = gym_env

        self.episode_return = None
        self.episode_step = None

    def initial(self):
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int)

        obs = self._format_observations(self.env.reset())
        obs.update(
            reward=torch.zeros(1, 1),
            done=torch.ones(1, 1, dtype=torch.int8),
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=torch.ones(1, 1, dtype=torch.long),
        )

        return obs

    def step(self, action):
        np_obs, reward, done, _ = self.env.step(action.item())
        self.episode_return += reward
        self.episode_step += 1
        episode_return = self.episode_return
        episode_step = self.episode_step
        if done:
            np_obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int)

        obs = self._format_observations(np_obs)
        reward = torch.tensor([[reward]])
        done = torch.tensor([[done]])

        obs.update(
            reward=torch.tensor([[reward]]),
            done=torch.tensor([[done]]),
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

        return obs

    @staticmethod
    def _format_observations(np_obs):
        obs = {}
        for key in np_obs:
            val = torch.from_numpy(np_obs[key])
            obs[key] = val.view((1, 1) + val.shape)

        return obs


def main():
    args = parse_args()

    savedir = Path(args.log_dir).resolve()

    flags = OmegaConf.load(savedir / "config.yaml")

    # adjust flags
    flags.num_actors = 1
    flags = get_common_flags(flags)
    env_flags = get_environment_flags(flags)
    flags["savedir"] = str(savedir)
    lrn_flags = get_learner_flags(flags)

    # make env
    env = Env(polybeast_env.create_env(env_flags, full_obs=True))
    env.env.seed(args.seed_core, args.seed_disp)

    # make and load model
    model = polybeast_learner.create_model(lrn_flags, "cpu")
    mnn.freeze(model)
    checkpoint = torch.load(lrn_flags.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # main
    observation = env.initial()
    try:
        agent_state = model.initial_state(batch_size=1)
        zero = torch.tensor(0)
        policy_outputs = {"action": zero}  # dummy

        returns = []

        while len(returns) < args.num_episodes:
            if args.render:
                env.env.render()

            policy_outputs, agent_state = model(observation, agent_state, greedy=args.greedy)

            if args.render:
                time.sleep(args.sleep_sec)

            observation = env.step(policy_outputs["action"])

            if observation["done"].item():
                step = observation["episode_step"].item()
                return_ = observation["episode_return"].item()
                logging.info(f"Episode step: {step} / return: {return_:.1f}")

                returns.append(return_)

                agent_state = model.initial_state(batch_size=1)
                policy_outputs = {"action": zero}  # dummy
    except KeyboardInterrupt:
        return
    finally:
        env.env.close()

        if len(returns) > 0:
            logging.info(
                f"Return mean over {len(returns)} episode[s]: {sum(returns) / len(returns):.1f}"
            )


if __name__ == "__main__":
    main()
