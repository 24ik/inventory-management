"""
Official code of "Keisuke Izumiya and Edgar Simo-Serra, Inventory Management with Attention-Based Meta Actions, IEEE Conference on Games (CoG), 2021."
    Copyright (C) 2021 Keisuke Izumiya

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run with OMP_NUM_THREADS=1.
#

import collections
import logging
import os
import threading
import time
import timeit
import traceback
from pathlib import Path

import libtorchbeast
import mmle
import mmle.nn as mnn
import mmle.utils as mut
import nest
import omegaconf
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from nle.agent.core import file_writer, vtrace
from nle.agent.models import create_model, losses
from nle.agent.models.model import NetHackNet


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def inference(
    inference_batcher, model, flags, actor_device, lock=threading.Lock()
):  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            observation, reward, done, last_action, *_ = batched_env_outputs
            # Observation is a dict with keys 'features' and 'glyphs'.
            observation["done"] = done
            observation["last_action"] = last_action
            observation, agent_state = nest.map(
                lambda t: t.to(actor_device, non_blocking=True),
                (observation, agent_state),
            )
            with lock:
                outputs = model(observation, agent_state)
            core_outputs, agent_state = nest.map(lambda t: t.cpu(), outputs)
            # Restructuring the output in the way that is expected
            # by the functions in actorpool.
            outputs = (
                tuple(
                    (
                        core_outputs["action"],
                        core_outputs["policy_log_prob"],
                        core_outputs["baseline"],
                        core_outputs["virtual_log_prob"],
                        core_outputs["item_log_prob"],
                    )
                ),
                agent_state,
            )
            batch.set_outputs(outputs)


# TODO(heiner): Given that our nest implementation doesn't support
# namedtuples, using them here doesn't seem like a good fit. We
# probably want to nestify the environment server and deal with
# dictionaries?
EnvOutput = collections.namedtuple(
    "EnvOutput", "frame rewards done last_action episode_step episode_return"
)
AgentOutput = NetHackNet.AgentOutput
Batch = collections.namedtuple("Batch", "env agent")


def clip(flags, rewards):
    if flags.reward_clipping == "tim":
        clipped_rewards = torch.tanh(rewards / 100.0)
    elif flags.reward_clipping == "soft_asymmetric":
        squeezed = torch.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed, squeezed) * 5.0
    elif flags.reward_clipping == "none":
        clipped_rewards = rewards
    else:
        raise NotImplementedError("reward_clipping=%s" % flags.reward_clipping)
    return clipped_rewards


def learn(
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    flags,
    plogger,
    learner_device,
    lock=threading.Lock(),  # noqa: B008
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        observation, reward, done, last_action, *_ = env_outputs
        observation["reward"] = reward
        observation["done"] = done
        observation["last_action"] = last_action

        lock.acquire()  # Only one thread learning at a time.

        output, _ = model(observation, initial_agent_state)

        # Use last baseline value (from the value function) to bootstrap.
        learner_outputs = AgentOutput._make((
            output["action"],
            output["policy_log_prob"],
            output["baseline"],
            output["virtual_log_prob"],
            output["item_log_prob"],
        ))

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in `batch` and `learner_outputs` at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        # Note that the env_outputs.frame is now a dict with 'features' and 'glyphs'
        # instead of actually being the frame itself. This is currently not a problem
        # because we never use actor_outputs.frame in the rest of this function.
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        rewards = env_outputs.rewards
        if flags.normalize_reward:
            model.update_running_moments(rewards)
            rewards /= model.get_running_std()

        total_loss = 0

        # STANDARD EXTRINSIC LOSSES / REWARDS
        entropy_loss = torch.tensor(0.0, device=learner_device)
        entropy_loss_virtual = None
        entropy_loss_item = None
        if flags.loss.entropy_cost.virtual > 0:
            entropy_loss_virtual = flags.loss.entropy_cost.virtual * losses.compute_entropy_loss(
                learner_outputs.virtual_log_prob)
            entropy_loss += entropy_loss_virtual
        if flags.model.policy.model == "meta" and flags.loss.entropy_cost.item > 0:
            entropy_loss_item = flags.loss.entropy_cost.item * losses.compute_entropy_loss(
                learner_outputs.item_log_prob,
                learner_outputs.virtual_log_prob[:, :, -1].exp(),
            )
            entropy_loss += entropy_loss_item
        total_loss += entropy_loss

        clipped_rewards = clip(flags, rewards)

        discounts = (~env_outputs.done).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_log_prob(
            behavior_policy_log_prob=actor_outputs.policy_log_prob,
            target_policy_log_prob=learner_outputs.policy_log_prob,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=learner_outputs.baseline[-1],
        )

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = losses.compute_policy_gradient_loss(
            learner_outputs.policy_log_prob,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.loss.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        total_loss += pg_loss + baseline_loss

        # BACKWARD STEP
        mnn.zero_grad(model)
        total_loss.backward()
        if flags.grad_norm_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        # LOGGING
        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        if entropy_loss_virtual is not None:
            stats["entropy_loss_virtual"] = entropy_loss_virtual.item()
        if entropy_loss_item is not None:
            stats["entropy_loss_item"] = entropy_loss_item.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if "state_visits" in observation:
            visits = observation["state_visits"][:-1]
            metric = visits[env_outputs.done].float()
            key1 = "mean_state_visits"
            key2 = "max_state_visits"
            if not len(episode_returns):
                stats[key1] = None
                stats[key2] = None
            else:
                stats[key1] = torch.mean(metric).item()
                stats[key2] = torch.max(metric).item()

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


def train(flags):
    logging.info("Logging results to %s", flags.savedir)
    if isinstance(flags, omegaconf.DictConfig):
        flag_dict = omegaconf.OmegaConf.to_container(flags)
    else:
        flag_dict = vars(flags)
    plogger = file_writer.FileWriter(xp_args=flag_dict, rootdir=flags.savedir)

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        learner_device = torch.device(flags.learner_device)
        actor_device = torch.device(flags.actor_device)
    else:
        logging.info("Not using CUDA.")
        learner_device = torch.device("cpu")
        actor_device = torch.device("cpu")

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static. We could make it dynamic, but that
    # requires a loss (and learning rate schedule) that's batch size
    # independent.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    model = create_model(flags, learner_device)

    plogger.metadata["model_numel"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logging.info("Number of model parameters: %i", plogger.metadata["model_numel"])

    actor_model = create_model(flags, actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.opt.learning_rate,
        momentum=flags.opt.momentum,
        eps=flags.opt.epsilon,
        alpha=flags.opt.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    if flags.checkpoint and os.path.exists(flags.checkpoint):
        logging.info("Loading checkpoint: %s" % flags.checkpoint)
        checkpoint_states = torch.load(
            flags.checkpoint, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
                flags,
                plogger,
                learner_device,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, actor_model, flags, actor_device),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint(checkpoint_path=None):
        if flags.checkpoint:
            if checkpoint_path is None:
                checkpoint_path = flags.checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "stats": stats,
                    "flags": vars(flags),
                },
                checkpoint_path,
            )

    # TODO: test this again then uncomment (from deleted polyhydra code)
    # def receive_slurm_signal(signal_num=None, frame=None):
    #     logging.info("Received SIGTERM, checkpointing")
    #     make_checkpoint()

    # signal.signal(signal.SIGTERM, receive_slurm_signal)

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    plot_keys = (
        "mean_episode_return",
        "mean_episode_step",
        "total_loss",
        "entropy_loss",
        "entropy_loss_virtual",
        "entropy_loss_item",
        "pg_loss",
        "baseline_loss",
    )
    manager = mut.Manager(log_dir=Path(flags.checkpoint).resolve().parent, mkdir=False)
    pbar = tqdm(total=int(flags.total_steps))

    try:
        train_start_time = timeit.default_timer()
        train_time_offset = stats.get("train_seconds", 0)  # used for resuming training
        last_checkpoint_time = timeit.default_timer()

        dev_checkpoint_intervals = [0, 0.25, 0.5, 0.75]

        loop_start_time = timeit.default_timer()
        loop_start_step = stats.get("step", 0)
        while True:
            if loop_start_step >= flags.total_steps:
                break
            time.sleep(5)
            loop_end_time = timeit.default_timer()
            loop_end_step = stats.get("step", 0)

            stats["train_seconds"] = round(
                loop_end_time - train_start_time + train_time_offset, 1
            )

            if loop_end_time - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = loop_end_time

            if len(dev_checkpoint_intervals) > 0:
                step_percentage = loop_end_step / flags.total_steps
                i = dev_checkpoint_intervals[0]
                if step_percentage > i:
                    checkpoint(flags.checkpoint[:-4] + "_" + str(i) + ".tar")
                    dev_checkpoint_intervals = dev_checkpoint_intervals[1:]

            step_diff = loop_end_step - loop_start_step

            loop_start_time = loop_end_time
            loop_start_step = loop_end_step

            for key in plot_keys:
                if key not in stats:
                    continue

                val = stats[key]
                if val is not None:
                    manager.plot(f'stat/{key}', 'val', val, loop_end_step)

            pbar.update(step_diff)
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])

        checkpoint()
    finally:
        manager.close()
        pbar.close()

        # Done with learning. Let's stop all the ongoing work.
        inference_batcher.close()
        learner_queue.close()

        actorpool_thread.join()

        for t in learner_threads + inference_threads:
            t.join()


def main(flags):
    train(flags)
