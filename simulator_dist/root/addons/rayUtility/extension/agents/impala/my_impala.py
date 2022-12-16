"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

Added "manual_update_for_untrainable_policies" key in the config into ImpalaTrainer of ray 1.13.0.
Setting True for this key disables broadcasting weights for untrainable policies.
This makes it possible to manage opponents' weights manually for custom match-making with past agents.
"""
import logging
from typing import Optional, Type

import ray
from ray.rllib.agents import impala
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceTFPolicy
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner_thread import MultiGPULearnerThread
from ray.rllib.execution.tree_agg import gather_experiences_tree_aggregation
from ray.rllib.execution.common import (
    STEPS_TRAINED_COUNTER,
    STEPS_TRAINED_THIS_ITER_COUNTER,
    _get_global_vars,
    _get_shared_metrics,
)
from ray.rllib.execution.replay_ops import MixInReplay
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.typing import PartialTrainerConfigDict, TrainerConfigDict
from ray.tune.utils.placement_groups import PlacementGroupFactory

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = impala.ImpalaTrainer.merge_trainer_configs(
    impala.DEFAULT_CONFIG,
    {
        #Whether to update weights for untrainable policies manually outside of the trainer or not.
        "manual_update_for_untrainable_policies":True,
    },
    _allow_unknown_configs=True,
)

def make_learner_thread(local_worker, config):
    if not config["simple_optimizer"]:
        logger.info(
            "Enabling multi-GPU mode, {} GPUs, {} parallel tower-stacks".format(
                config["num_gpus"], config["num_multi_gpu_tower_stacks"]
            )
        )
        num_stacks = config["num_multi_gpu_tower_stacks"]
        buffer_size = config["minibatch_buffer_size"]
        if num_stacks < buffer_size:
            logger.warning(
                "In multi-GPU mode you should have at least as many "
                "multi-GPU tower stacks (to load data into on one device) as "
                "you have stack-index slots in the buffer! You have "
                f"configured {num_stacks} stacks and a buffer of size "
                f"{buffer_size}. Setting "
                f"`minibatch_buffer_size={num_stacks}`."
            )
            config["minibatch_buffer_size"] = num_stacks

        learner_thread = MultiGPULearnerThread(
            local_worker,
            num_gpus=config["num_gpus"],
            lr=config["lr"],
            train_batch_size=config["train_batch_size"],
            num_multi_gpu_tower_stacks=config["num_multi_gpu_tower_stacks"],
            num_sgd_iter=config["num_sgd_iter"],
            learner_queue_size=config["learner_queue_size"],
            learner_queue_timeout=config["learner_queue_timeout"],
        )
    else:
        learner_thread = LearnerThread(
            local_worker,
            minibatch_buffer_size=config["minibatch_buffer_size"],
            num_sgd_iter=config["num_sgd_iter"],
            learner_queue_size=config["learner_queue_size"],
            learner_queue_timeout=config["learner_queue_timeout"],
        )
    return learner_thread


def gather_experiences_directly(workers, config):
    rollouts = ParallelRollouts(
        workers,
        mode="async",
        num_async=config["max_sample_requests_in_flight_per_worker"],
    )

    # Augment with replay and concat to desired train batch size.
    train_batches = (
        rollouts.for_each(lambda batch: batch.decompress_if_needed())
        .for_each(
            MixInReplay(
                num_slots=config["replay_buffer_num_slots"],
                replay_proportion=config["replay_proportion"],
            )
        )
        .flatten()
        .combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            )
        )
    )

    return train_batches


# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights:
    """ASRC: Added untrainables_exclusion flag into the original version
    """
    def __init__(self, learner_thread, workers, broadcast_interval, untrainables_exclusion = False):
        self.learner_thread = learner_thread
        self.steps_since_broadcast = 0
        self.broadcast_interval = broadcast_interval
        self.untrainables_exclusion = untrainables_exclusion
        self.workers = workers
        if(self.untrainables_exclusion):
            policies=workers.local_worker().foreach_policy_to_train(lambda policy,pid:pid)
            self.weights = workers.local_worker().get_weights(policies)
        else:
            self.weights = workers.local_worker().get_weights()

    def __call__(self, item):
        actor, batch = item
        self.steps_since_broadcast += 1
        if (
            self.steps_since_broadcast >= self.broadcast_interval
            and self.learner_thread.weights_updated
        ):
            if(self.untrainables_exclusion):
                policies=self.workers.local_worker().foreach_policy_to_train(lambda policy,pid:pid)
                self.weights = ray.put(self.workers.local_worker().get_weights(policies))
            else:
                self.weights = ray.put(self.workers.local_worker().get_weights())
            self.steps_since_broadcast = 0
            self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
        actor.set_weights.remote(self.weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())

class ManualUpdateForUntrainablesMixin:
    @staticmethod
    def execution_plan(workers, config, **kwargs):
        assert (
            len(kwargs) == 0
        ), "IMPALA execution_plan does NOT take any additional parameters"

        if config["num_aggregation_workers"] > 0:
            train_batches = gather_experiences_tree_aggregation(workers, config)
        else:
            train_batches = gather_experiences_directly(workers, config)

        # Start the learner thread.
        learner_thread = make_learner_thread(workers.local_worker(), config)
        learner_thread.start()

        # This sub-flow sends experiences to the learner.
        enqueue_op = train_batches.for_each(Enqueue(learner_thread.inqueue))
        # Only need to update workers if there are remote workers.
        if workers.remote_workers():
            enqueue_op = enqueue_op.zip_with_source_actor().for_each(
                BroadcastUpdateLearnerWeights(
                    learner_thread,
                    workers,
                    broadcast_interval=config["broadcast_interval"],
                    untrainables_exclusion = config["manual_update_for_untrainable_policies"]
                )
            )

        def record_steps_trained(item):
            count, fetches = item
            metrics = _get_shared_metrics()
            # Manually update the steps trained counter since the learner
            # thread is executing outside the pipeline.
            metrics.counters[STEPS_TRAINED_THIS_ITER_COUNTER] = count
            metrics.counters[STEPS_TRAINED_COUNTER] += count
            return item

        # This sub-flow updates the steps trained counter based on learner
        # output.
        dequeue_op = Dequeue(
            learner_thread.outqueue, check=learner_thread.is_alive
        ).for_each(record_steps_trained)

        merged_op = Concurrently(
            [enqueue_op, dequeue_op], mode="async", output_indexes=[1]
        )

        # Callback for APPO to use to update KL, target network periodically.
        # The input to the callback is the learner fetches dict.
        if config["after_train_step"]:
            merged_op = merged_op.for_each(lambda t: t[1]).for_each(
                config["after_train_step"](workers, config)
            )

        return StandardMetricsReporting(merged_op, workers, config).for_each(
            learner_thread.add_learner_metrics
        )

class MyImpalaTrainer(ManualUpdateForUntrainablesMixin, impala.ImpalaTrainer):
    @classmethod
    @override(impala.ImpalaTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG
