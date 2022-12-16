"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

extention of RLlib's MARWIL for RNN policy
1. Adds duumy RNN states into SampleBatch as postprocess.
2. Fixes memory leak in torch version reported in the issue #21291 but not merged as of ray 1.13.
"""
from typing import Optional, Type

from ray.rllib.agents import marwil
from ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil_tf_policy import RNNMARWILTFPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import (
    ConcatBatches,
    ParallelRollouts,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    TrainOneStep,
    train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
    WORKER_UPDATE_TIMER,
)
from ray.rllib.utils.typing import (
    PartialTrainerConfigDict,
    ResultDict,
    TrainerConfigDict,
)
from ray.util.iter import LocalIterator

DEFAULT_CONFIG = marwil.MARWILTrainer.merge_trainer_configs(
    marwil.DEFAULT_CONFIG,
    {
        "replay_sequence_length": 1, # should be same as "max_seq_len" when using LSTM, otherwise must be 1.
    },
    _allow_unknown_configs=True,
)

class RNNMARWILTrainer(marwil.MARWILTrainer):
    @classmethod
    @override(marwil.MARWILTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(marwil.MARWILTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil_torch_policy import RNNMARWILTorchPolicy
            return RNNMARWILTorchPolicy
        else:
            return RNNMARWILTFPolicy

    @override(marwil.MARWILTrainer)
    def setup(self, config: PartialTrainerConfigDict):
        super(marwil.MARWILTrainer, self).setup(config)
        # `training_iteration` implementation: Setup buffer in `setup`, not
        # in `execution_plan` (deprecated).
        if self.config["_disable_execution_plan_api"] is True:
            replay_sequence_length = self.config.get("replay_sequence_length", 1)
            self.local_replay_buffer = MultiAgentReplayBuffer(
                learning_starts=self.config["learning_starts"],
                capacity=self.config["replay_buffer_size"],
                replay_batch_size=self.config["train_batch_size"],
                replay_sequence_length=replay_sequence_length,
                replay_zero_init_states=True
            )

    @staticmethod
    @override(marwil.MARWILTrainer)
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        assert (
            len(kwargs) == 0
        ), "Marwill execution_plan does NOT take any additional parameters"

        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        replay_sequence_length = self.config.get("replay_sequence_length", 1)
        replay_buffer = MultiAgentReplayBuffer(
            learning_starts=config["learning_starts"],
            capacity=config["replay_buffer_size"],
            replay_batch_size=config["train_batch_size"],
            replay_sequence_length=replay_sequence_length,
            replay_zero_init_states=True
        )

        store_op = rollouts.for_each(StoreToReplayBuffer(local_buffer=replay_buffer))

        replay_op = (
            Replay(local_buffer=replay_buffer)
            .combine(
                ConcatBatches(
                    min_batch_size=config["train_batch_size"],
                    count_steps_by=config["multiagent"]["count_steps_by"],
                )
            )
            .for_each(TrainOneStep(workers))
        )

        train_op = Concurrently(
            [store_op, replay_op], mode="round_robin", output_indexes=[1]
        )

        return StandardMetricsReporting(train_op, workers, config)
