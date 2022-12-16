"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

Behavioral Cloning (derived from MARWIL) is also extended for RNN Policy.
1. Adds RNN states into SampleBatch as postprocess.
2. Fixes memory leak in torch version reported in the issue #21291 but not merged as of ray 1.13.
"""
from ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil import(
    RNNMARWILTrainer,
    DEFAULT_CONFIG as RNNMARWIL_CONFIG,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict

# fmt: off
# __sphinx_doc_begin__
BC_DEFAULT_CONFIG = RNNMARWILTrainer.merge_trainer_configs(
    RNNMARWIL_CONFIG, {
        # No need to calculate advantages (or do anything else with the
        # rewards).
        "beta": 0.0,
        # Advantages (calculated during postprocessing) not important for
        # behavioral cloning.
        "postprocess_inputs": False,
        # No reward estimation.
        "input_evaluation": [],
    })
# __sphinx_doc_end__
# fmt: on


def validate_config(config: TrainerConfigDict) -> None:
    if config["beta"] != 0.0:
        raise ValueError(
            "For behavioral cloning, `beta` parameter must be 0.0!")

class RNNBCTrainer(RNNMARWILTrainer):
    """Behavioral Cloning (derived from MARWIL).

    Simply uses the MARWIL agent with beta force-set to 0.0.
    """

    @classmethod
    @override(RNNMARWILTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return BC_DEFAULT_CONFIG

    @override(RNNMARWILTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["beta"] != 0.0:
            raise ValueError("For behavioral cloning, `beta` parameter must be 0.0!")
