"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

Added "manual_update_for_untrainable_policies" key in the config into APPOTrainer of ray 1.13.0.
Setting True for this key disables broadcasting weights for untrainable policies.
This makes it possible to manage opponents' weights manually for custom match-making with past agents.
"""
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PartialTrainerConfigDict, TrainerConfigDict
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
from ASRCAISim1.addons.rayUtility.extension.agents.impala.my_impala import ManualUpdateForUntrainablesMixin

DEFAULT_CONFIG = APPOTrainer.merge_trainer_configs(
    APPO_DEFAULT_CONFIG,
    {
        #Whether to update weights for untrainable policies manually outside of the trainer or not.
        "manual_update_for_untrainable_policies":True,
    },
    _allow_unknown_configs=True,
)

class MyAPPOTrainer(ManualUpdateForUntrainablesMixin, APPOTrainer):
    @classmethod
    @override(APPOTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG
