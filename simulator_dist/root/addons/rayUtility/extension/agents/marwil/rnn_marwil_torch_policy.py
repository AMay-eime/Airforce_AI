"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

extention of RLlib's MARWIL for RNN policy
1. Adds RNN states into SampleBatch as postprocess.
2. Fixes memory leak in torch version reported in the issue #21291 but not merged as of ray 1.13.
"""
import gym
import numpy as np
from typing import Dict

import ray
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.policy import Policy
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.agents.marwil.marwil_torch_policy import MARWILTorchPolicy
from ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil_tf_policy import postprocess_all
import ASRCAISim1

torch, _ = try_import_torch()

def marwil_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
) -> TensorType:
    model_out, _ = model(train_batch)
    action_dist = dist_class(model_out, model)
    actions = train_batch[SampleBatch.ACTIONS]
    # log\pi_\theta(a|s)
    logprobs = action_dist.logp(actions)

    # Advantage estimation.
    if policy.config["beta"] != 0.0:
        cumulative_rewards = train_batch[Postprocessing.ADVANTAGES]
        state_values = model.value_function()
        adv = cumulative_rewards - state_values
        adv_squared_mean = torch.mean(torch.pow(adv, 2.0))

        explained_var = explained_variance(cumulative_rewards, state_values)
        policy.explained_variance = torch.mean(explained_var)

        # Policy loss.
        # Update averaged advantage norm.
        rate = policy.config["moving_average_sqd_adv_norm_update_rate"]
        policy._moving_average_sqd_adv_norm.add_(
            rate * (adv_squared_mean - policy._moving_average_sqd_adv_norm)
        )

        # ====================================================================
        # Reported in the issue #21291, but not merged as o fray 1.13.
        # detach to get rid of the growing memory occupancy by grad fn chain.
        policy._moving_average_sqd_adv_norm = policy._moving_average_sqd_adv_norm.detach()
        # ====================================================================

        # Exponentially weighted advantages.
        exp_advs = torch.exp(
            policy.config["beta"]
            * (adv / (1e-8 + torch.pow(policy._moving_average_sqd_adv_norm, 0.5)))
        ).detach()
        # Value loss.
        policy.v_loss = 0.5 * adv_squared_mean
    else:
        # Policy loss (simple BC loss term).
        exp_advs = 1.0
        # Value loss.
        policy.v_loss = 0.0

    # logprob loss alone tends to push action distributions to
    # have very low entropy, resulting in worse performance for
    # unfamiliar situations.
    # A scaled logstd loss term encourages stochasticity, thus
    # alleviate the problem to some extent.
    logstd_coeff = policy.config["bc_logstd_coeff"]
    if logstd_coeff > 0.0:
        logstds = torch.mean(action_dist.log_std, dim=1)
    else:
        logstds = 0.0

    policy.p_loss = -torch.mean(exp_advs * (logprobs + logstd_coeff * logstds))

    # Combine both losses.
    policy.total_loss = policy.p_loss + policy.config["vf_coeff"] * policy.v_loss

    return policy.total_loss

RNNMARWILTorchPolicy = MARWILTorchPolicy.with_updates(
    name="RNNMARWILTorchPolicy",
    loss_fn=marwil_loss,
    get_default_config=lambda: ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil.DEFAULT_CONFIG,
    postprocess_fn=postprocess_all
)
