"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

extention of RLlib's MARWIL for RNN policy
1. Adds RNN states into SampleBatch as postprocess.
2. Fixes memory leak in torch version reported in the issue #21291 but not merged as of ray 1.13.
"""
import numpy as np
from typing import Optional, Dict

import ray
from ray.rllib.agents.marwil.marwil_tf_policy import postprocess_advantages
from ray.rllib.agents.marwil.marwil_tf_policy import MARWILTFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.evaluation.collectors.simple_list_collector import _AgentCollector
import ASRCAISim1

def postprocess_for_rnn(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[PolicyID, SampleBatch]] = None,
        episode=None) -> SampleBatch:
    """Add seq_lens and states into the offline trajectory data without RNN states.

    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.

    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """
    if policy.is_recurrent() and not "state_in_0" in sample_batch:
        count = sample_batch.count
        seq_lens = []
        max_seq_len = policy.config["model"]["max_seq_len"]
        while count > 0:
            seq_lens.append(min(count, max_seq_len))
            count -= max_seq_len
        sample_batch["seq_lens"] = np.array(seq_lens)
        sample_batch.max_seq_len = max_seq_len
        dummy=policy._get_dummy_batch_from_view_requirements(len(seq_lens))
        num_states=len([k for k in dummy.keys() if k.startswith("state_in")])
        for i in range(num_states):
            sample_batch["state_in_{}".format(i)]=dummy["state_in_{}".format(i)]
    return sample_batch
    
def postprocess_all(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
        sample_batch=postprocess_for_rnn(policy,sample_batch,other_agent_batches,episode)
        return postprocess_advantages(policy,sample_batch,other_agent_batches,episode)

RNNMARWILTFPolicy = MARWILTFPolicy.with_updates(
    name="RNNMARWILTFPolicy",
    get_default_config=lambda: ASRCAISim1.addons.rayUtility.extension.agents.marwil.rnn_marwil.DEFAULT_CONFIG,
    postprocess_fn=postprocess_all
)