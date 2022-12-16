"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

An independent version of get_in_ference_input_dict,
which is originally a member method of SimpleListCollector in ray.rllib.evaluation.collectors.simple_list_collector.
"""
import numpy as np
import logging
from gym.spaces import Space
import tree
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.evaluation.collectors.simple_list_collector import _AgentCollector

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

logger = logging.getLogger(__name__)

def get_inference_input_dict(view_requirements, collector):
    # derived from SimpleListCollector.get_in_ference_input_dict as of ray 1.13.
    # Independent of Policy instance and callable directly with a ViewRequirement and an _AgentCollector.

    buffer = collector.buffers
    buffer_structs = collector.buffer_structs
    input_dict = {}
    for view_col, view_req in view_requirements.items():
        # Not used for action computations.
        if not view_req.used_for_compute_actions:
            continue

        # Create the batch of data from the different buffers.
        data_col = view_req.data_col or view_col
        delta = (
            -1
            if data_col
            in [
                SampleBatch.OBS,
                SampleBatch.ENV_ID,
                SampleBatch.EPS_ID,
                SampleBatch.AGENT_INDEX,
                SampleBatch.T,
            ]
            else 0
        )
        # Range of shifts, e.g. "-100:0". Note: This includes index 0!
        if view_req.shift_from is not None:
            time_indices = (view_req.shift_from + delta, view_req.shift_to + delta)
        # Single shift (e.g. -1) or list of shifts, e.g. [-4, -1, 0].
        else:
            time_indices = view_req.shift + delta

        # Loop through agents and add up their data (batch).
        data = None
        # Buffer for the data does not exist yet: Create dummy
        # (zero) data.
        if data_col not in buffer:
            if view_req.data_col is not None:
                space = view_requirements[view_req.data_col].space
            else:
                space = view_req.space

            if isinstance(space, Space):
                fill_value = get_dummy_batch_for_space(
                    space,
                    batch_size=0,
                )
            else:
                fill_value = space

            collector._build_buffers({data_col: fill_value})

        if data is None:
            data = [[] for _ in range(len(buffer[data_col]))]

        # `shift_from` and `shift_to` are defined: User wants a
        # view with some time-range.
        if isinstance(time_indices, tuple):
            # `shift_to` == -1: Until the end (including(!) the
            # last item).
            if time_indices[1] == -1:
                for d, b in zip(data, buffer[data_col]):
                    d.append(b[time_indices[0]:])
            # `shift_to` != -1: "Normal" range.
            else:
                for d, b in zip(data, buffer[data_col]):
                    d.append(b[time_indices[0]:time_indices[1] + 1])
        # Single index.
        else:
            for d, b in zip(data, buffer[data_col]):
                d.append(b[time_indices])

        np_data = [np.array(d) for d in data]
        if data_col in buffer_structs:
            input_dict[view_col] = tree.unflatten_as(
                buffer_structs[data_col], np_data)
        else:
            input_dict[view_col] = np_data[0]

    return SampleBatch(input_dict)
