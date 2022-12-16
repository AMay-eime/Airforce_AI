"""
Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

A standalone version of RLlib Policy independent of Trainer.
The code for builing Policy instances is derived from RolloutWorker._build_policy_map
which is defined in ray.rllib.evaluation.rollout_worker.py  as of ray 1.13.
"""
import numpy as np
import gym
import logging
from gym.spaces import Space
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.evaluation.collectors.simple_list_collector import _AgentCollector
from ASRCAISim1.policy.StandalonePolicy import StandalonePolicy
from ASRCAISim1.addons.rayUtility.utility.common import loadPolicyWeights
from ASRCAISim1.addons.rayUtility.extension.evaluation import get_inference_input_dict

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

class StandaloneRayPolicy(StandalonePolicy):
    """Ray.RLLibのPolicyをTrainerから独立して動かすためのクラス。
    """
    def __init__(self,policyName,config,isLocal=False,isDeterministic=False):
        trainer_config=config["trainer_config"]
        self.name=policyName
        self.seed=config.get("seed",None)
        self.policy_class=config["policy_class"]
        self.weight=config.get("weight",None)
        self.policy_spec_config=config.get("policy_spec_config",{})
        self._remote_config=trainer_config
        self._local_config=merge_dicts(
            trainer_config,
            {"tf_session_args": trainer_config["local_tf_session_args"]})
        self.preprocessing_enabled=True
        self.isLocal=isLocal
        self.isDeterministic=isDeterministic
        self.policy=None
        self.collector=None
    def _build(self,policy_spec,policy_config):
        #derived from
        # ray.rllib.evaluation.rollout_worker.RolloutWorker._build_policy_map and
        # ray.rllib.policy.policy_map.PolicyMap.create_policy

        policy_cls,obs_space,act_space,conf=policy_spec
        logger.debug("Creating policy for {}".format(self.name))
        # Update the general policy_config with the specific config
        # for this particular policy.
        merged_conf = merge_dicts(policy_config, conf or {})
        # Update num_workers and worker_index.
        merged_conf["num_workeres"]=0
        merged_conf["worker_index"]=0
        # Preprocessors.
        if self.preprocessing_enabled:
            self.preprocessor = ModelCatalog.get_preprocessor_for_space(
                obs_space, merged_conf.get("model")
            )
            if self.preprocessor is not None:
                obs_space = self.preprocessor.observation_space
            else:
                self.preprocessor = NoPreprocessor(obs_space)
        else:
            #self.preprocessor = None
            self.preprocessor = NoPreprocessor(obs_space)


        # Create the actual policy object. (ray.rllib.policy.policy_map.PolicyMap.create_policy)
        framework = merged_conf.get("framework", "tf")
        class_ = get_tf_eager_cls_if_necessary(policy_cls, merged_conf)
        
        # Tf.
        if framework in ["tf2", "tf", "tfe"]:
            var_scope = self.name + (
                ("_wk" + str(merged_conf["worker_index"])) if merged_conf["worker_index"] else ""
            )

            # For tf static graph, build every policy in its own graph
            # and create a new session for it.
            if framework == "tf":
                with tf1.Graph().as_default():
                    if self.session_creator:
                        sess = self.session_creator()
                    else:
                        sess = tf1.Session(
                            config=tf1.ConfigProto(
                                gpu_options=tf1.GPUOptions(allow_growth=True)
                            )
                        )
                    with sess.as_default():
                        # Set graph-level seed.
                        if self.seed is not None:
                            tf1.set_random_seed(self.seed)
                        with tf1.variable_scope(var_scope):
                            self.policy = class_(
                                obs_space, act_space, merged_conf
                            )
            # For tf-eager: no graph, no session.
            else:
                with tf1.variable_scope(var_scope):
                    self.policy = class_(
                        obs_space, act_space, merged_conf
                    )
        # Non-tf: No graph, no session.
        else:
            class_ = policy_cls
            self.policy = class_(obs_space, act_space, merged_conf)

        logger.info("Built policy: {}".format(self.policy))
        logger.info("Built preprocessor: {}".format(self.preprocessor))
        if(self.weight is not None):
            loadPolicyWeights(self.policy,self.weight)
    def reset(self):
        self.states={}
        self.prev_action={}
        self.prev_rewards={}
        self.agent_index={}
        self.collectors={}
        self.timeStep={}
    def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
        if(done):
            return None
        if(self.policy is None):
            policy_spec=[self.policy_class,observation_space,action_space,self.policy_spec_config]
            if(self.isLocal):
                self._build(policy_spec,self._local_config)
            else:
                self._build(policy_spec,self._remote_config)
        if(not agentFullName in self.agent_index):
            self.agent_index[agentFullName]=len(self.agent_index)
            self.timeStep[agentFullName]=-1
        if(self.timeStep[agentFullName]==-1):
            self.collectors[agentFullName]=_AgentCollector(self.policy.view_requirements,self.policy)
            self.collectors[agentFullName].add_init_obs(
                0,
                self.agent_index[agentFullName],
                0,
                self.timeStep[agentFullName],
                self.preprocessor.transform(observation)
            )
        else:
            values={
                    "t":self.timeStep[agentFullName],
                    "env_id":0,
                    SampleBatch.AGENT_INDEX:self.agent_index[agentFullName],
                    SampleBatch.ACTIONS:self.prev_action[agentFullName],
                    SampleBatch.REWARDS:reward,
                    SampleBatch.DONES:done,
                    SampleBatch.NEXT_OBS:self.preprocessor.transform(observation)
                }
            for i,state in enumerate(self.states[agentFullName]):
                values["state_out_{}".format(i)]=state
            self.collectors[agentFullName].add_action_reward_next_obs(values)
        input_dict=get_inference_input_dict(self.policy.view_requirements, self.collectors[agentFullName])
        action,state_out,info=self.policy.compute_actions_from_input_dict(
            input_dict,
            timestep=self.timeStep[agentFullName],
            explore=not self.isDeterministic
        )
        self.states[agentFullName]=[s[0] for s in state_out]
        self.prev_action[agentFullName]=flatten_to_single_ndarray(action[0])
        self.timeStep[agentFullName]+=1
        return action[0]