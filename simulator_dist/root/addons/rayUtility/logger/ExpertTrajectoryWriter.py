# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
import sys,os,time
import numpy as np
from datetime import datetime
from ASRCAISim1.libCore import *
from ASRCAISim1.addons.rayUtility.extension.policy import DummyInternalRayPolicy
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.evaluation.collectors.simple_list_collector import _AgentCollector
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_writer import JsonWriter

class ExpertTrajectoryWriter(Callback):
	"""異なるobservation,action形式を持つAgentを摸倣する際のTrajectory記録用のLogger。
	RLlibのSampleBatchBuilderを使用する。
	Managerのconfigで指定することを要求する。
	"""
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		self.prefix=getValueFromJsonKRD(self.modelConfig,"prefix",self.randomGen,"")
		self.episodeCounter=-1
		self.writers={}
		self.view_reqs={}
		self.agentCollectors={}
		self.preprocessors={}
		self.dones={}
	def makeWriter(self,agent_index,identifier,observation_space,action_space):
		if(len(self.prefix)==0 or self.prefix[-1]=='/'):
			path=self.prefix+identifier
		else:
			path=self.prefix+"_"+identifier
		os.makedirs(os.path.dirname(path),exist_ok=True)
		ioctx=IOContext(
			log_dir=path,
			config={},
			worker_index=self.manager.worker_index
		)
		self.writers[identifier]=JsonWriter(path,ioctx)
		self.preprocessors[identifier]=get_preprocessor(observation_space)(observation_space)
		self.view_reqs[identifier]={
			SampleBatch.OBS: ViewRequirement(space=self.preprocessors[identifier].observation_space),
			SampleBatch.NEXT_OBS: ViewRequirement(
				data_col=SampleBatch.OBS,
				shift=1,
				space=self.preprocessors[identifier].observation_space),
			SampleBatch.ACTIONS:ViewRequirement(
				space=action_space, used_for_compute_actions=False),
			#SampleBatch.PREV_ACTIONS: ViewRequirement(
			#	data_col=SampleBatch.ACTIONS,
			#	shift=-1,
			#	space=action_space),
			SampleBatch.REWARDS: ViewRequirement(),
			SampleBatch.PREV_REWARDS: ViewRequirement(
				data_col=SampleBatch.REWARDS, shift=-1),
			SampleBatch.DONES: ViewRequirement(),
			SampleBatch.ENV_ID: ViewRequirement(),
			SampleBatch.EPS_ID: ViewRequirement(),
			SampleBatch.UNROLL_ID: ViewRequirement(),
			SampleBatch.AGENT_INDEX: ViewRequirement(),
			SampleBatch.ACTION_PROB: ViewRequirement(),
			SampleBatch.ACTION_LOGP: ViewRequirement(),
			"t": ViewRequirement(),
		}
	def parseInitialObs(self,agent_index,identifier,observation_space,action_space,obs):
		dummy=DummyInternalRayPolicy(
			observation_space,
			action_space,
			{}
		)
		self.agentCollectors[identifier]=_AgentCollector(self.view_reqs[identifier],dummy)
		self.agentCollectors[identifier].add_init_obs(
			self.episodeCounter,
			agent_index,
			self.manager.vector_index,
			-1,
			self.preprocessors[identifier].transform(obs)
		)
		self.dones[identifier]=False
	def addStepData(self,agent_index,identifier,obs,action,reward):
		self.agentCollectors[identifier].add_action_reward_next_obs(
			{
				"t":self.manager.getTickCount()//self.manager.getAgentInterval(),
				"env_id":self.manager.vector_index,
				SampleBatch.AGENT_INDEX:agent_index,
				SampleBatch.ACTIONS:action,
				SampleBatch.REWARDS:reward,
				SampleBatch.DONES:self.dones[identifier],
				SampleBatch.NEXT_OBS:self.preprocessors[identifier].transform(obs),
				SampleBatch.ACTION_PROB:1.0,
				SampleBatch.ACTION_LOGP:0.0
			}
		)
	def onEpisodeBegin(self):
		self.episodeCounter+=1
		agent_index=0
		for agent in self.manager.experts.values():
			agent=agent()
			identifier_base=agent.trajectoryIdentifier
			if(isinstance(agent.imitator,SimpleMultiPortCombiner)):
				childrenKeys=list(agent.imitator.children.keys())
				observation_space=self.manager.observation_space[agent.getFullName()][0]
				action_space=self.manager.action_space[agent.getFullName()][0]
				obs=agent.imitatorObs
				for idx in range(len(childrenKeys)):
					identifier=identifier_base+str(idx+1)
					key=childrenKeys[idx]
					if(not identifier in self.writers):
						self.makeWriter(
							agent_index,
							identifier,
							observation_space[key],
							action_space[key]
						)
					self.parseInitialObs(agent_index,identifier,observation_space[key],action_space[key],obs[key])
					agent_index+=1
				self.dones[identifier_base]=False
			else:
				identifier=identifier_base
				observation_space=self.manager.observation_space[agent.getFullName()][0]
				action_space=self.manager.action_space[agent.getFullName()][0]
				obs=agent.imitatorObs
				if(not identifier in self.writers):
					self.makeWriter(
						agent_index,
						identifier,
						observation_space,
						action_space
					)
				self.parseInitialObs(agent_index,identifier,observation_space,action_space,obs)
				agent_index+=1
	def onStepEnd(self):
		agent_index=0
		for agent in self.manager.experts.values():
			agent=agent()
			identifier_base=agent.trajectoryIdentifier
			if(isinstance(agent.imitator,SimpleMultiPortCombiner)):
				childrenKeys=list(agent.imitator.children.keys())
				done=self.manager.dones[agent.getFullName()]
				if(not self.dones[identifier_base]):
					self.dones[identifier_base]=done
					obs=agent.imitatorObs
					action=agent.imitatorAction
					reward=self.manager.rewards[agent.getFullName()]
					for idx in range(len(childrenKeys)):
						identifier=identifier_base+str(idx+1)
						key=childrenKeys[idx]
						if(not self.dones[identifier]):
							self.dones[identifier]=done or not obs["isAlive"][key]
							self.addStepData(agent_index,identifier,obs[key],action[key],reward/len(childrenKeys))
						agent_index+=1
				else:
					agent_index+=len(childrenKeys)
			else:
				identifier=identifier_base
				done=self.manager.dones[agent.getFullName()]
				if(not self.dones[identifier]):
					self.dones[identifier]=done
					obs=agent.imitatorObs
					action=agent.imitatorAction
					reward=self.manager.rewards[agent.getFullName()]
					self.addStepData(agent_index,identifier,obs,action,reward)
				agent_index+=1
	def onEpisodeEnd(self):
		for identifier in self.writers:
			self.writers[identifier].write(self.agentCollectors[identifier].build(self.view_reqs[identifier]))