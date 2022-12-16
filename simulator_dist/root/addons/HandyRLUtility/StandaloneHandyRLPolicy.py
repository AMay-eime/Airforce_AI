# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import numpy as np
import torch
from ASRCAISim1.policy.StandalonePolicy import StandalonePolicy
from ASRCAISim1.addons.HandyRLUtility.util import map_r
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass

class StandaloneHandyRLPolicy(StandalonePolicy):
	"""強化学習フレームワークから独立させてPolicyを使用するためのインターフェース。
	Environmentのresetとstepの呼び出しに合わせてこのクラスのresetとstepが呼ばれるものとする。
	"""
	def __init__(self,model_class,model_config,model_weight, actionDistributionClassGetter=None, isDeterministic=True):
		self.model=None
		self.model_class=model_class
		self.model_config=model_config
		self.model_weight=model_weight
		self.actionDistributionClassGetter = actionDistributionClassGetter or getActionDistributionClass
		self.isDeterministic=isDeterministic
		self.hidden={}
	def reset(self):
		"""Episodeの初期化処理を記述する。複数のAgentで共用する場合も1回のみ呼ばれる。特定のAgentに依存する処理が必要な場合はstep関数にて実施するものとする。

			Args:
				agentFullName (str): 計算対象のAgentの完全な名称であり、agentName:modelName:policyNameの形をとる。
		"""
		self.hidden={}
	def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
		"""observation等からactionを計算する処理を記述する。

			Args:
				observation: Environmentから得られたobservation。
				reward: Environmentから得られたreward。
				done: Environmentから得られたdone。
				info: Environmentから得られたinfo。
				agentFullName (str): 計算対象のAgentの完全な名称であり、agentName:modelName:policyNameの形をとる。
				observation_space (gym.space.Space): 与えられたObservationのSpace。
				action_space (gym.space.Space): 計算すべきActionのSpace。
		"""
		if(done):
			return None
		if(self.model is None):
			self.model=self.model_class(
				observation_space,
				action_space,
				self.actionDistributionClassGetter(action_space),
				self.model_config
			)
			if(self.model_weight is not None):
				self.model.load_state_dict(torch.load(self.model_weight, map_location=torch.device('cpu')), strict=True)
			self.model.eval()
		hidden=self.hidden.get(agentFullName,self.model.init_hidden())
		observation = map_r(observation, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
		hidden = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)
		outputs=map_r(self.model(observation,hidden), lambda o: o.detach().numpy().squeeze(0))
		self.hidden[agentFullName]=outputs.get('hidden',None)
		dist=self.model.action_dist_class(action_space, outputs['policy'])
		action=dist.sample(self.isDeterministic)
		return action
