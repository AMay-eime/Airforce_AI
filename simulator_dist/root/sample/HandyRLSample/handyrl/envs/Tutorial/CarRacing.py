import random
import itertools
import copy
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...environment import BaseEnvironment
from ASRCAISim1.addons.HandyRLUtility.distribution import getDefaultLegalActions

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.env=gym.make("CarRacing-v0")
        self.env.reset()
        orig=self.env.observation_space
        self.converted_obs_space = gym.spaces.Box(low=np.transpose(orig.low/255.0,[2,0,1]), high=np.transpose(orig.high/255.0,[2,0,1]),shape=np.array(orig.shape)[[2,0,1]])
        self.policy_config = args["policy_config"]
        for policyName in self.policy_config:
            self.policy_config[policyName]["observation_space"]=self.converted_obs_space
            self.policy_config[policyName]["action_space"]=self.env.action_space
        self.all_done=False
        self.need_render = args.get('render',False)
        self.maxStep = args.get('max_step',-1)
        self.outcome_type = args.get('outcome_type','Sum')
        assert(self.outcome_type in ["Sum","Mean"])
        self.outcome_scale = args.get('outcome_scale',1.0)
        self.reward_scale = args.get('reward_scale',1.0)
        self.reward_at_max_step = args.get('reward_at_max_step',0)
        self.outcome_at_max_step = args.get('outcome_at_max_step',0)
        self.reward_per_step = args.get('reward_per_step',0)
        self.outcome_per_step = args.get('outcome_per_step',0)
        self.outcome_offset = args.get('outcome_offset',0)
        self.score_type = args.get('score_type','outcome')
    def reset(self, args={}):
        obs = self.env.reset()
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, [2,0,1])
        self.policy_map = [next(iter(self.policy_base.keys()))]
        self.last_obs = {0: obs}
        self.total_rewards = {0: 0.0}
        self.all_done=False
        self.stepCount=0
        if self.need_render:
            self.env.render()
    def step(self, actions):
        self.last_action=actions
        obs, rewards, dones, infos = self.env.step(actions[0])
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, [2,0,1])
        self.all_done = dones or (self.maxStep>0 and self.stepCount>=self.maxStep)
        if self.maxStep>0 and self.stepCount>=self.maxStep:
            rewards += self.reward_at_max_step
        rewards += self.reward_per_step
        self.rewards = {0: rewards*self.reward_scale}
        self.total_rewards[0]+=rewards*self.reward_scale
        self.last_obs = {0: obs}
        self.stepCount+=1
        if self.need_render:
            self.env.render()

    def turns(self):
        return [0]

    def terminal(self):
        return self.all_done

    def reward(self):
        return {0: self.rewards.get(0,0.0)}

    def outcome(self):
        o=self.total_rewards.get(0,0.0)
        if(self.outcome_type=="Mean"):
            o/=max(1,self.stepCount)
        o += self.outcome_offset
        if self.maxStep>0 and self.stepCount>=self.maxStep:
            o += self.outcome_at_max_step
        o += self.outcome_per_step * self.stepCount
        return {0: o*self.outcome_scale}

    def score(self):
        if(self.score_type == 'reward_sum'):
            return {0: self.total_rewards.get(0,0.0)}
        elif(self.score_type == 'reward_mean'):
            return {0: self.total_rewards.get(0,0.0)/max(1,self.stepCount)}
        else: #'outcome'
            return self.outcome()

    def legal_actions(self, player):
        ac_space=self.action_space(player)
        return getDefaultLegalActions(ac_space)

    def players(self):
        return [0]

    def observation(self, player=None):
        return self.last_obs[player]

    def net(self, policyName):
        return self.policy_config[policyName]["model_class"]

    def observation_space(self, player):
        return self.converted_obs_space
    def action_space(self, player):
        return self.env.action_space
    def setMatch(self,matchInfo):
        self.policy_base={
            info["Policy"]+info["Suffix"]: info["Policy"]
            for info in matchInfo.values()
        }
        self.matchInfo=matchInfo
    def getImitationInfo(self):
        ret={}
        return ret
