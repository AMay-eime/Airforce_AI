# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import random
import itertools
import copy

from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util import map_r
from ...environment import BaseEnvironment
from ASRCAISim1.addons.HandyRLUtility.distribution import getDefaultLegalActions

from ASRCAISim1.libCore import ExpertWrapper, SimpleMultiPortCombiner
from ASRCAISim1.GymManager import GymManager, getDefaultPolicyMapper
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import wrapEnvForBVRMatchMaking, managerConfigReplacer

def detach(src):
    return map_r(src, copy.deepcopy)

class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        original_overrider=args["env_config"].get("overrider",lambda c,w,v:c)
        def overrider(config,worker_index,vector_index):
            nw=1024
            if("seed" in config["Manager"]):
                config["Manager"]["seed"]=config["Manager"]["seed"]+worker_index+nw*vector_index
            config=original_overrider(config,worker_index,vector_index)
            return config
        self.env_config=args["env_config"]
        context = copy.deepcopy(self.env_config)
        context.update({
            "overrider": overrider,
            "worker_index": args.get("id",-1) + 1,
            "vector_index": 0
        })
        self.env = wrapEnvForBVRMatchMaking(GymManager)(context)
        self.completeEnvConfig={
            key:value for key,value in self.env_config.items() if key!="config"
        }
        self.completeEnvConfig["config"]={
                "Manager":self.env.getManagerConfig()(),
                "Factory":self.env.getFactoryModelConfig()()
            }
        self.policy_config = args["policy_config"]
        self._policySetup()
    def reset(self, args={}):
        obs = self.env.reset()
        self.agentFullNames=[key for key in obs.keys()]
        self.agentConfig={}
        idx=0
        for key in self.agentFullNames:
            agent=self.env.manager.getAgent(key)()
            policyName=self.policyMapper(key)
            policy_config=self.policy_config[self.policy_base[policyName]]
            self.agentConfig[key]={
                "team":agent.getTeam(),
                "isExpertWrapper":isinstance(agent,ExpertWrapper),
                "isSimpleMultiPortCombiner":{"Expert":False,"Imitator":False},
                "splitExpert":False,
                "expertChildrenKeys":[],
                "splitImitator":False,
                "imitatorChildrenKeys":[],
                "policyName":policyName,
                "playerIndex":[idx],
                "imitatorIndex":[]
            }
            if(self.agentConfig[key]["isExpertWrapper"]):
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["splitExpert"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"] and not policy_config.get("multi_port",False)
                self.agentConfig[key]["splitImitator"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"] and policy_config.get("split_imitator",False)
                if(self.agentConfig[key]["splitExpert"]):
                    self.agentConfig[key]["playerIndex"]=list(range(idx,idx+len(agent.expert.children)))
                    self.agentConfig[key]["expertChildrenKeys"]=list(agent.expert.children.keys())
            idx+=len(self.agentConfig[key]["playerIndex"])
        for key in self.agentFullNames:
            agent=self.env.manager.getAgent(key)()
            if(self.agentConfig[key]["isExpertWrapper"]):
                if(self.agentConfig[key]["splitImitator"]):
                    self.agentConfig[key]["imitatorIndex"]=list(range(idx,idx+len(agent.imitator.children)))
                    self.agentConfig[key]["imitatorChildrenKeys"]=list(agent.imitator.children.keys())
                else:
                    self.agentConfig[key]["imitatorIndex"]=[idx]
            idx+=len(self.agentConfig[key]["imitatorIndex"])
        self.player_list = sum([
            [(k,i) for i in range(len(c["playerIndex"]))] for k,c in self.agentConfig.items()
        ],[])
        self.num_actual_players =len(self.player_list)
        self.policy_map = [self.agentConfig[key]["policyName"] for key,idx in self.player_list]
        self.teams = [self.agentConfig[key]["team"] for key,idx in self.player_list]
        #imitatorの付加
        self.imitator_list = sum([
            [(k,i) for i in range(len(c["imitatorIndex"]))] for k,c in self.agentConfig.items()
        ],[])
        self.player_list.extend(self.imitator_list)
        self.policy_map.extend(["Imitator" for i in self.imitator_list])
        self.teams.extend([self.agentConfig[key]["team"] for key,idx in self.imitator_list])
        self._parse_obs_dones(obs,self.env.manager.dones)
    def _parse_obs_dones(self,obs,dones):
        self.dones = {}
        for key, d in dones.items():
            if(key=="__all__"):
                self.all_done=d
            else:
                c=self.agentConfig[key]
                for idx in c["playerIndex"]+c["imitatorIndex"]:
                    self.dones[idx]=d
        self.last_obs={}
        for key, o in obs.items():
            c=self.agentConfig[key]
            if(c["isExpertWrapper"]):
                imObs=o[0]
                if(self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]):
                    imIsAlive=imObs.pop("isAlive")
                if(c["splitImitator"]):
                    for idx,sub in zip(c["imitatorIndex"],c["imitatorChildrenKeys"]):
                        self.last_obs[idx]=imObs[sub]
                        self.dones[idx]=self.dones[idx] or not imIsAlive[sub]
                else:
                    self.last_obs[c["imitatorIndex"][0]]=imObs
                exObs=o[1]
                if(self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]):
                    exIsAlive=exObs.pop("isAlive")
                if(c["splitExpert"]):
                    for idx,sub in zip(c["playerIndex"],c["expertChildrenKeys"]):
                        self.last_obs[idx]=exObs[sub]
                        self.dones[idx]=self.dones[idx] or not exIsAlive[sub]
                else:
                    self.last_obs[c["playerIndex"][0]]=exObs
            else:
                self.last_obs[c["playerIndex"][0]]=o
    def _parse_rewards(self,rewards):
        self.rewards={}
        for key, r in rewards.items():
            c=self.agentConfig[key]
            for idx in c["playerIndex"]:
                self.rewards[idx]=r/len(c["playerIndex"])
            for idx in c["imitatorIndex"]:
                self.rewards[idx]=r/len(c["imitatorIndex"])
    def _gather_actions(self,actions):
        ret={}
        for idx, action in actions.items():
            key,_ = self.player_list[idx]
            c=self.agentConfig[key]
            ret[key]=action
            if(c["splitExpert"]):
                for ck, sub in zip(c["expertChildrenKeys"],action):
                    ret[key][ck] = sub
        return ret
    def step(self, actions):
        try:
            obs, rewards, dones, infos = self.env.step(self._gather_actions(actions))
        except Exception as e:
            print(self.env.manager.getTime(),",",actions)
            raise e
        self._parse_rewards(rewards)
        self._parse_obs_dones(obs,dones)

    def turns(self):
        return [idx for idx in range(self.num_actual_players) if not self.dones[idx]]

    def terminal(self):
        return self.all_done

    def reward(self):
        return {idx: self.rewards.get(idx,0.0) for idx, key in enumerate(self.player_list)}

    def outcome(self):
        winner = self.env.manager.getRuler()().winner
        return {idx: +1 if self.teams[idx] == winner else -1
            for idx, key in enumerate(self.player_list)
        }
    def score(self):
        scores = self.env.manager.scores
        return {idx: scores[self.teams[idx]]
            for idx, key in enumerate(self.player_list)
        }

    def legal_actions(self, player):
        ac_space=self.action_space(player)
        return getDefaultLegalActions(ac_space)

    def players(self):
        #maximum number of players
        return list(range(len(self.player_list)))

    def observation(self, player=None):
        return self.last_obs[player]

    def net(self, policyName):
        return self.policy_config[policyName]["model_class"]

    #ASRC
    def _observation_space_internal(self, manager, agentFullName, asImitator=False, splitIndex=0):
        space=manager.get_observation_space()[agentFullName]
        c=self.agentConfig[agentFullName]
        if(c["isExpertWrapper"]):
            if(asImitator):
                if(c["splitImitator"]):
                    return space[0][c["imitatorChildrenKeys"][splitIndex]]
                else:
                    return space[0]
            else:
                if(c["splitExpert"]):
                    return space[1][c["expertChildrenKeys"][splitIndex]]
                else:
                    return space[1]
        else:
            return space
    def _action_space_internal(self, manager, agentFullName, asImitator=False, splitIndex=0):
        space=manager.get_action_space()[agentFullName]
        c=self.agentConfig[agentFullName]
        if(c["isExpertWrapper"]):
            if(asImitator):
                if(c["splitImitator"]):
                    return space[0][c["imitatorChildrenKeys"][splitIndex]]
                else:
                    return space[0]
            else:
                if(c["splitExpert"]):
                    return space[1][c["expertChildrenKeys"][splitIndex]]
                else:
                    return space[1]
        else:
            return space
    def observation_space(self, player):
        asImitator = player >= self.num_actual_players
        agentFullName, splitIndex = self.player_list[player]
        return self._observation_space_internal(self.env.manager, agentFullName, asImitator, splitIndex)
    def action_space(self, player):
        asImitator = player >= self.num_actual_players
        agentFullName, splitIndex = self.player_list[player]
        return self._action_space_internal(self.env.manager, agentFullName, asImitator, splitIndex)
    def setMatch(self,matchInfo):
        self.policy_base={
            info["Policy"]+info["Suffix"]: info["Policy"]
            for info in matchInfo.values()
        }
        self.env.setMatch(matchInfo)
    def _policySetup(self):
        for policyName,policy_config in self.policy_config.items():
            dummyContext = copy.deepcopy(self.completeEnvConfig)
            dummyContext["config"]["Manager"]["Viewer"] = "None"
            dummyContext["config"]["Manager"]["Loggers"] = {}
            isMultiPort = policy_config.get("multi_port",False) is True
            matchInfo={
                "Blue":{"Policy":policyName,"Suffix":"","Weight":-1,"MultiPort":isMultiPort},
                "Red":{"Policy":policyName,"Suffix":"","Weight":-1,"MultiPort":isMultiPort}
            }
            dummyContext["config"]["Manager"]=managerConfigReplacer(dummyContext["config"]["Manager"],matchInfo)
            dummyContext.update({
                "worker_index": -1,
                "vector_index": -1,
            })
            dummyEnv = GymManager(dummyContext)
            dummyEnv.reset()

            ac=dummyEnv.get_action_space()
            key=next(iter(ac))
            _,m_name,p_name=key.split(":")
            if(p_name==policyName):
                pass
            elif(p_name=="Internal"):
                pass
            else:
                raise ValueError("Invalid policy config.")
            agent=dummyEnv.manager.getAgent(key)()
            self.agentConfig={
                key:{
                    "isExpertWrapper":isinstance(agent,ExpertWrapper),
                    "isSimpleMultiPortCombiner":{"Expert":False,"Imitator":False},
                }
            }
            if(self.agentConfig[key]["isExpertWrapper"]):
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"]=isinstance(agent.expert,SimpleMultiPortCombiner)
                self.agentConfig[key]["splitExpert"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Expert"] and not isMultiPort
                if(self.agentConfig[key]["splitExpert"]):
                    self.agentConfig[key]["expertChildrenKeys"]=list(agent.expert.children.keys())
                self.agentConfig[key]["splitImitator"]=self.agentConfig[key]["isSimpleMultiPortCombiner"]["Imitator"] and policy_config.get("split_imitator",False)
                if(self.agentConfig[key]["splitImitator"]):
                    self.agentConfig[key]["imitatorChildrenKeys"]=list(agent.imitator.children.keys())
            self.policy_config[policyName]["observation_space"]=self._observation_space_internal(dummyEnv.manager, key, False, 0)
            self.policy_config[policyName]["action_space"]=self._action_space_internal(dummyEnv.manager, key, False, 0)
    def policyMapper(self,agentId):
        """
        エージェントのfullNameから対応するポリシー名を抽出する関数。
        agentId=agentName:modelName:policyName
        """
        agentName,modelName,policyName=agentId.split(":")
        if(agentName.startswith("Blue")):
            info=self.env.matchInfo["Blue"]
        else:
            info=self.env.matchInfo["Red"]
        return info["Policy"]+info["Suffix"]
    def getImitationInfo(self):
        """ExpertWrapperを使用しているAgentについて、模倣する側の情報を返す。
        """
        ret={}
        for key,idx in self.imitator_list:
            agent=self.env.manager.getAgent(key)()
            c=self.agentConfig[key]
            if(not self.dones[c["imitatorIndex"][idx]]):
                imObs=detach(self.last_obs[c["imitatorIndex"][idx]])
                if(c["splitImitator"]):
                    imAction=detach(agent.imitatorAction[c["imitatorChildrenKeys"][idx]])
                else:
                    imAction=detach(agent.imitatorAction)
                ret[c["imitatorIndex"][idx]]={
                    "observation": imObs,
                    "action": imAction
                }
        return ret
