# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""
ray RLlibでMatchMaker/MatchMonitorを使用するための追加機能。
rayのTrainerクラスに対するCallbackを利用してMatchMaker/MatchMonitorの処理を割り込ませる方式としている。
また、MatchMakerクラスは学習プログラム側でray.remoteを用いてリモートインスタンスとして生成することを前提としている。
"""
from math import *
import sys,os,time
import pickle
import numpy as np
import datetime
import copy
import cloudpickle
from collections import OrderedDict,defaultdict
from ASRCAISim1.libCore import Callback,Fighter,getValueFromJsonKRD
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ASRCAISim1.addons.MatchMaker.PayOff import PayOff
from ASRCAISim1.addons.MatchMaker.util import managerConfigReplacer
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import BVRMatchMaker,BVRMatchMonitor
from ASRCAISim1.addons.rayUtility.utility.common import loadWeights,saveWeights

def loadInitialWeight(policyName,policy_config):
    """指定したpolicyNameに対する初期重みの読み込みを行う。
    """
    if(policyName in policy_config):
        initial_weight_path=config.get("initial_weight",None)
        if(initial_weight_path is not None):
            return loadWeights(initial_weight_path)
    return None

def loadInitialWeights(policy_config):
    """初期重みの読み込みを行う。
    """
    initial_weights={}
    for policyName,config in policy_config.items():
        initial_weight_path=config.get("initial_weight",None)
        if(initial_weight_path is not None):
            initial_weights[policyName]=loadWeights(initial_weight_path)
    return initial_weights

def populate(weight_pool, policyName, weight, weight_id):
    """重みをweight_poolに追加する。
    """
    dstPath = os.path.join(weight_pool, policyName+"-"+str(weight_id)+".dat")
    saveWeights(weight,dstPath)
    print("=====weight populated===== ",policyName," -> ",os.path.basename(dstPath))

class MatchMakerOnRayBase(DefaultCallbacks):
    """rayのCallbackとしてTrainer側でPolicyの重みを読み書きするクラス。
    rayのtrainer_configにこれをそのまま登録することはせず、
    createMatchMakerOnRayForEachTrainerによりTrainerの名称と紐付けたサブクラスを登録するものとする。
    """
    def __init__(self):
        super().__init__()
        self.currentWeights=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:-1)))
        self.counter=0
        self.trainerName="Learner"
        self.resetRequested=False
        self.resetWeights={}
        self.numWeightsPerPolicy=100 #メモリに保持しておく重みの数(ポリシー毎)
        self.weightBuffer=defaultdict(lambda:{})
        self.trainingWeightCount=defaultdict(lambda:-1)
        self.matchMaker=ray.get_actor("MatchMaker")
        self.matchMonitors={}
        self.weight_pool=ray.get(self.matchMaker.get_weight_pool.remote())
        self.policy_config=ray.get(self.matchMaker.get_policy_config.remote())
    def makeNextMatch(self,vector_index):
        matchType=self.trainerName
        matchInfo=ray.get(self.matchMaker.makeNextMatch.remote(matchType))
        for team in matchInfo:
            if(matchInfo[team]["Suffix"]!="" and vector_index > 0):
                matchInfo[team]["Suffix"]+="_{}".format(vector_index)
        return matchInfo
    def on_sub_environment_created(
        self,
        *,
        worker,
        sub_environment,
        env_context,
        **kwargs,) -> None:
        """初回の対戦カードを設定する。
        """
        sub_environment.setMatch(self.makeNextMatch(sub_environment.vector_index))
        sub_environment.reset()
    def on_episode_start(self,
                        *,
                        worker,
                        base_env,
                        policies,
                        episode,
                        env_index = None,
                        **kwargs) -> None:
        env=base_env.get_sub_environments()[env_index]
        self.matchMonitors[env_index]=self.matchMonitorClass(env)
        self.matchMonitors[env_index].onEpisodeBegin()
        self.counter+=1
        for team,info in env.matchInfo.items():
            policyBase=info["Policy"]
            policy=policyBase+info["Suffix"]
            weight_id=info["Weight"]
            if(weight_id>=0 and weight_id!=self.currentWeights[policy][env.worker_index][env.vector_index]):
                self.loadWeights(policies[policy], policyBase, weight_id)
                self.currentWeights[policy][env.worker_index][env.vector_index]=weight_id
    def loadWeights(self, policy, policyBase, weight_id):
        """重みを読み込む
        """
        if(self.policy_config[policyBase]["is_internal"]):
            return
        if(weight_id==0):
            #weight_id==0 : その時点の学習中重みのコピー
            count=ray.get(self.matchMaker.getTrainingWeightUpdateCount.remote(policyBase))
            if(count>self.trainingWeightCount[policyBase]):
                self.trainingWeightCount[policyBase]=count
                latest=ray.get(self.matchMaker.getLatestTrainingWeight.remote(policyBase))
                self.weightBuffer[policyBase][weight_id]=latest
            policy.set_weights(self.weightBuffer[policyBase][weight_id])
        elif(weight_id>0):
            #weight_id>0 : 過去のある時点の重み
            if(weight_id in self.weightBuffer[policyBase]):
                weight=self.weightBuffer[policyBase].pop(weight_id)
            else:
                path=os.path.join(self.weight_pool,policyBase+"-"+str(weight_id)+".dat")
                weight=loadWeights(path)
            self.weightBuffer[policyBase][weight_id]=weight
            policy.set_weights(weight)
            if(len(self.weightBuffer[policyBase])>self.numWeightsPerPolicy):
                #メモリに保持している重みが上限を超えたら直近の読み込み時刻が最も古いもの(ただし、0を除く)を削除する。
                it=iter(self.weightBuffer[policyBase].keys())
                removed=next(it)
                if(removed==0):
                    removed=next(it)
                self.weightBuffer[policyBase].pop(removed,None)
    def on_episode_end(self,
                        *,
                        worker,
                        base_env,
                        policies,
                        episode,
                        env_index = None,
                        **kwargs) -> None:
        env=base_env.get_sub_environments()[env_index]
        matchType=self.trainerName
        result=self.matchMonitors[env_index].onEpisodeEnd(matchType)
        populate_config=ray.get(self.matchMaker.onEpisodeEnd.remote(env.matchInfo,result))
        for policyBase,conf in populate_config.items():
            populate(self.weight_pool, policyBase, policies[policyBase].get_weights(), conf['weight_id'])
            if(conf['reset']):
                self.resetRequested=True
                initial_weight=loadInitialWeight(policyBase,self.policy_config)
                if(initial_weight is None):
                    raise ValueError("Weight reset is available only for pretrained policies.")
                self.resetWeights[policyBase]=initial_weight
        customMetrics=ray.get(self.matchMaker.get_metrics.remote(env.matchInfo,result))
        for key,value in customMetrics.items():
            episode.custom_metrics[key]=value
        env.setMatch(self.makeNextMatch(env.vector_index))
    def on_train_result(self, *, trainer: "Trainer", result: dict,
        **kwargs) -> None:
        if(self.resetRequested):
            trainer.set_weights(self.resetWeights)
            trainer.workers.sync_weights(list(self.resetWeights.keys()))
            self.resetRequested=False
            self.resetWeights={}
            
def getMatchMakerOnRayClass(trainerName,matchMonitorClass):
    """MatchMakerOnRayを特定のtrainerNameとMatchMonitorクラスに紐付けたサブクラスを生成する。
    rayのtrainer_configにはこちらで生成したサブクラスを登録するものとする。
    """
    class derived(MatchMakerOnRayBase):
        def __init__(self):
            super().__init__()
            self.trainerName=trainerName
            self.matchMonitorClass=matchMonitorClass
    return derived
