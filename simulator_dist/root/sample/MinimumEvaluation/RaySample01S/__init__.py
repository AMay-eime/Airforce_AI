# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#ray.RLlibを用いた学習サンプルで学習したモデルの登録方法例
import os
import json
import glob
import ASRCAISim1
from ASRCAISim1.addons.rayUtility.extension.evaluation import StandaloneRayPolicy

#①Agentクラスオブジェクトを返す関数を定義
"""
以下はサンプルのAgentクラスを借りてくる場合の例
"""
def getUserAgentClass(args=None):
    from OriginalModelSample import R4AgentSample01S
    return R4AgentSample01S

#②Agentモデル登録用にmodelConfigを表すjsonを返す関数を定義
"""
なお、modelConfigとは、Agentクラスのコンストラクタに与えられる二つのjson(dict)のうちの一つであり、設定ファイルにおいて
{
    "Factory":{
        "Agent":{
            "modelName":{
                "class":"className",
                "config":{...}
            }
        }
    }
}の"config"の部分に記載される{...}のdictが該当する。
"""    
def getUserAgentModelConfig(args=None):
    return json.load(open(os.path.join(os.path.dirname(__file__),"agent_config.json"),"r"))

#③Agentの種類(一つのAgentインスタンスで1機を操作するのか、陣営全体を操作するのか)を返す関数を定義
"""AgentがAssetとの紐付けに使用するportの名称は本来任意であるが、
　簡単のために1機を操作する場合は"0"、陣営全体を操作する場合は"0"〜"機数-1"で固定とする。
"""
def isUserAgentSingleAsset(args=None):
    #1機だけならばTrue,陣営全体ならばFalseを返すこと。
    return True

#④StandalonePolicyを返す関数を定義
def getUserPolicy(args=None):
    from ASRCAISim1.addons.rayUtility.extension.agents.ppo.my_appo import MyAPPOTrainer, DEFAULT_CONFIG
    from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
    from OriginalModelSample.R4TorchNNSampleForRay import R4TorchNNSampleForRay
    import gym
    tc=MyAPPOTrainer.merge_trainer_configs(
        DEFAULT_CONFIG,
        json.load(open(os.path.join(os.path.dirname(__file__),"trainer_config.json"),"r"))
    )
    policyClass=AsyncPPOTorchPolicy
    weightPath=None
    if(args is not None):
        weightPath=args.get("weightPath",None)
    if(weightPath is None):
        cwdWeights=glob.glob(os.path.join(os.path.dirname(__file__),"*.dat"))
        weightPath=cwdWeights[0] if len(cwdWeights)>0 else None
    else:
        weightPath=os.path.join(os.path.dirname(__file__),weightPath)
    policyConfig={
        "trainer_config":tc,
        "policy_class":policyClass,
        "policy_spec_config":{},
        "weight":weightPath
    }
    return StandaloneRayPolicy("my_policy",policyConfig,False,False)

