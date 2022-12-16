# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os
import sys
import json
import time
import argparse
import importlib
from ASRCAISim1.libCore import Factory
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager, SimpleEvaluator

def agentConfigMaker(team: str, userModuleID: str, isSingle: bool, number: int) -> dict:
    # 青と赤のAgent指定部分のコンフィグを生成
    if(isSingle):
        # 1個のインスタンスで1機なので異なるnameを割り当てることで2個のインスタンスとし、portはどちらも"0"
        return {
            "type": "group",
            "order": "fixed",
            "elements": [
                {"type": "External", "model": "Agent_"+userModuleID, "policy": "Policy_" +
                    userModuleID, "name": team+"_"+userModuleID+"_"+str(i+1), "port": "0"} for i in range(number)
            ]
        }
    else:
        # 1個のインスタンスで2機分なので同じnameを割り当てることで1個のインスタンスとし、それぞれ異なるport("0"と"1")を割り当てる
        return {
            "type": "group",
            "order": "fixed",
            "elements": [
                {"type": "External", "model": "Agent_"+userModuleID,
                    "policy": "Policy_"+userModuleID, "name": team+"_"+userModuleID, "port": str(i)} for i in range(number)
            ]
        }


def run(matchInfo):
    blueUserModuleID=matchInfo["blue"]["userModuleID"]
    redUserModuleID=matchInfo["red"]["userModuleID"]
    blueArgs=matchInfo["blue"].get("args",None)
    redArgs=matchInfo["red"].get("args",None)
    seed=matchInfo.get("seed",None)
    if(seed is None):
        import numpy as np
        seed = np.random.randint(2**31)
    logName=matchInfo["blue"].get("logName",blueUserModuleID)+"_vs_"+matchInfo["red"].get("logName",redUserModuleID)
    # ユーザーモジュールの読み込み
    importedUsers={}
    if(not blueUserModuleID in importedUsers):
        try:
            blueModule = importlib.import_module(blueUserModuleID)
            assert hasattr(blueModule, "getUserAgentClass")
            assert hasattr(blueModule, "getUserAgentModelConfig")
            assert hasattr(blueModule, "isUserAgentSingleAsset")
            assert hasattr(blueModule, "getUserPolicy")
        except Exception as e:
            raise e  # 読み込み失敗時の扱いは要検討
        importedUsers[blueUserModuleID]=blueModule
        blueAgentClass = blueModule.getUserAgentClass(blueArgs)
        addPythonClass("Agent", "Agent_"+blueUserModuleID, blueAgentClass)
        Factory.addDefaultModel(
            "Agent",
            "Agent_"+blueUserModuleID,
            {
                "class": "Agent_"+blueUserModuleID,
                "config": blueModule.getUserAgentModelConfig(blueArgs)
            }
        )
    if(not redUserModuleID in importedUsers):
        try:
            redModule = importlib.import_module(redUserModuleID)
            assert hasattr(redModule, "getUserAgentClass")
            assert hasattr(redModule, "getUserAgentModelConfig")
            assert hasattr(redModule, "isUserAgentSingleAsset")
            assert hasattr(redModule, "getUserPolicy")
        except Exception as e:
            raise e  # 読み込み失敗時の扱いは要検討
        importedUsers[redUserModuleID]=redModule
        redAgentClass = redModule.getUserAgentClass(redArgs)
        addPythonClass("Agent", "Agent_"+redUserModuleID, redAgentClass)
        Factory.addDefaultModel(
            "Agent",
            "Agent_"+redUserModuleID,
            {
                "class": "Agent_"+redUserModuleID,
                "config": redModule.getUserAgentModelConfig(redArgs)
            }
        )
    blueModule = importedUsers[blueUserModuleID]
    redModule = importedUsers[redUserModuleID]
    blueAgentClass = blueModule.getUserAgentClass(blueArgs)
    redAgentClass = redModule.getUserAgentClass(redArgs)

    # コンフィグの生成
    agentConfig = {
        "Manager": {
            "AgentConfigDispatcher": {
                "BlueAgents": agentConfigMaker("Blue",blueUserModuleID, blueModule.isUserAgentSingleAsset(blueArgs),4),
                "RedAgents": agentConfigMaker("Red",redUserModuleID, redModule.isUserAgentSingleAsset(redArgs),4)
            }
        }
    }
    configs = [
        os.path.join(os.path.dirname(__file__), "common/R4_contest_mission_config.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_asset_models.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_agent_ruler_reward_models.json"),
        agentConfig,
        {
            "Manager": {
                "Rewards": [],
                "seed":seed,
                "ViewerType":"God" if matchInfo["visualize"] else "None",
                "Loggers":{
                    "MultiEpisodeLogger":{
                        "class":"MultiEpisodeLogger",
                        "config":{
                            "prefix":os.path.join(matchInfo["log_dir"],logName),
                            "episodeInterval":1,
                            "ratingDenominator":100
                        }
                    },
                    "GodViewStateLogger":{
                        "class":"GodViewStateLogger",
                        "config":{
                            "prefix":os.path.join(matchInfo["log_dir"],logName),
                            "episodeInterval":1,
                            "innerInterval":1
                        }
                    }
                } if "log_dir" in matchInfo else {}
            }
        }
    ]
    context = {
        "config": configs,
        "worker_index": 0,
        "vector_index": 0
    }
    # 環境の生成
    env = GymManager(context)

    # StandalonePolicyの生成
    policies = {
        "Policy_"+blueUserModuleID: blueModule.getUserPolicy(blueArgs),
        "Policy_"+redUserModuleID: redModule.getUserPolicy(redArgs)
    }
    # policyMapperの定義(基本はデフォルト通り)

    def policyMapper(fullName):
        agentName, modelName, policyName = fullName.split(":")
        return policyName

    # 生成状況の確認
    observation_space = env.observation_space
    action_space = env.action_space
    print("=====Agent classes=====")
    print("Agent_"+blueUserModuleID, " = ", blueAgentClass)
    print("Agent_"+redUserModuleID, " = ", redAgentClass)
    print("=====Policies=====")
    for name, policy in policies.items():
        print(name, " = ", type(policy))
    print("=====Policy Map (at reset)=====")
    for fullName in action_space:
        print(fullName, " -> ", policyMapper(fullName))
    print("=====Agent to Asset map=====")
    for agent in [a() for a in env.manager.getAgents()]:
        print(agent.getFullName(), " -> ", "{")
        for port, parent in agent.parents.items():
            print("  ", port, " : ", parent.getFullName())
        print("}")

    # シミュレーションの実行
    print("=====running simulation(s)=====")
    numEpisodes = matchInfo["num_episodes"]
    for episodeCount in range(numEpisodes):
        obs = env.reset()
        rewards = {k: 0.0 for k in obs.keys()}
        dones = {k: False for k in obs.keys()}
        infos = {k: None for k in obs.keys()}
        for p in policies.values():
            p.reset()
        dones["__all__"] = False
        while not dones["__all__"]:
            observation_space = env.get_observation_space()
            action_space = env.get_action_space()
            actions = {k: policies[policyMapper(k)].step(
                o,
                rewards[k],
                dones[k],
                infos[k],
                k,
                observation_space[k],
                action_space[k]
            ) for k, o in obs.items() if policyMapper(k) in policies}
            obs, rewards, dones, infos = env.step(actions)
        print("episode(", episodeCount+1, "/", numEpisodes, "), winner=",
              env.manager.getRuler()().winner, ", scores=", {k: v for k, v in env.manager.scores.items()})


if __name__ == "__main__":
    candidates=json.load(open("candidates.json","r"))
    parser=argparse.ArgumentParser()
    parser.add_argument("blue",type=str,help="name of blue team")
    parser.add_argument("red",type=str,help="name of red team")
    parser.add_argument("-n","--num_episodes",type=int,default=1,help="number of evaluation episodes")
    parser.add_argument("-l","--log_dir",type=str,help="log directory")
    parser.add_argument("-v","--visualize",action="store_true",help="use when you want to visualize episodes")
    args=parser.parse_args()
    assert(args.blue in candidates and args.red in candidates)
    matchInfo={
        "blue":candidates[args.blue],
        "red":candidates[args.red],
        "num_episodes":args.num_episodes,
        "visualize":args.visualize,
    }
    if(args.log_dir is not None):
        matchInfo["log_dir"]=args.log_dir
    run(matchInfo)
