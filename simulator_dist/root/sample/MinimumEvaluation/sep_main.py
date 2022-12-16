# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os
import time
import importlib
import ray
from ASRCAISim1.libCore import SimulationManager
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager,SimpleEvaluator
from ASRCAISim1.addons.AgentIsolation import PolicyDelegator

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

def postGlobalCommand(command,data,server,port):
    #終了処理(kill)や次エピソードへの準備(clear)のため
    import socket
    import pickle
    bufferSize=4096
    conn=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    conn.connect((server,port))
    msg=pickle.dumps(["",command,data])
    header="HEADER:{:16d}".format(len(msg)).encode("utf-8")
    conn.send(header)
    ack=conn.recv(bufferSize).decode("utf-8")
    assert(ack[4:]=="OK")
    conn.send(msg)
    header=conn.recv(bufferSize).decode("utf-8")
    msgLen=int(header[7:])
    conn.send("ACK:OK".encode("utf-8"))
    received=0
    ret=b""
    while received<msgLen:
        part=conn.recv(bufferSize)
        received+=len(part)
        if(len(part)>0):
            ret+=part
    assert(received==msgLen)
    conn.close()
    return pickle.loads(ret)

def run(sep_config,matchInfo):
    blueServer=sep_config["blue"]["server"]
    blueAgentPort=sep_config["blue"]["agentPort"]
    bluePolicyPort=sep_config["blue"]["policyPort"]
    redServer=sep_config["red"]["server"]
    redAgentPort=sep_config["red"]["agentPort"]
    redPolicyPort=sep_config["red"]["policyPort"]

    blueUserModuleID=matchInfo["blue"]["userModuleID"]
    redUserModuleID=matchInfo["red"]["userModuleID"]
    blueArgs=matchInfo["blue"].get("args",None)
    redArgs=matchInfo["red"].get("args",None)
    seed=matchInfo.get("seed",None)
    if(seed is None):
        import numpy as np
        seed = np.random.randint(2**31)
    logName=matchInfo["blue"].get("logName",blueUserModuleID)+"_vs_"+matchInfo["red"].get("logName",redUserModuleID)
    
    #edge側にuserModuleIDを伝達
    postGlobalCommand("userModuleID",(blueUserModuleID,blueArgs),blueServer,blueAgentPort)
    postGlobalCommand("userModuleID",(blueUserModuleID,blueArgs),blueServer,bluePolicyPort)
    postGlobalCommand("userModuleID",(redUserModuleID,redArgs),redServer,redAgentPort)
    postGlobalCommand("userModuleID",(redUserModuleID,redArgs),redServer,redPolicyPort)
    #ユーザーモジュールの読み込み
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
    blueModule = importedUsers[blueUserModuleID]
    redModule = importedUsers[redUserModuleID]
    
    #コンフィグの生成
    agentConfig={
        "Factory":{
            "Agent":{
                "Agent_"+blueUserModuleID:{
                    "class":"AgentDelegator",
                    "config":{
                        "socketServer":blueServer,
                        "socketPort":blueAgentPort
                    }
                },
                "Agent_"+redUserModuleID:{
                    "class":"AgentDelegator",
                    "config":{
                        "socketServer":redServer,
                        "socketPort":redAgentPort
                    }
                }
            }
        },
        "Manager":{
            "AgentConfigDispatcher":{
                "BlueAgents": agentConfigMaker("Blue",blueUserModuleID, blueModule.isUserAgentSingleAsset(blueArgs),4),
                "RedAgents": agentConfigMaker("Red",redUserModuleID, redModule.isUserAgentSingleAsset(redArgs),4)
            }
        }
    }
    configs=[
        os.path.join(os.path.dirname(__file__), "common/R4_contest_mission_config.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_asset_models.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_agent_ruler_reward_models.json"),
        agentConfig,
        {
            "Manager":{
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
                    }
                } if "log_dir" in matchInfo else {}
            }
        }
    ]
    context={
        "config":configs,
        "worker_index":0,
        "vector_index":0
    }
    #edge側にFactoryのconfigを伝達
    fullConfig=SimulationManager.parseConfig(configs)() #末尾に()をつけてnl::json→dictに変換
    postGlobalCommand("factoryConfig",fullConfig.get("Factory",{}),blueServer,blueAgentPort)
    postGlobalCommand("factoryConfig",fullConfig.get("Factory",{}),redServer,redAgentPort)
    #edge側の事前準備を完了(AgentDelegatee,PolicyDelegateeの通信待受に移行)
    postGlobalCommand("ready",None,blueServer,blueAgentPort)
    postGlobalCommand("ready",None,blueServer,bluePolicyPort)
    postGlobalCommand("ready",None,redServer,redAgentPort)
    postGlobalCommand("ready",None,redServer,redPolicyPort)

    #環境の生成
    env=GymManager(context)
    

    #StandalonePolicyの生成
    policies={
        "Policy_"+blueUserModuleID:PolicyDelegator("Policy_"+blueUserModuleID,blueServer,bluePolicyPort),
        "Policy_"+redUserModuleID:PolicyDelegator("Policy_"+redUserModuleID,redServer,redPolicyPort)
    }
    #policyMapperの定義(基本はデフォルト通り)
    def policyMapper(fullName):
        agentName,modelName,policyName=fullName.split(":")
        return policyName

    #生成状況の確認
    observation_space=env.observation_space
    action_space=env.action_space
    print("=====Policy Map (at reset)=====")
    for fullName in action_space:
        print(fullName," -> ",policyMapper(fullName))
    print("=====Agent to Asset map=====")
    for agent in [a() for a in env.manager.getAgents()]:
        print(agent.getFullName(), " -> ","{")
        for port,parent in agent.parents.items():
            print("  ",port," : ",parent.getFullName())
        print("}")
    
    #シミュレーションの実行
    print("=====running simulation(s)=====")
    numEpisodes = matchInfo["num_episodes"]
    for episodeCount in range(numEpisodes):
        obs=env.reset()
        rewards={k:0.0 for k in obs.keys()}
        dones={k:False for k in obs.keys()}
        infos={k:None for k in obs.keys()}
        for p in policies.values():
            p.reset()
        dones["__all__"]=False
        while not dones["__all__"]:
            observation_space=env.get_observation_space()
            action_space=env.get_action_space()
            actions={k:policies[policyMapper(k)].step(
                o,
                rewards[k],
                dones[k],
                infos[k],
                k,
                observation_space[k],
                action_space[k]
            ) for k,o in obs.items() if policyMapper(k) in policies}
            obs, rewards, dones, infos = env.step(actions)
        postGlobalCommand("clear",None,blueServer,blueAgentPort)
        postGlobalCommand("clear",None,redServer,redAgentPort)
        print("episode(",episodeCount+1,"/",numEpisodes,"), winner=",env.manager.getRuler()().winner,", scores=",{k:v for k,v in env.manager.scores.items()})
    #終了処理
    postGlobalCommand("kill",None,blueServer,blueAgentPort)
    postGlobalCommand("kill",None,blueServer,bluePolicyPort)
    postGlobalCommand("kill",None,redServer,redAgentPort)
    postGlobalCommand("kill",None,redServer,redPolicyPort)


if __name__ == "__main__":
    import json
    import argparse
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
    sep_config=json.load(open("sep_config.json","r"))
    run(sep_config,matchInfo)
