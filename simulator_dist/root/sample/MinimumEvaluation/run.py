import os
import sys
import json
import time
import importlib
import tracemalloc
from timeout_decorator import timeout, TimeoutError
from argparse import ArgumentParser

from ASRCAISim1.libCore import Factory, Fighter
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager


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


def policyMapper(fullName):
    agentName, modelName, policyName = fullName.split(":")
    return policyName


def wrap_get_action_from_obs(args, time_out):
    @timeout(time_out)
    def get_action_from_obs(policies, rewards, dones, infos, observation_space, action_space, policyMapper, obs, team):
        actions = {}
        for k, o in obs.items():
            suff_k = k.split('_')[0]
            if policyMapper(k) in policies:
                if team in suff_k:
                    actions[k] = policies[policyMapper(k)].step(o, rewards[k], dones[k], infos[k], k, observation_space[k], action_space[k])
        return actions
    
    return get_action_from_obs(*args)


def step_forward(func, args, time_out):
    """
    func: decorated with timeout
    """
    try:
        r = func(args, time_out)
        return r
    except TimeoutError:
        print('TimeOut')
        return args[5].sample()


def fight(matchInfo):
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
        try:
            addPythonClass("Agent", "Agent_"+blueUserModuleID, blueAgentClass)
            Factory.addDefaultModel(
                "Agent",
                "Agent_"+blueUserModuleID,
                {
                    "class": "Agent_"+blueUserModuleID,
                    "config": blueModule.getUserAgentModelConfig(blueArgs)
                }
            )
        except Exception as e:
            print(e)
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
        try:
            addPythonClass("Agent", "Agent_"+redUserModuleID, redAgentClass)
            Factory.addDefaultModel(
                "Agent",
                "Agent_"+redUserModuleID,
                {
                    "class": "Agent_"+redUserModuleID,
                    "config": redModule.getUserAgentModelConfig(redArgs)
                }
            )
        except Exception as e:
            print(e)
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
    loggers = {
        "MultiEpisodeLogger":{
            "class":"MultiEpisodeLogger",
            "config":{
                "prefix":os.path.join(matchInfo["log_dir"],logName),
                "episodeInterval":1,
                "ratingDenominator":100
            }
        }
    }
    if matchInfo["replay"]:
        loggers["GodViewStateLogger"]={
            "class":"GodViewStateLogger",
            "config":{
                "prefix":os.path.join(matchInfo["log_dir"],logName),
                "episodeInterval":1,
                "innerInterval":15
            }
        }

    envConfig = {
                    "Manager": {
                        "Rewards": [],
                        "seed":seed,
                        "ViewerType":"God" if matchInfo["visualize"] else "None",
                        "Loggers": loggers
                    }
                }
    configs = [
        os.path.join(os.path.dirname(__file__), "common/R4_contest_mission_config.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_asset_models.json"),
        os.path.join(os.path.dirname(__file__), "common/R4_contest_agent_ruler_reward_models.json"),
        agentConfig,
        envConfig
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
    time_out = matchInfo['time_out']
    start_time = time.time()
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

        # get Red action
        red_actions = step_forward(func = wrap_get_action_from_obs, args=(policies, rewards, dones, infos, observation_space, action_space, policyMapper, obs, 'Red'), time_out=time_out)

        # get Blue action
        blue_actions = step_forward(func = wrap_get_action_from_obs, args=(policies, rewards, dones, infos, observation_space, action_space, policyMapper, obs, 'Blue'), time_out=time_out)

        actions = dict(red_actions, **blue_actions)

        obs, rewards, dones, infos = env.step(actions)
        

    time_elapsed = time.time() - start_time
    winner = env.manager.getRuler()().winner
    detail = {
        "time_elapsed":time_elapsed,
        "finishedTime":float(env.manager.getTime()),
        "numSteps":float(env.manager.getTickCount()/env.manager.getAgentInterval()),
        "totalRewards":{agent.getName():float(env.manager.totalRewards[agent.getFullName()]) for agent in [a() for a in env.manager.getAgents()]},
        "numAlives":{team:float(np.sum([f.isAlive() for f in [f() for f in env.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team)]])) for team in env.manager.getRuler()().score},
        "endReason":env.manager.getRuler()().endReason.name
        }
    scores = {k: v for k, v in env.manager.scores.items()}
    detail['scores'] = scores

    return winner, detail


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exec-path")
    parser.add_argument("--log-dir", default = './log')
    parser.add_argument("--replay", default = 1, type=int)
    parser.add_argument("--time-out", default = 0.5, type=float)
    parser.add_argument("--memory-limit", default = 7, type=float)

    return parser.parse_args()


def main():
    # parse the arguments and set the path
    args = parse_args()
    abs_exec_path = os.path.abspath(args.exec_path)
    with open(os.path.join(abs_exec_path, "params.json")) as f:
        params = json.load(f)
    model_args = params.get("args", None)
    sys.path.append(abs_exec_path)

    # set match info("blue"=you)
    matchInfo={
        "blue":{
            "userModuleID": "Agent",
            "args": model_args,
            "logName": "MyAgent"
            },
        "red":{
            "userModuleID": "RuleBased",
            "args": {
                "type": "Fixed"
                },
            "logName": "Rule-Fixed"
            },
        "visualize":False,
        "log_dir":args.log_dir,
        "replay":args.replay,
        "time_out":args.time_out
    }

    # execute the fight against "RuleBased" algorithm
    memory_limit = args.memory_limit * (1024**3)
    tracemalloc.start()
    winner, detail = fight(matchInfo)
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")
    total_memory = 0
    for stat in stats:
        total_memory += stat.size
    print("\nUsed memory: {} [Bytes]".format(total_memory))
    if total_memory >= memory_limit:
        print("Total memory: {} [Bytes](>={})".format(total_memory, memory_limit))
        return None
    tracemalloc.clear_traces()
    print("\nResult:")
    print("  You(Blue) vs Rule-Based(Red), winner: {}".format(winner))
    print("\nDetails:")
    for k, v in detail.items():
        print("  ", k, v)

if __name__ == "__main__":
    main()
