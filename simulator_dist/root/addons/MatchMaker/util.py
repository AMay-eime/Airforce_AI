# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import copy

def managerConfigReplacer(config,matchInfo):
    """対戦カードを表すmatchInfoに従い、SimulationManagerのコンフィグを置き換える関数。
    configはSimulationManager::getManagerConfig()で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Manager"キー以下の部分となる。
    """
    ret=copy.deepcopy(config)
    agentConfigDispatcher=ret["AgentConfigDispatcher"]
    numBlue=len(agentConfigDispatcher["BlueAgents"]["overrider"][0]["elements"])
    numRed=len(agentConfigDispatcher["RedAgents"]["overrider"][0]["elements"])
    agentConfigDispatcher["BlueAgents"]["alias"]=matchInfo["Blue"]["Policy"]
    if(matchInfo["Blue"]["MultiPort"]):
        #中央集権型のAgent
        agentConfigDispatcher["BlueAgents"]["overrider"][0]["elements"]=[
            {"type":"direct","value":{"name":"Blue","port":str(i),"policy":matchInfo["Blue"]["Policy"]+matchInfo["Blue"]["Suffix"]}}
            for i in range(numBlue)
        ]
    else:
        #SingleAssetAgent
        agentConfigDispatcher["BlueAgents"]["overrider"][0]["elements"]=[
            {"type":"direct","value":{"name":"Blue"+str(i+1),"policy":matchInfo["Blue"]["Policy"]+matchInfo["Blue"]["Suffix"]}}
            for i in range(numBlue)
        ]
    agentConfigDispatcher["RedAgents"]["alias"]=matchInfo["Red"]["Policy"]
    if(matchInfo["Red"]["MultiPort"]):
        #中央集権型のAgent
        agentConfigDispatcher["RedAgents"]["overrider"][0]["elements"]=[
            {"type":"direct","value":{"name":"Red","port":str(i),"policy":matchInfo["Red"]["Policy"]+matchInfo["Red"]["Suffix"]}}
            for i in range(numRed)
        ]
    else:
        #SingleAssetAgent
        agentConfigDispatcher["RedAgents"]["overrider"][0]["elements"]=[
            {"type":"direct","value":{"name":"Red"+str(i+1),"policy":matchInfo["Red"]["Policy"]+matchInfo["Red"]["Suffix"]}}
            for i in range(numRed)
        ]
    return ret
