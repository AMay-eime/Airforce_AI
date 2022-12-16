# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from typing import Callable, Dict
from ray.rllib.env import MultiAgentEnv,EnvContext
from ASRCAISim1.libCore import SimulationManager
from ASRCAISim1.policy.StandalonePolicy import StandalonePolicy
from ASRCAISim1.GymManager import GymManager,SinglizedEnv,SimpleEvaluator

class RayManager(GymManager,MultiAgentEnv):
	"""
	Ray.RLLibのMultiAgentEnvをMix-in継承した、SimulationManagerのラッパークラス
	Ray.RLLibのインターフェースに準じて、コンストラクタに与える引数はdict型のconfigのみ。
		Args:
			context (EnvContext): context["config"]にSimulationManagerに渡すjsonを持たせる。また、必要に応じて、context["overrider"]にworker_index及びvector_indexに応じたconfig置換関数(std::function<nl::json(const nl::json&,int,int)>相当)を与える。
	"""
	def __init__(self,context: EnvContext):
		asdict=dict(context)
		asdict["worker_index"]=context.worker_index
		asdict["vector_index"]=context.vector_index
		super().__init__(asdict)

class RaySinglizedEnv(SinglizedEnv):
	"""SinglizedEnvのコンストラクタ引数をrayのEnvContextに置き換えたもの。
	"""
	def __init__(self,context: EnvContext):
		asdict=dict(context)
		asdict["worker_index"]=context.worker_index
		asdict["vector_index"]=context.vector_index
		super().__init__(asdict)

def getDefaultRayPolicyMapper():
	"""ray準拠のエージェントとポリシーのマッピング関数を生成する。
		基本的には環境側のconfigでポリシー名を指定しておくものとする。
		環境の出力となる辞書のキーとなるagentIdは、agentId=agentName:modelName:policyNameの形式としている。
	"""
	def ret(agentId,episode=None,**kwargs):
		agentName,modelName,policyName=agentId.split(":")
		return policyName
	return ret

class RaySimpleEvaluator(SimpleEvaluator):
	"""SimpleEvaluatorのコンストラクタ引数をrayのEnvContextに置き換えたもの。
	"""
	def __init__(self,context: EnvContext, policies: Dict[str,StandalonePolicy], policyMapper: Callable[[str],str]=getDefaultRayPolicyMapper()):
		asdict=dict(context)
		asdict["worker_index"]=context.worker_index
		asdict["vector_index"]=context.vector_index
		super().__init__(asdict,policies,policyMapper)