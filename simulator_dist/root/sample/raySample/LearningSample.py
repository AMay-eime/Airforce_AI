# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""RayLeagueLearnerを用いて分散学習を行うためのサンプルスクリプト。
RayLeagueLearnerのコンフィグの大半をコマンドライン引数でjsonファイルを指定することで設定し、
一部のjson化しにくい部分を本スクリプトで設定している。
以下のようにコマンドライン引数としてjsonファイルを与えることで学習が行われる。
python LearningSample.py config.json
"""
import sys
import json
import ray
from ray.rllib.env.env_context import EnvContext
from ASRCAISim1.addons.rayUtility.RayManager import RayManager
from ASRCAISim1.addons.MatchMaker.RayLeagueLearner import RayLeagueLearner
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import BVRMatchMaker, BVRMatchMonitor, wrapEnvForBVRMatchMaking
import OriginalModelSample #Factoryへの登録が必要なものは直接使用せずともインポートしておく必要がある
from OriginalModelSample.R4TorchNNSampleForRay import R4TorchNNSampleForRay
from ASRCAISim1.addons.rayUtility.extension.policy import DummyInternalRayPolicy
from ASRCAISim1.addons.rayUtility.extension.agents.impala import MyImpalaTrainer
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceTFPolicy
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy
from ASRCAISim1.addons.rayUtility.extension.agents.ppo import MyAPPOTrainer
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
from ray.rllib.agents.dqn.r2d2 import R2D2Trainer
from ray.rllib.agents.dqn.r2d2 import R2D2TFPolicy
from ray.rllib.agents.dqn.r2d2 import R2D2TorchPolicy

class DummyInternalRayPolicyForAPPO(DummyInternalRayPolicy):
	#APPOの場合、self.update_targetとして引数なしメソッドを持っていなければならない。
	#
	def __init__(self,observation_space,action_space,config):
		super().__init__(observation_space,action_space,config)
		def do_update_dummy():
			return
		self.update_target=do_update_dummy


availableTrainers={
	"IMPALA":{
		"trainer":MyImpalaTrainer,
		"tf":VTraceTFPolicy,
		"torch":VTraceTorchPolicy
	},
	"APPO":{
		"trainer":MyAPPOTrainer,
		"tf":AsyncPPOTFPolicy,
		"torch":AsyncPPOTorchPolicy,
		"internal":DummyInternalRayPolicyForAPPO
	},
	"R2D2":{
		"trainer":R2D2Trainer,
		"tf":R2D2TFPolicy,
		"torch":R2D2TorchPolicy
	},
}

def envCreator(config: EnvContext):
	import OriginalModelSample #Factoryへの登録が必要なものは直接使用せずともインポートしておく必要がある
	return wrapEnvForBVRMatchMaking(RayManager)(config)

if __name__ == "__main__":
	if(len(sys.argv)>1):
		config=json.load(open(sys.argv[1],'r'))
		#既存のRay Clusterに接続する場合は以下を指定。(jsonで指定してもよい)
		if(not "head_ip_address" in config):
			config["head_ip_address"]="auto"
		#本スクリプトを実行するノードがClusterに接続する際のIPアドレスがデフォルト値("127.0.0.1")でない場合、以下で指定。(jsonで指定してもよい)
		if(not "entrypoint_ip_address" in config):
			config["entrypoint_ip_address"]="127.0.0.1"
		config["envCreator"]=envCreator
		learner=RayLeagueLearner(
			config,
			availableTrainers,
			BVRMatchMaker,
			BVRMatchMonitor
		)
		learner.run()

