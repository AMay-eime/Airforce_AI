# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""RayLeagueLearnerを用いて摸倣学習を行うためのサンプルスクリプト。
RayLeagueLearnerのコンフィグの大半をコマンドライン引数でjsonファイルを指定することで設定し、
一部のjson化しにくい部分を本スクリプトで設定している。
以下のようにコマンドライン引数としてjsonファイルを与えることで学習が行われる。
python ImitationSample.py config.json

また、摸倣学習を行う際には教師データの指定を行う必要があり、
Trainerのconfigにおいて、
	"input":"/your/expert/data/path/Traj*/*.json",
のようにデータのパスを指定する必要がある。
本サンプルでは、ExpertTrajectortGather.pyで生成された軌跡データ(jsonファイル)を教師データとして用いることを前提としている。
"""
import sys
import json
import ray
from ray.rllib.env.env_context import EnvContext
from ASRCAISim1.addons.rayUtility.RayManager import RaySinglizedEnv
from ASRCAISim1.addons.MatchMaker.RayLeagueLearner import RayLeagueLearner
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import BVRMatchMaker, BVRMatchMonitor, wrapEnvForBVRMatchMaking
import OriginalModelSample #Factoryへの登録が必要なものは直接使用せずともインポートしておく必要がある
from OriginalModelSample.R4TorchNNSampleForRay import R4TorchNNSampleForRay
from ASRCAISim1.addons.rayUtility.extension.policy import DummyInternalRayPolicy
from ASRCAISim1.addons.rayUtility.extension.agents.marwil import RNNMARWILTrainer
from ASRCAISim1.addons.rayUtility.extension.agents.marwil import RNNMARWILTFPolicy
from ASRCAISim1.addons.rayUtility.extension.agents.marwil import RNNMARWILTorchPolicy

availableTrainers={
	"RNNMARWIL":{
		"trainer":RNNMARWILTrainer,
		"tf":RNNMARWILTFPolicy,
		"torch":RNNMARWILTorchPolicy
	},
}

def envCreator(config: EnvContext):
	import OriginalModelSample #Factoryへの登録が必要なものは直接使用せずともインポートしておく必要がある
	return wrapEnvForBVRMatchMaking(RaySinglizedEnv)(config)

if __name__ == "__main__":
	if(len(sys.argv)>1):
		config=json.load(open(sys.argv[1],'r'))
		#既存のRay Clusterに接続する場合は以下を指定。(jsonで指定してもよい)
		if(not "head_ip_address" in config):
			config["head_ip_address"]="auto"
		#本スクリプトを実行するノードがClusterに接続する際のIPアドレスがデフォルト値("127.0.0.1")でない場合、以下を指定。(jsonで指定してもよい)
		if(not "entrypoint_ip_address" in config):
			config["entrypoint_ip_address"]="192.168.1.101"
		#摸倣学習はシングルエージェント化した環境を与える
		config["envCreator"]=envCreator
		config["register_env_as"]="ASRCAISim1Singlized"
		config["as_singleagent"]=True
		learner=RayLeagueLearner(
			config,
			availableTrainers,
			BVRMatchMaker,
			BVRMatchMonitor
		)
		learner.run()

