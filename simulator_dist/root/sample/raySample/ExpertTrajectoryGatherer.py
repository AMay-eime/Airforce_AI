# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import sys
import time
import json
import ray
from ray.rllib.env.env_context import EnvContext
import ray.ray_constants as ray_constants
import OriginalModelSample #Factoryへの登録が必要なものは直接使用せずともインポートしておく必要がある
from ASRCAISim1.addons.rayUtility.RayManager import RayManager

@ray.remote
def worker(config,worker_index):
	from ASRCAISim1.addons.rayUtility.RayManager import RayManager
	import OriginalModelSample #Factoryへの登録のためにこのファイルで直接使用せずとも必須
	env=RayManager(EnvContext(
		config["env_config"],
		worker_index,
		0
	))
	env.seed(config["seed"]+worker_index)
	for i in range(config["num_episodes_per_worker"]):
		startT=time.time()
		ob=env.reset()
		action_space=env.get_action_space()
		action=action_space.sample()
		dones={"__all__":False}
		while not dones["__all__"]:
			ob, reward, dones, info = env.step(action)
		endT=time.time()
		print("Episode(",worker_index,",",i+1,") ended in ",endT-startT," seconds.")
	return True
class ExpertTrajectoryGatherer:
	def __init__(self,config):
		self.config=json.load(open(config,'r'))
		#ダミー環境を一度生成し、完全な形のenv_configを取得する。
		dummyEnv=RayManager(EnvContext(
			self.config["env_config"],
			-1,
			-1
		))
		self.completeEnvConfig={
			key:value for key,value in self.config["env_config"].items() if key!="config"
		}
		self.completeEnvConfig["config"]={
				"Manager":dummyEnv.getManagerConfig()(),
				"Factory":dummyEnv.getFactoryModelConfig()()
			}
		self.config["env_config"]=self.completeEnvConfig
		writerConfig={
			"class":"ExpertTrajectoryWriter",
			"config":{
				"prefix":self.config["save_dir"]+("" if self.config["save_dir"][-1]=="/" else "/")
			}
		}
		if("Loggers" in self.config["env_config"]["config"]["Manager"]):
			self.config["env_config"]["config"]["Manager"]["Loggers"]["ExpertTrajectoryWriter"]=writerConfig
		else:
			self.config["env_config"]["config"]["Manager"]["Loggers"]={"ExpertTrajectoryWriter":writerConfig}
	def run(self):
		if(not ray.is_initialized()):
			#rayの初期化がされていない場合、ここで初期化する。
			#既存のRay Clusterがあればconfigに基づき接続し、既存のClusterに接続できなければ、新たにClusterを立ち上げる。
			#強制的に新しいClusterを立ち上げる場合は、"head_ip_address"にnull(None)を指定する。
			#既存のClusterに接続する場合は、"head_ip_address"にHead nodeのIPアドレスとポートを指定する。rayの機能で自動的に接続先を探す場合は"auto"を指定する。
			head_ip_address=self.config.get("head_ip_address","auto")
			entrypoint_ip_address=self.config.get("entrypoint_ip_address",ray_constants.NODE_DEFAULT_IP)
			try:
				ray.init(address=head_ip_address,_node_ip_address=entrypoint_ip_address)
			except:
				print("Warning: Failed to init with the given head_ip_address. A new cluster will be launched instead.")
				ray.init(_node_ip_address=entrypoint_ip_address)
		import signal
		original=signal.getsignal(signal.SIGINT)
		res=[worker.remote(self.config,i) for i in range(self.config["num_workers"])]
		def sig_handler(sig,frame):
			for r in res:
				ray.kill(r)
			signal.signal(signal.SIGINT,original)
		signal.signal(signal.SIGINT,sig_handler)
		ray.get(res)
		return res
		

if __name__ == "__main__":
	if(len(sys.argv)>1):
		gatherer=ExpertTrajectoryGatherer(sys.argv[1])
		gatherer.run()
