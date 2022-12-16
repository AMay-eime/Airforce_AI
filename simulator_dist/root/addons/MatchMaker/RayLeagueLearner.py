# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import gym
import os
import sys
import json
import glob
import shutil
import copy
import datetime
import numpy as np
import ray
import ray.ray_constants as ray_constants
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
from ray.tune.utils.trainable import TrainableUtil
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ASRCAISim1.addons.rayUtility.RayManager import RayManager
from ASRCAISim1.addons.rayUtility.RayManager import RaySinglizedEnv
from ASRCAISim1.addons.rayUtility.utility.common import loadWeights,saveWeights
from ASRCAISim1.addons.rayUtility.extension.policy import DummyInternalRayPolicy
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import BVRMatchMaker, BVRMatchMonitor, wrapEnvForBVRMatchMaking, managerConfigReplacer
from ASRCAISim1.addons.MatchMaker.MatchMakerForRay import loadInitialWeights, populate, getMatchMakerOnRayClass

def defaultEnvCreator(config: EnvContext):
    return wrapEnvForBVRMatchMaking(RayManager)(config)

def defaultSinglizedEnvCreator(config: EnvContext):
    return wrapEnvForBVRMatchMaking(RaySinglizedEnv)(config)

class RayLeagueLearner:
    """複数のTrainerを並列に動かし、単一のMatchMakerにより全エピソードの対戦を管理する一連の学習を行うためのクラス。
    __init__に与えるconfig及びavailableTrainersの記述方法は以下の通り。
    config={
        "head_ip_address":Optional[str], #ray.initに渡す、rayのHeadノードのIPアドレス(ポート番号を含む)。"auto"とすることで自動検索、Noneとすることで新規クラスタとなる。
        "entrypoint_ip_address":Optional[str], #ray.initに渡す、このインスタンスが走るノードのIPアドレス(ポート番号を除く)。省略時は"127.0.0.1"となる。
        "envCreator": Optional[Callable[[EnvContext],gym.Env]], #環境インスタンスを生成する関数。省略時はRayManagerを生成する関数となる。
        "register_env_as": Optional[str], #登録する環境の名称。省略時は"ASRCAISim1"
        "as_singleagent": Optional[bool], #シングルエージェントとして環境及びTrainerを扱うかどうか。デフォルトはFalseだが、MARWILによる摸倣学習を使用する場合にはTrueとする必要がある。
        "seed":Optional[int], #乱数のシード値。
        "train_steps":int, #学習のステップ数。rayのTrainer.trainを呼ぶ回数を指す。
        "refresh_interval":int, #学習中にTrainerの再読込を行う周期。workerを複数使用する際に発生すると報告されているメモリリーク対策として、Trainerインスタンスの再生成による緩和を試みるもの。
        "checkpoint_interval":int, #チェックポイントの生成周期。
        "save_dir":str, #ログの保存場所。rayの全ノードから共通してアクセスできるパスでなければならない。(以下全てのパスについて同様)
        "experiment_name":str, #試行に付与する名称。save_dir以下にこの名称のディレクトリが生成され、各試行のログはその下にrun_YYYY-mm-dd_HH-MM-SSというディレクトリとして保存される。
        "restore":Optional[str], #既存のチェックポイントから読み込む場合にチェックポイントを含むログのパスを指定する。上記のrun_YYYY-mm-dd_HH-MM-SSの階層まで指定する。
        "restore_checkpoint_number":Union[int,"latest",None], #既存のチェックポイントから読み込む場合、チェックポイントの番号を指定する。"latest"と指定することで当該ログ内の最新のチェックポイントを自動で検索する。
        "trainer_common_config":{ #各Trainerに与えるコンフィグの共通部分を記述する。デフォルト値はray.rllib.trainer.pyを始めとする各Trainerクラスの定義とともに示されている。以下は主要な項目を例示する。
            "env_config":{ #環境に渡されるコンフィグを記述する。詳細は各Envクラスの定義を参照のこと。
                "config":Union[dict,list[Union[dict,str]], #GymManagerクラスで要求
                "overrider": Optional[Callable[[dict,int,int],dict]], #GymManagerクラスで要求
                "target": Union[Callable[[str],bool],str], #SinglizedEnvクラスで要求
                "policies": Dict[str,StandalonePolicy], #SinglizedEnvクラスで要求
                "policyMapper": Optional[Callable[[str],str]], #SinglizedEnvクラスで要求
                "exposeImitator": bool (False if omitted), #SinglizedEnvクラスで要求
                "runUntilAllDone": bool (True if omitted), #SinglizedEnvクラスで要求
            },
            "model":dict,#NNモデルの構造を定義する。ray.rllib.models.catalog.pyに設定項目の一覧とデフォルト値が示されているものを使用してもよいし、
                         #独自のカスタムモデルを定義して仕様してもよい。
            "manual_update_for_untrainable_policies":true #MatchMakerによる重み読み書きに対応するために、ray.rllibのIMPALAとAPPOクラスを拡張して非学習対象Policyの自動重み配信を無効化するためのフラグ
        },
        "trainers":{ #Trainer名をキーとしたdictにより、生成するTrainerインスタンスを指定する。
            <Trainer's name>:{
                "trainer_type": str, #Trainerの種類。availableTrainersのキーから選択する。
                "config_overrider":Optional[dict[str,Any]], #trainer_common_configを上書きするためのdict。主に"num_workers"を上書きすることになる。
                "policies_to_train":list[str], #このTrainerにより学習を行うPolicy名のリスト。
                "node_designation":Optional[str], #このTrainerインスタンスを生成するrayノードのIPアドレス。省略時はrayにより自動で選択される。
            },
            ...
        },
        "policies": { #Policyに関する設定。Policy名をキーとしたdictで与える。MatchMakerのpolicy_configとしても使用される。
            <Policy's name>: {
                "multi_port": bool, #1体で1陣営分を動かすか否か。省略時はFalse。
                "active_limit": int, #保存された過去の重みを使用する数の上限を指定する。
                "is_internal": bool, #SimulationManagerクラスにおけるInternalなPolicyかどうか。
                "populate": None or { #重み保存条件の指定
                    "firstPopulation": int, # 初回の保存を行うエピソード数。0以下の値を指定すると一切保存しない。
                    "interval": int, # 保存間隔。0以下の値を指定すると一切保存しない。
                    "on_start": bool, # 開始時の初期重みで保存するかどうか。省略時はFalseとなる。
                    "reset"; float, #重み保存時の重みリセット確率(0〜1)
                },
                "rating_initial": float, #初期レーティング
                "rating_fixed": bool, #レーティングを固定するかどうか。
                "initial_weight": None or str, #初期重み(リセット時も含む)のパス。
                "save_path_at_the_end": None or str, #学習終了時の重みを別途保存したい場合、そのパスを記述する。
            },
            ...
        },
        "match_maker":{ #MatchMakerに関する設定。外から与えるべきものは上記"policies"以外は以下の二つのみ。
            "seed":12345, #MatchMakerとしての乱数シード。
            "match_config": Any #対戦カードの生成に関する指定。サンプルのBVRMatchMakerクラスを使用する場合には不要。
        }
    }
    availableTrainers={
        <trainer_type>:{
            "trainer": Trainer, #使用するTrainerクラス。
            "tf": TFPolicy, #Tensorflowを使用する場合のPolicyクラス。
            "torch": TorchPolicy, #PyTorchを使用する場合のPolicyクラス。
            "internal": Optional[Policy], #actionをInternalに計算するAgentに対応する、ダミーのPolicyクラス。省略時はDummyInternalRayPolicyとなる。
        }
    }

    また、保存されるログは以下のような構成となる。なお、拡張子datで保存される学習モデル(重み)は、
    Policy.get_weights()で得られるnp.arrayのlist又はdictをそのままpickleでdumpしたものである。
    <save_dir> #jsonで指定したsave_dir
        <experiment_name> #jsonで指定したexperiment_name
            policies #Policyの重みに関するログ
                checkpoints #チェックポイント
                    checkpoint_<step>
                        <policy>.dat
            matches #MatchMakerに関するログ
                checkpoint_<step> #チェックポイント
                    MatchMaker-<step>.dat
                matches_YYYYmmddHHMMSS.csv #全対戦結果の概要
            trainers #Trainerに関するログ
                <trainer> #各Trainer名のディレクトリをrayのチェックポイント保存先とする
                    checkpoint_<step>
                        checkpoint-<step>
                    <other files created by ray>
    """
    def __init__(self,
                config,
                availableTrainers,
                matchMakerClass=BVRMatchMaker,
                matchMonitorClass=BVRMatchMonitor):
        """初期化処理
        
        Args:
            config (dict): 学習の各種設定値を記述したdict。
            aveilableTrainers (dict): 指定可能なTrainerの一覧を示すdict。
        """
        self.config=config
        self.availableTrainers=availableTrainers
        self.matchMakerClass=matchMakerClass
        self.matchMonitorClass=matchMonitorClass
    def _policySetup(self,trainer_type,trainer_config,policies_to_train):
        """対象のTrainerインスタンスにとって必要となるPolicyインスタンスを生成する。
        本サンプルでは、SimulationManagerのコンフィグが以下のように記述されていることを前提とする。
        (1)BlueとRedのAgentがAgentConfigDispatcherの"BlueAgents"と"RedAgents"をaliasとして参照することとされている。
        (2)AgentConfigDispatcherにおいて、各policy名に対応するAgentモデルを表すコンフィグが同名のaliasとして記述されている。
        (3)AgentConfigDispatcherにおいて、"BlueAgents"と"RedAgents"が"type"=="alias"であり、これらに(2)に該当するpolicy名のaliasを指定することで適切なコンフィグが得られるものとなっている。

        Args:
            trainer_type (str): Trainerの種類を表す文字列。self.availableTrainersのキーから選択する。
            trainer_config (dict): 対象とするTrainerインスタンスを生成するためのコンフィグ。
            policies_to_train (list): 対象とするTrainerインスタンスが学習対象とするPolicyの名前のリスト。
        """
        policies={}
        #MatchMaker.pyのmanagerConfigReplacerを用いてMatchMakerと同じルールでconfigを書き換え、全種類のPolicyを一度ずつ登場させる。
        dummyConfig=copy.deepcopy(trainer_config["env_config"])
        dummyConfig["config"]["Manager"]["Viewer"]="None"
        dummyConfig["config"]["Manager"]["Loggers"]={}
        for policyName in self.config["policies"]:
            isMultiPort=self.config["policies"][policyName].get("multi_port",False) is True
            matchInfo={
                "Blue":{"Policy":policyName,"Suffix":"","Weight":-1,"MultiPort":isMultiPort},
                "Red":{"Policy":policyName,"Suffix":"","Weight":-1,"MultiPort":isMultiPort}
            }
            print(matchInfo)
            dummyConfig["config"]["Manager"]=managerConfigReplacer(dummyConfig["config"]["Manager"],matchInfo)
            dummyEnv=RayManager(EnvContext(
                dummyConfig,
                -1,
                -1
            ))
            dummyEnv.reset()
            ob=dummyEnv.get_observation_space()
            ac=dummyEnv.get_action_space()
            for key in ac:
                _,m_name,p_name=key.split(":")
                if(p_name==policyName):
                    policyClass=self.availableTrainers[trainer_type]["torch"] if trainer_config.get("framework")=="torch" else self.availableTrainers[trainer_type]["tf"]
                    policy=[policyClass,ob[key],ac[key],{}]
                    if(policyName in policies_to_train):
                        policies[policyName]=policy
                    if(self.matchMaker is not None):
                        #trainingの対象でないPolicyは、MatchMakerを使用しない場合のみ有効とする(InternalなPolicyは除く)
                        for v in range(trainer_config["num_envs_per_worker"]):
                            if(v==0):
                                policies[policyName+"_Past_Blue"]=policy
                                policies[policyName+"_Past_Red"]=policy
                            else:
                                policies[policyName+"_Past_Blue_{}".format(v)]=policy
                                policies[policyName+"_Past_Red_{}".format(v)]=policy
                elif(p_name=="Internal"):
                    policies[m_name]=[self.availableTrainers[trainer_type].get("internal",DummyInternalRayPolicy),ob[key],ac[key],{}]
                else:
                    raise ValueError("Invalid policy config.")
        return policies
    def run(self):
        """
        configに従い学習を実行する。
        """
        if(self.config["trainer_common_config"]["framework"]=="tf"):
            import tensorflow as tf
            tf.compat.v1.disable_eager_execution()
        namespace=self.config.get("namespace","ASRCAISim1")
        entrypoint_ip_address=self.config.get("entrypoint_ip_address",ray_constants.NODE_DEFAULT_IP)
        if(not ray.is_initialized()):
            #rayの初期化がされていない場合、ここで初期化する。
            #既存のRay Clusterがあればconfigに基づき接続し、既存のClusterに接続できなければ、新たにClusterを立ち上げる。
            #強制的に新しいClusterを立ち上げる場合は、"head_ip_address"にnull(None)を指定する。
            #既存のClusterに接続する場合は、"head_ip_address"にHead nodeのIPアドレスとポートを指定する。rayの機能で自動的に接続先を探す場合は"auto"を指定する。
            head_ip_address=self.config.get("head_ip_address","auto")
            try:
                ray.init(address=head_ip_address,namespace=namespace,_node_ip_address=entrypoint_ip_address)
            except:
                print("Warning: Failed to init with the given head_ip_address. A new cluster will be launched instead.")
                ray.init(namespace=namespace,_node_ip_address=entrypoint_ip_address)
        node_ip_address=ray._private.services.get_node_ip_address()
        #Environmentの登録(envCreatorに別のCallableを与えることで、環境の生成をカスタマイズすることも可能)
        envCreator=self.config.get("envCreator",defaultEnvCreator)
        envName=self.config.get("register_env_as","ASRCAISim1")
        self.isSingleAgent=self.config.get("as_singleagent",False) #Env,Trainerともにシングルエージェントとして動かす場合にTrueとする。
        register_env(envName,envCreator)
        #チェックポイントの読み込み設定
        # restore (str or None): 読み込むログのパス。後述のexperiment_log_pathに相当。
        # restore_checkpoint_number (int or "latest"): チェックポイントの番号。"latest"を指定すると最新のものが自動的に検索される。
        restore_path=self.config.get("restore",None)
        if(restore_path is not None):
            restore_checkpoint_number=self.config.get("restore_checkpoint_number","latest")
            if(restore_checkpoint_number=="latest"):
                restore_checkpoint_number=max([int(os.path.basename(c)[11:])
                    for c in glob.glob(os.path.join(restore_path,"**","checkpoint_*"),recursive=True)])
            assert(isinstance(restore_checkpoint_number,int))
        # ログ保存ディレクトリの設定を行う。
        # save_dir (str): 保存先のディレクトリ。省略時はrayのデフォルト(~/ray_results)となる。
        # experiment_name (str): 試行の名称。省略時は"RayLeagueLearner"となる。
        # save_dir/experiment_name/run_YYYY-mm-dd-HH-MM-SS/というディレクトリ(=experiment_log_path)以下に各ログファイルが生成される。
        save_dir=self.config.get("save_dir",DEFAULT_RESULTS_DIR)
        experiment_name=self.config.get("experiment_name","RayLeagueLearner")
        suffix=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_log_path=os.path.join(save_dir,experiment_name,"run_"+suffix)
        #初期重みとして指定した重みファイルのコピー
        if(restore_path is None):
            #チェックポイントからの再開でないとき、configからパスを取得
            for key,value in self.config["policies"].items():
                initial_weight=value.get("initial_weight",None)
                if(initial_weight is not None):
                    dstDir=os.path.join(experiment_log_path,"policies","initial_weight")
                    if(not os.path.exists(dstDir)):
                        os.makedirs(dstDir)
                    shutil.copy2(initial_weight,os.path.join(dstDir,key+".dat"))
        else:
            #チェックポイントからの再開のとき、読み込み元からコピー
            for key in self.config["policies"].keys():
                initial_weight=os.path.join(restore_path,"policies","initial_weight",key+".dat")
                if(os.path.exists(initial_weight)):
                    dstDir=os.path.join(experiment_log_path,"policies","initial_weight")
                    if(not os.path.exists(dstDir)):
                        os.makedirs(dstDir)
                    shutil.copy2(initial_weight,os.path.join(dstDir,key+".dat"))
        #初期重みのパスの置換
        for key in self.config["policies"].keys():
            initial_weight=os.path.join(experiment_log_path,"policies","initial_weight",key+".dat")
            if(os.path.exists(initial_weight)):
                self.config["policies"][key]["initial_weight"]=initial_weight
            else:
                self.config["policies"][key]["initial_weight"]=None
        #weight_poolのコピー(チェックポイントから再開するときのみ)
        if(restore_path is not None):
            if(os.path.exists(os.path.join(restore_path,"policies","weight_pool"))):
                dstDir=os.path.join(experiment_log_path,"policies")
                if(not os.path.exists(dstDir)):
                    os.makedirs(dstDir)
                shutil.copytree(
                    os.path.join(restore_path,"policies","weight_pool"),
                    os.path.join(dstDir,"weight_pool")
                )
        self.policy_config=self.config["policies"]
        #MatchMaker(ray Actor)の生成(configで"match_maker"を与えた場合のみ)
        matchMakerConfig=self.config.get("match_maker",None)
        if(matchMakerConfig is not None):
            if(self.isSingleAgent):
                raise ValueError("MatchMaker cannot be used under 'single agent' condition.")
            matchMakerConfig["policy_config"]=self.config["policies"]
            matchMakerConfig["weight_pool"]=os.path.join(experiment_log_path,"policies","weight_pool")
            matchMakerConfig["log_prefix"]=os.path.join(experiment_log_path,"matches","matches")
            if(restore_path is not None):
                checkpoint_dirs=[os.path.basename(c)
                    for c in glob.glob(os.path.join(restore_path,"matches","checkpoint_*"))]
                checkpoint_dir=checkpoint_dirs[[int(c[11:]) for c in checkpoint_dirs].index(restore_checkpoint_number)]
                #MatchMakerの再開時、restore以外は上書き対象のキーのみ指定する。
                matchMakerConfig={
                    "restore":os.path.join(restore_path,"matches",checkpoint_dir,"MatchMaker-{}.dat".format(restore_checkpoint_number)),
                    "policy_config":matchMakerConfig["policy_config"],
                    "weight_pool":matchMakerConfig["weight_pool"],
                    "log_prefix":matchMakerConfig["log_prefix"]
                }
            else:
                matchMakerConfig["restore"]=None
            try:
                local_node="node:{}".format(node_ip_address)
                self.matchMaker=ray.remote(name="MatchMaker",resources={local_node:0.01})(self.matchMakerClass).remote(matchMakerConfig)
            except ValueError:
                self.matchMaker=ray.get_actor("MatchMaker")
            self.weight_pool=ray.get(self.matchMaker.get_weight_pool.remote())
        else:
            self.matchMaker=None
        #ダミー環境を一度生成し、完全な形のenv_configを取得する。
        dummyEnv=RayManager(EnvContext(
            self.config["trainer_common_config"]["env_config"],
            -1,
            -1
        ))
        self.completeEnvConfig={
            key:value for key,value in self.config["trainer_common_config"]["env_config"].items() if key!="config"
        }
        self.completeEnvConfig["config"]={
                "Manager":dummyEnv.getManagerConfig()(),
                "Factory":dummyEnv.getFactoryModelConfig()()
            }
        self.config["trainer_common_config"]["env_config"]=self.completeEnvConfig
        def policyMapper(agentId,episode=None,**kwargs):
            """
            エージェントのfullNameから対応するポリシー名を抽出する関数。
            agentId=agentName:modelName:policyName
            """
            agentName,modelName,policyName=agentId.split(":")
            if(policyName=="Internal"):
                policyName=modelName
            return policyName
        #初期重みの読み込みとweight_poolへの追加
        initial_weights=loadInitialWeights(self.policy_config)
        populate_config={}
        if(restore_path is None and self.matchMaker is not None):
            populate_config=ray.get(self.matchMaker.checkInitialPopulation.remote())
        for policyName,conf in populate_config.items():
            if(policyName in initial_weights):
                populate(self.weight_pool, policyName, initial_weights[policyName], conf['weight_id'])
        #Trainerインスタンスの生成
        trainer_index=0
        trainers={}
        refresh_interval=self.config.get("refresh_interval",0)
        if(refresh_interval>=1):
            #リフレッシュ有りの場合
            trainer_construct_info={}
        policies_on_which_trainer={}
        for trainerName,trainerSpec in self.config["trainers"].items():
            trainer_config=copy.deepcopy(self.config["trainer_common_config"])
            trainer_config["env"]=envName
            if("seed" in self.config):
                trainer_config["env_config"]["config"]["Manager"]["seed"]=self.config["seed"]+trainer_index
            elif(not "seed" in trainer_config["env_config"]["config"]["Manager"]):
                trainer_config["env_config"]["config"]["Manager"]["seed"]=np.random.randint(2**31)+trainer_index
            tc_overrider=trainerSpec.get("config_overrider",None)
            if(tc_overrider is not None):
                trainer_config.update(tc_overrider)
            policies_to_train=trainerSpec.get("policies_to_train",[])
            #Policyインスタンスの生成
            policies=self._policySetup(trainerSpec["trainer_type"],trainer_config,policies_to_train)
            if(len(policies_to_train)==0):
                policies["DummyPolicyForEmptyTrainer"]=[self.availableTrainers[trainerSpec["trainer_type"]].get("internal",DummyInternalRayPolicy),gym.spaces.Discrete(1),gym.spaces.Discrete(1),{}]
                policies_to_train=["DummyPolicyForEmptyTrainer"]
            print("===========policies for ",trainerName,"===========")
            print(list(policies))
            print("===========policies for ",trainerName,"===========")
            #SimulationManagerのインスタンスごとにシードを変えるためのoverriderを追加
            original_overrider=trainer_config["env_config"].get("overrider",lambda c,w,v:c)
            def overrider(config,worker_index,vector_index):
                nw=trainer_config["num_workers"]
                if("seed" in config["Manager"]):
                    config["Manager"]["seed"]=config["Manager"]["seed"]+worker_index+nw*vector_index
                config=original_overrider(config,worker_index,vector_index)
                return config
            trainer_config["env_config"]["overrider"]=overrider
            if(not self.isSingleAgent):
                trainer_config["multiagent"]={
                    "policies":policies,
                    "policy_mapping_fn":policyMapper,
                    "policies_to_train":policies_to_train
                }    
            else:
                # シングルエージェントとして扱う場合、policies及びpolicies_to_trainには一つのみ指定する。
                # なお、そのポリシーは、TrainerとしてはDEFAULT_POLICY_ID="default_policy"という名称で保持することになる。
                if(len(policies)!=1 or len(policies_to_train)!=1):
                    raise ValueError("You must specify exactly 1 policy for each trainer, under 'single agent' condition.")
                trainer_config["multiagent"]={
                    "policies":{DEFAULT_POLICY_ID:next(iter(policies.values()))}
                }
            #rayのTrainer側にMatchMaker用のコールバックを追加(configに"match_maker"を与えた場合のみ)
            if(self.matchMaker is not None):
                trainer_config["callbacks"]=getMatchMakerOnRayClass(trainerName,self.matchMonitorClass)
            #各Trainerインスタンスのログ保存ディレクトリの設定
            def get_logger_creator(trainerName_):
                experiment_log_path_for_trainer=os.path.join(experiment_log_path,"trainers",trainerName_)
                if not os.path.exists(experiment_log_path_for_trainer):
                    os.makedirs(experiment_log_path_for_trainer)
                def logger_creator(config):
                    return UnifiedLogger(config, experiment_log_path_for_trainer, loggers=None)
                return logger_creator
            on_which_node=trainerSpec.get("node_designation",None)#ノードのIPアドレスを指定
            trainerClass=self.availableTrainers[trainerSpec["trainer_type"]]["trainer"]
            #remote_workersへの初期重み分配用に、remoteからsync_weightsを実行可能にする
            def force_sync_weights(self,policies=None):
                self.workers.sync_weights(policies)
            trainerClass.force_sync_weights=force_sync_weights
            if(on_which_node is None):
                #動作するノードを指定しなかった場合
                remoteTrainerClass=ray.remote(num_gpus=trainer_config["num_gpus"])(trainerClass)
            else:
                #動作するノードを指定した場合
                if(on_which_node==entrypoint_ip_address):
                    on_which_node=node_ip_address #127.0.0.1対策
                remoteTrainerClass=ray.remote(num_gpus=trainer_config["num_gpus"],resources={"node:{}".format(on_which_node): 0.01})(trainerClass)
            logger_creator=get_logger_creator(trainerName)
            if(refresh_interval>=1):
                #リフレッシュ有りの場合、各Trainerの再構成に必要な情報を残しておく
                trainer_construct_info[trainerName]={
                    "class":remoteTrainerClass,
                    "config":trainer_config,
                    "logger_creator":logger_creator
                }
            trainers[trainerName]=remoteTrainerClass.remote(config=trainer_config,logger_creator=logger_creator)
            for policyName in policies_to_train:
                if(policyName in policies_on_which_trainer):
                    raise ValueError("A policy can be trained by only one trainer.")
                policies_on_which_trainer[policyName]=trainerName
            #初期重みのTrainerへのセット
            initial_weights_for_trainer={key:initial_weights[key] for key in policies_to_train if key in initial_weights}
            if(len(initial_weights_for_trainer)>0):
                if(self.isSingleAgent):
                    initial_weights_for_trainer={DEFAULT_POLICY_ID:next(iter(initial_weights_for_trainer))}
                ray.get(trainers[trainerName].set_weights.remote(initial_weights_for_trainer))
                ray.get(trainers[trainerName].force_sync_weights.remote(list(initial_weights_for_trainer.keys())))
            #チェックポイントから再開する場合、ここで読み込みを実施
            if(restore_path is not None):
                checkpoints=[c
                    for c in glob.glob(os.path.join(restore_path,"trainers",trainerName,"**","checkpoint-*"),recursive=True)
                    if not ".tune_metadata" in c]
                checkpoint=checkpoints[[int(os.path.basename(c)[11:]) for c in checkpoints].index(restore_checkpoint_number)]
                ray.get(trainers[trainerName].restore.remote(checkpoint))
            trainer_index+=1
        trainingWeights={}
        for trainerName,trainer in trainers.items():
            if(len(self.config["trainers"][trainerName]["policies_to_train"])>0):
                if(self.isSingleAgent):
                    policyName=next(iter(self.config["trainers"][trainerName]["policies_to_train"]))
                    trainingWeights.update({policyName:ray.get(trainer.get_weights.remote([DEFAULT_POLICY_ID]))[DEFAULT_POLICY_ID]})
                else:
                    trainingWeights.update(ray.get(trainer.get_weights.remote(self.config["trainers"][trainerName]["policies_to_train"])))
        if(self.matchMaker is not None):
            ray.get(self.matchMaker.uploadTrainingWeights.remote(trainingWeights))
        #学習の実行
        latestCheckpoint={}
        for i in range(self.config["train_steps"]):
            if(restore_path is None):
                total_steps=i+1
            else:
                total_steps=restore_checkpoint_number+i+1
            results=ray.get([trainer.train.remote() for trainer in trainers.values()])
            for trainerName,trainer in trainers.items():
                if(len(self.config["trainers"][trainerName]["policies_to_train"])>0):
                    if(self.isSingleAgent):
                        policyName=next(iter(self.config["trainers"][trainerName]["policies_to_train"]))
                        trainingWeights.update({policyName:ray.get(trainer.get_weights.remote([DEFAULT_POLICY_ID]))[DEFAULT_POLICY_ID]})
                    else:
                        trainingWeights.update(ray.get(trainer.get_weights.remote(self.config["trainers"][trainerName]["policies_to_train"])))
            if(self.matchMaker is not None):
                ray.get(self.matchMaker.uploadTrainingWeights.remote(trainingWeights))
            if(total_steps%self.config["checkpoint_interval"]==0 or (i+1)==self.config["train_steps"]):
                #途中経過の保存(デフォルトでは"~/ray_results/以下に保存される。)
                for trainerName,trainer in trainers.items():
                    latestCheckpoint[trainerName]=ray.get(trainer.save.remote())
                #重みとMatchMakerの保存
                checkpoint_dir=TrainableUtil.make_checkpoint_dir(os.path.join(experiment_log_path,"policies/checkpoints"),total_steps)
                for key,weights in trainingWeights.items():
                    filename=key+".dat"
                    saveWeights(weights,os.path.join(checkpoint_dir,filename))
                if(self.matchMaker is not None):
                    checkpoint_dir=TrainableUtil.make_checkpoint_dir(os.path.join(experiment_log_path,"matches"),total_steps)
                    makerfilename="MatchMaker-{}.dat".format(total_steps)
                    latestCheckpoint["MatchMaker"]=os.path.join(checkpoint_dir,makerfilename)
                    ray.get(self.matchMaker.save.remote(latestCheckpoint["MatchMaker"]))
                print("Checkpoint at ",total_steps,"-th step is saved.")
            if(refresh_interval>=1 and total_steps%refresh_interval==0 and (i+1)<self.config["train_steps"]):
                #Trainerのリフレッシュ(ray1.8.0時点で存在するメモリリーク対策)
                #チェックポイントを作成し、新たなTrainerインスタンスで読み直すことで引き継ぐ
                print("===================refresh!==============")
                #このステップでチェックポイントが未作成の場合、作成
                if(total_steps%self.config["checkpoint_interval"]==0 or (i+1)==self.config["train_steps"]):
                    pass
                else:
                    for trainerName,trainer in trainers.items():
                        latestCheckpoint[trainerName]=ray.get(trainer.save.remote())
                    checkpoint_dir=TrainableUtil.make_checkpoint_dir(os.path.join(experiment_log_path,"policies","checkpoints"),total_steps)
                    for key,weights in trainingWeights.items():
                        filename=key+".dat"
                        saveWeights(weights,os.path.join(checkpoint_dir,filename))
                    if(self.matchMaker is not None):
                        checkpoint_dir=TrainableUtil.make_checkpoint_dir(os.path.join(experiment_log_path,"matches"),total_steps)
                        makerfilename="MatchMaker-{}.dat".format(total_steps)
                        latestCheckpoint["MatchMaker"]=os.path.join(checkpoint_dir,makerfilename)
                        ray.get(self.matchMaker.save.remote(latestCheckpoint["MatchMaker"]))
                #Trainerインスタンスの再生成とチェックポイントの読み込み
                #MatchMakerはチェックポイントの保存だけで、読込は行わない。
                print("latestCheckpoint=",latestCheckpoint)
                for trainerName,trainer in trainers.items():
                    ray.get(trainer.cleanup.remote())
                    ray.kill(trainer)
                trainers={}
                for trainerName,info in trainer_construct_info.items():
                    trainers[trainerName]=info["class"].remote(config=info["config"],logger_creator=info["logger_creator"])
                    ray.get(trainers[trainerName].restore.remote(latestCheckpoint[trainerName]))
        #学習終了時の重み保存先が指定されているものについて、指定されたパスへ保存する。
        for key,value in self.config["policies"].items():
            save_path_at_the_end=value.get("save_path_at_the_end",None)
            if(save_path_at_the_end is not None and key in trainingWeights):
                weights=trainingWeights[key]
                saveWeights(weights,save_path_at_the_end)
        print("Training is finished.")
        [trainer.cleanup.remote() for trainer in trainers.values()]
