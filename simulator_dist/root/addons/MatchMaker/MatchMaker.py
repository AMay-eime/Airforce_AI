# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import copy
import cloudpickle
from collections import defaultdict
"""
複数のPolicyの重み保存、読み込みを管理しつつ、
その勝率等に応じたエピソードごとの対戦カードを生成するための基底クラス群であり、以下の2つのクラスからなる。
(1)一つだけ生成し、全体の管理を行うMatchMaker
(2)実際にエピソードを実行するインスタンスごとに生成し、対戦結果を抽出してMatchMakerに渡すresultを生成するMatchMonitor

<使用方法>
(1)MatchMaker.__init__について
コンストラクタはdict型の引数configをとる。必須のキーは以下の通り。
config={
    "restore": None or str, #チェックポイントを読み込む場合、そのパスを指定。
    "weight_pool": str, #重みの保存場所。<policy>-<id>.datのようにPolicy名と重み番号を示すファイル名で保存される。
    "policy_config": { #Policyに関する設定。Policy名をキーとしたdictで与える。
        <Policy's name>: Any, #具象クラスごとの様式に従った各Policyの設定
        ...
    },
    "match_config": Any #対戦カードの生成に関する指定。具象クラスで自由に定める
}
(2)MatchMakerとMatchMonitorの生成
学習プログラムにおいて、メインプロセスで一つだけMatchMakerインスタンスを生成する必要がある。
また、MatchMonitorはエピソードを生成するWorkerインスタンスごとに生成する必要がある。
(3)resultの伝達
学習プログラムに適した方法にて、各MatchMonitorのonEpisodeEndが抽出したエピソードの結果をMatchMakerのonEpisodeEndに供給する必要がある。
(4)対戦カードを環境に反映するための機構の準備
学習プログラムにおいて、生成された対戦カードを用いて環境の生成を行う機構も必要である。
(5)重みの読み書きや初期化に関する処理の実装
MatchMaker/MatchMonitorは特定の強化学習ライブラリに依存させないために、重みを直接扱わないようにしている。
そのため、重みの読み書きや初期化を実際に行う処理は学習プログラム側で準備する必要がある。

<カスタマイズ>
実際にMatchMakerを使用するうえでは、環境の仕様に応じて以下の関数群を適宜オーバーライドして使用する。
(1)MatchMaker.makeNextMatchのオーバーライド
対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeを引数にとり何らかの対戦カードを返す関数であり、各ユーザーが独自の対戦方式を実装するものである。
対戦カードはdictで表現され、場に登場するチームごとにどのPolicyのどの重みを使うかをキー"Policy"と"Weight"により指定する。
また、同名のPolicyで異なる重みを複数同時に使用したい場合等のために、それらを区別するための"Suffix"を指定可能である。
重み番号は基本的に
    負数(-1): 学習中の現物
    0: 学習中重みの最新のコピー(随時更新される)
    自然数: 過去のある時点で保存されたコピー
とすることを想定しているが、MatchMakerを使用する学習プログラムの書き方次第である。
また、これら以外に必要な情報があれば適宜追加してもよい。
 {
    <Team's name>: {
        "Policy": str, teamを動かすPolicyの名前
        "Weight": int, teamを動かすPolicyの重み番号
        "Suffix": str, teamを動かすPolicyの接尾辞。同名のPolicyで異なる重みを使用する場合等に指定する。省略は不可。
    }
}
(2)MatchMaker.onEpisodeEndのオーバーライド
いずれかの対戦が終わったときに呼び出し、その対戦を実施したインスタンスにおける次の対戦カードを生成して返す関数。
引数は、終了した対戦のmatchInfoと、対応するMatchMonitorのonEpisodeEndで生成されたresultを与えるものとする。
また、返り値として、重み保存要否を記したdictを返す必要がある。基本的には重み保存を行うPolicyのみを対象として以下のようなdictとして記述することを想定している。
{
    <Policy's name>: {
        "weight_id": int, # 保存先の重み番号
        "reset": bool, # 保存後に重みをリセットするかどうか
    }
}
(3)MatchMonitor.onEpisodeEndのオーバーライド
環境インスタンスごとに対戦結果を抽出してMatchMakerに渡すresultを生成する関数。
MatchMakerのonEpisodeEndで重み保存や次回の対戦カードの生成の判定を行うために必要な情報があれば、環境インスタンスから抽出して返す。
(4)MatchMaker.checkInitialPopulationのオーバーライド
学習開始時の重みをweight_poolに追加するかどうかを返す関数。返り値の形式はonEpisodeEndと同じ。
用途としては、別の学習済モデルを読み込んで開始したときにその初期状態を対戦候補に含めたいような場合等が考えられる。
(5)MatchMaker.get_metrics関数のオーバーライド
Tensorboardログ等に記録するための値を格納したdictを返す関数として、必要に応じてオーバーライドする。
(6)MatchMaker.initialize, load, saveのオーバーライド
MatchMakerの初期化やチェックポイントの生成、読み込みを行うための各関数を必要に応じてオーバーライドする。
"""

class MatchMaker:
    def __init__(self,config):
        restore=config.get("restore",None)
        if(restore is not None):
            """チェックポイントからの再開のときは、
                "restore"キーにチェックポイントのパスを指定し、
                それ以外のキーにconfigの上書きを行いたいキーのみ値を指定する。
            """
            self.load(restore,config)
        else:
            """チェックポイントからの再開でないときは、
                "restore"キーを省略するかNoneとする。
            """
            self.initialize(config)
    def initialize(self,config):
        self.resumed=False
        self.config=copy.deepcopy(config)
        self.weight_pool = self.config["weight_pool"] #重み保存先ディレクトリ
        self.policy_config=self.config["policy_config"]
        self.match_config=self.config.get("match_config",{})
        self.trainingWeights={} #各Policyの学習中重み(のコピー)
        self.trainingWeightUpdateCounts=defaultdict(lambda: -1) #各Policyの学習中重みの更新回数
    def load(self,path,configOverrider={}):
        """チェックポイントを保存する。その際、configOverriderで与えた項目はconfigを上書きする。
        """
        with open(path,"rb") as f:
            obj=cloudpickle.load(f)
            config=copy.deepcopy(obj["config"])
            config.update(configOverrider)
            self.initialize(config)
            self.resumed=True
        print("MatchMaker has been loaded from ",path)
    def save(self,path):
        """チェックポイントを読み込む。
        """
        with open(path,"wb") as f:
            obj={
                "config":self.config,
            }
            cloudpickle.dump(obj,f)
    def get_weight_pool(self):
        """重み保存先のディレクトリを返す。
        """
        return self.weight_pool
    def get_policy_config(self):
        """policy_configを返す。
        """
        return self.policy_config
    def makeNextMatch(self,matchType):
        """対戦カードを生成する。
        引数として、対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeも与えられるため、
        複数のグループを使い分ける場合は適宜参照しつつ生成する。。
        {
            team: {
                "Policy": str, teamを動かすPolicyの名前
                "Weight": int, teamを動かすPolicyの重み番号
                "Suffix": str, teamを動かすPolicyの接尾辞。同名のPolicyで異なる重みを使用する場合等に指定する。省略は不可。
            }
        }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        return {
            "Team1":{
                "Policy": "Learner",
                "Weight": -1,
                "Suffix": ""
            }
        }
    #
    # 対戦結果の処理
    #
    def onEpisodeEnd(self,match,result):
        """いずれかのgym.Envにおいて対戦が終わったときに呼ばれ、
        そのインスタンスにおける次の対戦カードを生成して返す関数。
        引数は、終了した対戦のmatchInfoと、対応するMatchMonitorのonEpisodeEndで生成されたresultを与える。
        また、返り値として、重み保存要否を記したdictを返す必要がある。
            populate_config = {
                policyName: {
                    "weight_id": int, # 保存先の重み番号
                    "reset": bool, # 保存後に重みをリセットするかどうか
                }
            }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        populate_config={}
        return populate_config
    def get_metrics(self,match,result):
        """SummaryWriter等でログとして記録すべき値をdictで返す。
        """
        return {}
    #
    # 初期重み保存要否の判定
    #
    def checkInitialPopulation(self):
        """開始時の初期重みをpopulateするかどうかを判定し、必要に応じweight_poolに追加する。
            populate_config = {
                policyName: {
                    "weight_id": int, # 保存先の重み番号
                    "reset": bool, # 保存後に重みをリセットするかどうか
                }
            }
        なお、重み番号の指定方法は以下の通り。
            負数(-1): 学習中の現物
            0: 学習中重みの最新のコピー(随時更新される)
            自然数: 過去のある時点で保存されたコピー
        """
        populate_config={}
        return populate_config
    #
    # 学習中重みの集中管理
    #
    def uploadTrainingWeights(self,trainingWeights):
        """指定したポリシー名の学習中重みを書き込む。
        そのポリシーを学習中のTrainer以外からweight_id=0で最新の重みを読み込もうとする場合には、
        この関数で書き込まれた重みを使用できるが、そのためには適切な頻度で外からこの関数を呼ぶ必要がある。
        """
        for policyName,weight in trainingWeights.items():
            self.trainingWeights[policyName]=weight
            self.trainingWeightUpdateCounts[policyName]+=1
    def getTrainingWeightUpdateCount(self,policyName):
        """指定したポリシー名の学習中重みの更新回数を返す。未登録が-1、初期状態が0である。
        """
        return self.trainingWeightUpdateCounts[policyName]
    def getLatestTrainingWeight(self,policyName):
        """指定したポリシー名の学習中重み(のコピー)を返す。
        """
        return self.trainingWeights.get(policyName,None)


class MatchMonitor:
    def __init__(self, env):
        self.env=env
    def onEpisodeBegin(self):
        pass
    def onEpisodeEnd(self,matchType):
        """MatchMakerのonEpisodeEndで使用するresultを生成して返す。
        引数として、この対戦を生成した対戦グループ(○○リーグのようなイメージ)を表す何らかの変数matchTypeも与えられる。
        """
        result = {}
        return result
