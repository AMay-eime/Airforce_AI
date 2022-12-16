# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法について説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成方法](#応募用ファイルの作成方法)

## 配布データ

配布されるデータは以下の通り.

- [readme](#readme)
- [説明資料](#説明資料)
- [シミュレータ](#シミュレータ)
- [応募用サンプルファイル](#応募用サンプルファイル)

### readme

本ファイル(readme.md)で, 配布用データの説明と応募用ファイルの作成方法を説明したドキュメント. マークダウン形式で, プレビューモードで見ることを推奨する.

### 説明資料

説明資料は"docs.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
docs
├─ 初期行動判断モデルの概要説明.pdf
├─ 配布シミュレータの取扱説明書.pdf
├─ 問題設定の概要説明.pdf
└─ 問題設定及び初期行動判断モデルに関する説明資料.pdf
```

内容は以下の通り.

#### 問題設定

本コンペティションにおける戦闘場面や勝敗に関するルールなどについては"問題設定の概要説明.pdf"で説明されている. より詳細な説明については"問題設定及び初期行動判断モデルに関する説明資料.pdf"を参照.

#### 配布シミュレータの取扱説明書

本コンペティションで使用するシミュレータの導入方法や仕様, 一般的な行動判断の実装方法などの詳細については"配布シミュレータの取扱説明書.pdf"を参照.

#### 初期行動判断モデル

シミュレータにデフォルトで実装されている初期行動判断モデルの説明については"初期行動判断モデルの概要説明.pdf"と"問題設定及び初期行動判断モデルに関する説明資料.pdf"を参照. なお, 本モデルは本コンペティションにおいてベンチマークとして参戦する.

### シミュレータ

空戦を再現するシミュレータ. "simulator_dist.zip"が本体で, 解凍すると, 以下のようなディレクトリ構造のデータが作成される.

```bash
simulator_dist
├── root
│   ├── addons                    : 本シミュレータの追加機能として実装されたサブモジュール
│   │   ├── AgentIsolation        : 行動判断モデルをシミュレータ本体と隔離して動作させるための評価用機能
│   │   ├── HandyRLUtility        : HandyRLを用いて学習を行うための拡張機能
│   │   ├── MatchMaker            : 複数の行動判断モデルを用いてSelf-Play等の対戦管理を行うための拡張機能
│   │   ├── rayUtility            : ray RLlibを用いて学習を行うための拡張機能
│   │   └── torch_truncnorm       : PyTorchで切断ガウス分布を用いるための拡張機能
│   ├── ASRCAISim1                : 最終的にPythonモジュールとしての本シミュレータが格納されるディレクトリ
│   ├── include                   : コア部を構成するヘッダファイル
│   ├── sample                    : 戦闘環境を定義し学習を行うためのサンプル
│   │   ├── HandyRLSample         : HandyRLを用いて基本的な強化学習を行うためのサンプル
│   │   ├── MinimumEvaluation     : 各行動判断モデルを隔離して対戦させる最小限の評価環境
│   │   ├── OriginalModelSample   : 独自のAgentクラスと報酬クラスを定義するためのサンプル
│   │   └── raySample             : rayを用いて基本的な強化学習を行うためのサンプル
│   ├── src                       : コア部を構成するソースファイル
│   └── thirdParty                : 外部ライブラリの格納場所(同梱は改変を加えたもののみ)
│　     └── include
│           └── pybind11_json　   ※オリジナルを改変したものを同梱
├── Dockerfile                    : 環境構築に必要なDockerfile
└── requirements.txt              : 環境構築に必要なPythonライブラリ一覧

```

使用方法の詳細については"配布シミュレータの取扱説明書.pdf"を参照すること.

### 応募用サンプルファイル

応募用のサンプルファイル. 実体は"sample_submit.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
sample_submit
├── Agent
│   ├── __init__.py
│   └── config.json
└── params.json
```

詳細や作成方法については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.

## 応募用ファイルの作成方法

学習済みモデルを含めた, 行動を実行するためのソースコード一式をzipファイルでまとめたものとする.

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定している.

```bash
.
├── Agent              必須: 学習済モデルを含んだソースコードなどを置くディレクトリ
│   ├── __init__.py  　必須: 実行時に最初に呼ばれるinitファイル
│   ├── config.json    任意: アルゴリズムに対する設定を含んだファイル　 
│   └── ...
└── params.json        必須: __init__内で定義されたメソッドに渡す情報を含んだファイル
```

- 学習済みモデルを含んだ実行ソースコードの格納場所は"Agent"ディレクトリを想定している.
  - 名前は必ず"Agent"とすること.
- エージェントに対する設定に関しては例えば"config.json"などの名前でファイルとして保存しておいて, 実行時に読み込んで使用する想定である.
  - "Agent"直下で任意に作成してもよい.
- `__init__.py`で定義されたメソッドに渡す情報は"params.json"を想定している.
  - 名前は必ず"params.json"とすること.
  - keyとして"args"を含めること.

### `__init__.py`の実装方法

以下のメソッドを実装すること. "配布シミュレータの取扱説明書.pdf"の7も参照.

#### getUserAgentClass

Agentクラスオブジェクトを返すメソッド. 以下の条件を満たす.

- 引数argsを指定する.
  - "params.json"において"args"の値が渡される想定である.
  - "args"をkeyとして含んでいない場合は`None`が渡される.
- Agentクラスオブジェクトを返す.
  - Agentクラスオブジェクトは自作してもよいし, もともと実装されているものを直接importして用いてもよい. Agentクラスの詳細は"配布シミュレータの取扱説明書.pdf"の4.5や5.1などを参照. また, 実装例についてはroot/sample/OriginalModelSample/以下も参照すること.

#### getUserAgentModelConfig

Agentモデル登録用にmodelConfigを表すjsonを返すメソッド. 以下の条件を満たす.

- 引数argsを指定する.
  - "params.json"において"args"の値が渡される想定である.
  - "args"をkeyとして含んでいない場合は`None`が渡される.
- modelConfigを返す. なお、modelConfigとは, Agentクラスのコンストラクタに与えられる二つのjson(dict)のうちの一つであり、設定ファイルにおいて

    ```json
    {
        "Factory":{
            "Agent":{
                "modelName":{
                    "class":"className",
                    "config":{...}
                }
            }
        }
    }
    ```

    の"config"の部分に記載される`{...}`のdictが該当する. "配布シミュレータの取扱説明書.pdf"の4.1なども参照すること.  

#### isUserAgentSingleAsset

Agentの種類(一つのAgentインスタンスで1機を操作するのか、陣営全体を操作するのか)を返すメソッド. 以下の条件をみたす.

- 引数argsを指定する.
  - "params.json"において"args"の値が渡される想定である.
  - "args"をkeyとして含んでいない場合は`None`が渡される.
- 1機だけならばTrue, 陣営全体ならばFalseを返す想定である.

#### getUserPolicy

StandalonePolicyを返すメソッド. StandalonePolicyについては, "配布シミュレータの取扱説明書.pdf"の2.6などを参照すること. 以下の条件を満たす.

- 引数argsを指定する.
  - "params.json"において"args"の値が渡される想定である.
  - "args"をkeyとして含んでいない場合は`None`が渡される.
- StandalonePolicyを返す.

以下は`__init__.py`の実装例.

```Python
import os,json
import ASRCAISim1
from ASRCAISim1.policy import StandalonePolicy


def getUserAgentClass(args=None):
    from ASRCAISim1 import R4InitialFighterAgent01
    return R4InitialFighterAgent01


def getUserAgentModelConfig(args=None):
    configs=json.load(open(os.path.join(os.path.dirname(__file__),"config.json"),"r"))
    modelType="Fixed"
    if(args is not None):
        modelType=args.get("type",modelType)
    return configs.get(modelType,"Fixed")


def isUserAgentSingleAsset(args=None):
    #1機だけならばTrue,陣営全体ならばFalseを返すこと。
    return True


class DummyPolicy(StandalonePolicy):
    """
    actionを全く参照しない場合、適当にサンプルしても良いし、Noneを与えても良い。
    """
    def step(self,observation,reward,done,info,agentFullName,observation_space,action_space):
        return None

def getUserPolicy(args=None):
    return DummyPolicy()
```

応募用サンプルファイル"sample_submit.zip"も参照すること. また, /root/sample/MinimumEvaluation以下にあるHandyRLSample01SやRaySample01Sなどのサンプルモジュールも参考にされたい. なお, HandyRLとrayRLlibについてはサンプルで使用しているNNクラスに対応したStandalonePolicyの派生クラスが提供されている(/root/addons/HandyRLUtility, /root/addons/rayUtility/extension以下を参照.).

### 動作テスト

行動を実行するためのプログラムが実装できたら, 正常に動作するか確認する.

#### 環境構築

"配布シミュレータの取扱説明書.pdf"の1.2に従ってシミュレータを動かすための環境を構築する.

#### 対戦の実行

初期行動判断モデルと対戦を実行し, 対戦結果を確認する.

```bash
cd /path/to/root/sample/MinimumEvaluation
python run.py  --exec-path /path/to/submit --log-dir /path/to/log/dir --replay replay --time-out time_out --memory-limit memory_limit
```

- 引数"--exec-path"には実装したプログラム("Agent")が存在するパス名を指定する.
- 引数"--log-dir"には戦闘終了後保存されるログデータの保存先のパス名を指定する. デフォルトは"./log".
- 引数"--replay"には後で戦闘結果の可視化を行うためのデータを保存するか否かを指定する. 1は保存し, 0は保存しない. 保存先は"--log-dir"で指定したパス.
- 引数"--time-out"には1ステップ(アルゴリズムが観測情報を受け取って行動を返すまでの間隔)のタイムアウト時間(秒)を指定する. 指定した時間までに行動を返さなければ行動空間よりランダムサンプリングされる. デフォルトは0.5[s].
- 引数"--memory-limit"には1対戦(初期行動判断モデルとの対戦)において使用するメモリの上限を指定する. モデル等の読み込みを含めて指定の値を超えると無効とする想定である. デフォルトでは7.0[GB].

実行に成功すると, 対戦結果が出力され, 詳細は"--log-dir"で指定した場所に以下の情報がcsvファイルとして保存される.

- Episode
- score[Blue, Red]
- totalReward[Blue_Agent, Red_Agent]
- finishedTime[s]
- numSteps
- calcTime[s]
- BlueWin
- WinCount
- WinRate[%]
- numAlives[Blue, Red]
- endReason

引数"--replay"に1を指定した場合は"--log-dir"で指定した場所に可視化の情報をまとめたdatファイルが保存される. なお, **GUI対応している環境が整っていることが前提である**が, 作成したdatファイルから以下のコマンドにより, 可視化結果を動画, または画像として保存することができ, 閲覧することが可能.

```bash
python replay.py --movie-dir /path/to/movie/dir --as-video as_video
```

- 引数"--movie-dir"にはdatファイルの保存先のパス名を指定する. デフォルトでは"./log".
- 引数"--as-video"には結果を動画として保存するか(連番)静止画像として保存するか指定する. 1なら動画, 0なら静止画像. デフォルトでは1.

GUI対応していない場合, Xvfbでヘッドレスに実行することで同様の結果が得られる.

```bash
xvfb-run python replay.py --movie-dir /path/to/movie/dir --as-video as_video
```

Dockerで環境構築した場合, Xvfbはあらかじめインストールされているが, ない場合は, 実行する前にインストールしておく必要がある.

実行に成功すると, 動画の場合はmp4ファイルが, 静止画像の場合はpngファイルが--movie-dirで指定したディレクトリ以下に保存される. 可視化結果の例としては"配布シミュレータの取扱説明書.pdf"の6.4を参照.

投稿する前にエラーなどが出ずに対戦結果が出力されることを確認すること.

### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する.
