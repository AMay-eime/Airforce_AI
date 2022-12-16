# インストール方法
以下は一例。
## C++依存ライブラリの準備
### Boost
例えばUbuntuの場合、
```
sudo apt-get install libboost-dev
```
などでインストールしておく。  
なお、CMakeのfind_package(Boost)で見つけられるような状態にしておく必要がある。

### Eigen
https://eigen.tuxfamily.org/
に従いダウンロードし、インストールしておく。  
なお、CMakeのfind_package(Eigen3)で見つけられるような状態にしておく必要がある。

### NLopt
https://nlopt.readthedocs.io/
に従いダウンロードし、インストールしておく。  
特にWindowsの場合、パスの設定やDLLの適切な配置を忘れないこと。

### nlohmann json
https://github.com/nlohmann/json
から入手できる。  
ヘッダオンリーのため予めパスの通った場所に、
```c
#include <nlohmann/json.h>
```
で読み込めるように配置する。
### magic enum
https://github.com/Neargye/magic_enum
から入手できる。  
ヘッダオンリーのため予めパスの通った場所に、
```c
#include <magic_enum/magic_enum.hpp>
```
で読み込めるように配置する。
### thread-pool
https://github.com/bshoshany/thread-pool
から入手できる。  
ヘッダオンリーのため予めパスの通った場所に、
```c
#include <thread-pool/thread_pool.hpp>
```
で読み込めるように配置する。
## numpy,pybind11のインストール
この二つはビルド時に必要なため事前にインストールが必要。
```
pip install numpy pybind11>=2.6.2
```
## シミュレータのビルド・インストール
リポジトリ直下で
```
pip install .
```
によりビルド及びインストールできる。
インストール対象のアドオンは、デフォルトではsetup.pyの冒頭に記載されたinclude_addonsに書かれているもののみである。
アドオンを追加したい場合は
```
pip install . --global-option="--addons=addon1,addon2"
```
のようにコンマ区切りで指定すると追加できる。また、除外したい場合も同様に
```
pip install . --global-option="--ex-addons=addon3,addon4"
```
のようにコンマ区切りで指定できる。なお、両方のオプションに同じアドオン名を指定すると除外として扱われる。

また、C++部分をデバッグビルドとしたい場合は、
```
pip install . --global-option="--Debug"
```
とする。WindowsでMSVCでなくMSYSを用いるときは
```
pip install . --install-option="--MSYS"
```
とする。
なお、上記のオプションは競合しないため共存できる。

## 深層学習用フレームワークのインストール
シミュレータ及びサンプルの全ての機能を使用するためにはPyTorchとTensorflow(Tensorboard)が必要であるため、
各ユーザーは自身の所望のバージョンのこれらのフレームワークを別途インストールしておく必要がある。
なお、PyTorchについては一部のアドオンのrequirementsに含まれているため、未インストールだった場合はpip経由で
自動的にインストールされる。

## サンプルモジュールのビルド・インストール
本シミュレータは独自の行動判断モデルや報酬関数等を定義するためのサンプルとして、独立したPythonモジュールとして
動作する形式のサンプルをsample/OriginalModelSampleとして同梱している。学習サンプルを動作させるためにはインストールが必須であり、sample/OriginalModelSampleに移動し、シミュレータ本体と同様に
```
pip install .
```
でビルド及びインストールできる。
## サンプルの実行

### 1. シミュレータとしての動作確認用サンプル

sample/MinimumEvaluationに移動し、
```
python evaluator.py "Rule-Fixed" "Rule-Fixed" -n 1 -v -l "./result"
```
を実行すると、ルールベースの初期行動判断モデルどうしの戦闘が可視化しながら1回行われ、その結果を保存したログが
./resultに保存される。GUIの無い環境で使用したい場合は、-vのオプションを省略することで可視化を無効化できる。

### 2. サンプルAgentモデルのobservation,actionの確認方法
上記のevaluator.pyのコマンドライン引数で"Rule-Fixed"はルールベースのためobservation,actionはともに空であるが、
これを学習サンプルで使用する"HandyRLSample01S"や"HandyRLSample01M"等に変更して実施することによって、
evaluator.pyのメインループの中でobservationやactionの内容を確認することができる。

### 3. HandyRLを用いた学習のサンプル

#### 1. 学習の実施
sample/HandyRLSampleに移動し、
```
python main.py R4_contest_sample_S.yaml --train
```
を実行すると、1体で1機を操作する行動判断モデルについて、Self-Playによる学習が行われる。
学習結果は./results/Single/YYYYmmddHHMMSS以下に保存される。

#### 2. 学習済モデルの評価
学習済モデルは、./results/Single/YYYYmmddHHMMSS/policies/checkpoints以下に保存されている。
このモデルの評価は上記のevaluator.pyを使用することで可能である。
sample/MinimumEvaluation/candidates.jsonを開き、例えば
```
"test":{
    "userModuleID":"HandyRLSample01S",
    "args":{"weightPath":<学習済の.pthファイルのフルパス>}
}
```
のように候補を追加し、sample/MinimumEvaluation上で
```
python evaluator.py "test" "Rule-Fixed" -n 10 -v -l "eval_test.csv"
```
と実行すると、初期行動判断モデルとの対戦による学習済モデルの評価を行うことができる。

### 4. ray RLlibを用いた学習のサンプル

#### 1. 学習の実施
sample/raySampleに移動し、
```
python LearningSample.py APPO_sample_S.yaml
```
を実行すると、1体で1機を操作する行動判断モデルについて、Self-Playによる学習が行われる。
学習結果は./results/Single/APPO/run_YYYY-mm-dd_HH-MM-SS以下に保存される。

#### 2. 学習済モデルの評価
学習済モデルは、./results/Single/APPO/run_YYYY-mm-dd_HH-MM-SS/policies/checkpoints以下に保存されている。
このモデルの評価は上記のevaluator.pyを使用することで可能である。
sample/MinimumEvaluation/candidates.jsonを開き、例えば
```
"test":{
    "userModuleID":"raySample01S",
    "args":{"weightPath":<学習済の.datファイルのフルパス>}
}
```
のように候補を追加し、sample/MinimumEvaluation上で
```
python evaluator.py "test" "Rule-Fixed" -n 10 -v -l "eval_test.csv"
```
と実行すると、初期行動判断モデルとの対戦による学習済モデルの評価を行うことができる。
# competition-atla
