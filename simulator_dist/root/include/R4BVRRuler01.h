// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "MathUtility.h"
#include "Utility.h"
#include "Ruler.h"
namespace py=pybind11;
namespace nl=nlohmann;
class PhysicalAsset;

DECLARE_CLASS_WITH_TRAMPOLINE(R4BVRRuler01,Ruler)
    public:
	/*令和４年度版の空対空目視外戦闘のルール(その1)。得点は陣営(team)単位
	1. 座標系はNEDで、地表面をz=0とする。戦域の中心はx=y=0とする。
	2. 戦域の中心から東西(y軸)方向に±dLine[m]の位置に引かれた直線を各々の防衛ラインとする。
	3. 南北(x軸)方向は±dOut[m]までを戦域とする。
    4. 全滅判定や突破判定及び関連した得点増減の計算対象とする機体を、Factoryに登録されたモデル名単位で指定可能である。
        ただし、後述の場外ペナルティについては全ての機体に対して与える。
    5. 終了条件はステップの終了時に以下のいずれかを満たしていた場合とし、終了時の得点が高い陣営を勝者とする。同点の場合は引き分けとする。
        (1) いずれかの陣営の全滅判定対象機が全滅（被撃墜または墜落）し、かつ以後の撃墜の可能性が消滅したとき
        (2) いずれかの陣営の突破判定対象機が相手の防衛ラインを突破し、かつ双方の以後の撃墜の可能性が消滅したとき
        (3) 戦闘開始からminTimeが経過した以降において、両陣営の全機が各々の防衛ラインより後方まで下がったとき
        (4) 制限時間maxTimeが経過したとき
        (5) いずれかの陣営の得点がpDisq以下となったとき
        ただし、以下のフラグをmodelConfigで指定することにより終了条件を変更可能とする。
        (a) enableAdditionalTime (bool)・・・Falseとした場合、(1)及び(2)から「以後の撃墜の可能性が消滅したとき」という条件を削除する。
        (b) terminalAtElimination (bool)・・・Falseとした場合、(1)の条件を削除する。
        (c) terminalAtBreak (bool)・・・Falseとした場合、(2)の条件を削除する。
        なお、「以後の撃墜の可能性が消滅したとき」とは、「生存している陣営に対立する陣営の航空機、飛翔中誘導弾が全て消滅したとき」を指すものとする。
    6. ステップの終了時に同時に複数の終了条件が満たされていた場合は、以下の優先度とする。
        高　(1) > (2) > (5) > (3) > (4)　低
    7. modelConfigにおいてconsiderFuelConsumptionフラグをTrueとした場合、各機の燃料消費を考慮する。
        燃料消費を考慮する場合、終了時に生存していた機体は自陣営の基地に帰還するために必要な燃料を残しているかどうかを判定され、
        残燃料が不十分だった機体は帰還不可として扱い、得点計算においては墜落したものとみなす。
    8. 得点計算は以下の通りとする。
        1. 相手を撃墜した数1機につき、その機体種別に応じて定義されるpHit点を加算する。(撃墜された側は増減なし)
        2. 突破判定対象機が防衛ラインを突破したとき、突破した側の陣営にpBreak点を加算する。
        3. 終了条件(1)を満たしたとき、未突破の陣営に「そこから突破して更に帰還可能な」突破判定対象機が存在している場合、pBreak点を加算する。(残った側が以後妨害を受けずに突破できるとみなせるため。)
        4. 終了時に生存しており帰還可能な全滅判定対象機があれば、その機体種別に応じて定義されるpAlive点を加算する。
        5. 終了条件(4)又は(5)を満たして終了した若しくは両陣営が全滅したときに、いずれの陣営も未突破だったならば、両陣営の最前線にいた機体の直近の位置どうしを結んだ線分の中点のy座標の絶対値に応じ、より進出している陣営に1kmあたりpAdv点を加算する。
        6. ペナルティとして、随時以下の減点を与える。
            (a) 全滅判定対象機が墜落(地面に激突等)したとき、その機体種別に応じて定義されるpCrash点を減算する。
            (b) 任意の機体が各内部ステップの終了時に南北方向の場外に出ていたとき、1秒、1kmにつきpOut点を減算する。
    9. 得点計算のバリアントとして、全滅時の総増減量を機数によらず一定とするために、
        pHit,pCrash,pAliveを「1機あたりの増減量」でなく「陣営全体での総増減量」として扱うモードも使用可能とする。
        modelConfigにおいて"pHitPerAircraft","pCrashPerAircraft","pAlivePerAircraft"がそれぞれTrueのとき前者、Falseのとき後者とする。
	*/
    enum class DownReason{
        CRASH,
        HIT
    };
    enum class EndReason{
        ELIMINATION,
        BREAK,
        TIMEUP,
        WITHDRAWAL,
        PENALTY,
        NOTYET
    };
    bool debug;
    double minTime;
    double dLine,dOut,hLim;
    std::string westSider,eastSider;
    double pDisq;//失格となる得点(陣営単位)
    double pBreak;//突破時の得点(陣営単位)
    std::map<std::string,std::vector<std::string>> modelNamesToBeConsideredForBreak;//突破判定対象のFactoryモデルを陣営ごとに指定。"Any"を全モデルを対象とすることを表す予約語として扱う。
    std::map<std::string,std::vector<std::string>> modelNamesToBeExcludedForBreak;//突破判定対象から除外するFactoryモデルを陣営ごとに指定。
    std::map<std::string,double> pHit;//被撃墜時の相手陣営への加点。Factoryモデルごとに設定し、省略時の得点として"Default"を予約語として扱う。なお、後述の全滅判定対象になっているモデルのみ加点対象となる。
    std::map<std::string,std::map<std::string,double>> pHitScale;//陣営、Factoryモデルごとの各機の被撃墜時の相手陣営への加点の倍率。
    std::map<std::string,double> pCrash;//墜落時の自陣営への減点。Factoryモデルごとに設定し、省略時の得点として"Default"を予約語として扱う。なお、後述の全滅判定対象になっているモデルのみ減点対象となる。
    std::map<std::string,std::map<std::string,double>> pCrashScale;//陣営、Factoryモデルごとの各機の墜落時の自陣営への減点の倍率。
    std::map<std::string,double> pAlive;//終了時の自陣営への加点(生存点)。Factoryモデルごとに設定し、省略時の得点として"Default"を予約語として扱う。なお、後述の全滅判定対象になっているモデルのみ加点対象となる。
    std::map<std::string,std::map<std::string,double>> pAliveScale;//陣営、Factoryモデルごとの各機の墜落時の自陣営への減点の倍率。
    std::map<std::string,std::vector<std::string>> modelNamesToBeConsideredForElimination;//全滅判定対象のFactoryモデルを陣営ごとに指定。"Any"を全モデルを対象とすることを表す予約語として扱う。
    std::map<std::string,std::vector<std::string>> modelNamesToBeExcludedForElimination;//全滅判定対象から除外するFactoryモデルを陣営ごとに指定
    double pAdv;//優勢度による得点(陣営単位)
    double pOut;//場外のペナルティ(Factoryモデルに依存せず、機体単位で計算)
    std::map<std::string,std::map<std::string,int>> crashCount,hitCount;//[team,[model,count]]
    std::map<std::string,double> leadRange;
    std::map<std::string,double> lastDownPosition;
    std::map<std::string,DownReason> lastDownReason;
    std::map<std::string,double> outDist;
    std::map<std::string,double> eliminatedTime;
    std::map<std::string,double> breakTime;
    std::map<std::string,double> disqTime;
    std::map<std::string,Eigen::Vector2d> forwardAx,sideAx;
    EndReason endReason,endReasonSub;
    //機数が可変の時に、全滅時の総得点を機数非依存にするようにするためのフラグ3種
    //それぞれTrueとした場合、pHit,pCrash,pAliveは1機あたりの増減でなく各陣営、各機種全体での総増減量として扱う
    bool pHitPerAircraft;
    bool pCrashPerAircraft;
    bool pAlivePerAircraft;
    bool enableAdditionalTime;//相討ちの有無を確認するために、一方が全滅又は突破後の計算を継続するかどうか。
    bool terminalAtElimination;//終了条件(1)を有効とするかどうか
    bool terminalAtBreak;//終了条件(2)を有効とするかどうか
    bool considerFuelConsumption;//燃料消費を考慮するかどうか。
    double fuelMargin;//帰還可否判定時の燃料消費量のマージン。最適巡航時の(1+fuelMargin)倍の燃料を消費するものとする。
    std::map<std::string,double> distanceFromBase;//各陣営の防衛ラインから基地までの距離。帰還に必要な燃料を計算するために指定する。また、進出に要する燃料を消費した状態で開始する。
    std::vector<std::string> deadFighters;//撃墜or墜落済のFighterのfullNameのリスト。得点の二重計算防止用。
    //constructors & destructor
    R4BVRRuler01(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~R4BVRRuler01();
    //functions
    void debugPrint(const std::string& reason,const std::string& team,double value);
    virtual void onCrash(const nl::json& args);
    virtual void onHit(const nl::json& args);
    virtual void onEpisodeBegin();
    virtual void onValidationEnd();
    virtual void onInnerStepBegin();
    virtual void onInnerStepEnd();
    virtual void checkDone();
    virtual void _setupPDownConfig(
        std::map<std::string,double>& _config,
        const nl::json& _modelConfig,
        const std::string& _key,
        double _defaultValue);
    virtual void _setupPDownScale(
        std::map<std::string,std::map<std::string,double>>& _scale,
        const std::map<std::string,double>& _config,
        bool _perAircraft);
    virtual double _getPDownImpl(
        const std::map<std::string,std::map<std::string,double>>& _scale,
        const std::map<std::string,double>& _config,
        const std::string& _team,
        const std::string& _modelName) const;
    virtual double getPHit(const std::string& team,const std::string& modelName) const;
    virtual double getPCrash(const std::string& team,const std::string& modelName) const;
    virtual double getPAlive(const std::string& team,const std::string& modelName) const;
    virtual int& getCrashCount(const std::string& team,const std::string& modelName);
    virtual int& getHitCount(const std::string& team,const std::string& modelName);
    virtual bool isToBeConsideredForBreak(const std::string& team,const std::string& modelName);
    virtual bool isToBeConsideredForElimination(const std::string& team,const std::string& modelName);
    virtual bool checkHitPossibility() const;
    virtual bool isReturnableToBase(const std::shared_ptr<PhysicalAsset>& asset) const;
    virtual bool isBreakableAndReturnableToBase(const std::shared_ptr<PhysicalAsset>& asset) const;
};
DECLARE_TRAMPOLINE(R4BVRRuler01)
    virtual void onCrash(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onCrash,args);
    }
    virtual void onHit(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onHit,args);
    }
    virtual void _setupPDownConfig(
        std::map<std::string,double>& _config,
        const nl::json& _modelConfig,
        const std::string& _key,
        double _defaultValue) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,_setupPDownConfig,_config,_modelConfig,_key,_defaultValue);
    }
    virtual void _setupPDownScale(
        std::map<std::string,std::map<std::string,double>>& _scale,
        const std::map<std::string,double>& _config,
        bool _perAircraft) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,_setupPDownScale,_scale,_config,_perAircraft);
    }
    virtual double _getPDownImpl(
        const std::map<std::string,std::map<std::string,double>>& _scale,
        const std::map<std::string,double>& _config,
        const std::string& _team,
        const std::string& _modelName) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,_getPDownImpl,_scale,_config,_team,_modelName);
    }
};
PYBIND11_MAKE_OPAQUE(std::map<std::string,R4BVRRuler01::DownReason>);

void exportR4BVRRuler01(py::module& m);
