// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Reward.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Missile.h>
namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITHOUT_TRAMPOLINE(R4RewardSample02,AgentReward)
	/*いくつかの観点に基づいた個別報酬の実装例。
    (1)敵ミサイルの回避成功
    (2)被撃墜
    (3)墜落
    (4)撃墜
    (5)場外
    (6)終了時の生存
    (7)余剰燃料の有無
	*/
    public:
    //parameters
    double pAvoid;
    double pHitE_head,pHitE_tail;
    double pCrash;
    double pHit_head,pHit_tail;
    double pOut;
    double pAlive;
    double pFuelShortage;
    //internal variables
    std::string westSider,eastSider;
    double dOut,dLine;
    std::map<std::string,Eigen::Vector2d> forwardAx,sideAx;
    double fuelMargin;
    std::map<std::string,double> distanceFromBase;
    std::map<std::string,std::vector<std::weak_ptr<Fighter>>> parents;
    std::map<std::string,std::vector<std::weak_ptr<Missile>>> missiles;
    std::unordered_map<std::string,bool> checked;
    std::unordered_map<std::string,std::string> assetToTargetAgent;
    std::vector<std::string> deadFighters;//撃墜or墜落済のFighterのfullNameのリスト。得点の二重計算防止用。
    //constructors & destructor
    R4RewardSample02(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~R4RewardSample02();
    //functions
    virtual void onCrash(const nl::json& args);
    virtual void onHit(const nl::json& args);
    virtual void onEpisodeBegin();
    virtual void onInnerStepEnd();
    virtual void onStepEnd();
};
void exportR4RewardSample02(py::module& m);
