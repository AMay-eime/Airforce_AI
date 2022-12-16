// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R4RewardSample02.h"
#include <algorithm>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Asset.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/R4BVRRuler01.h>
using namespace util;

R4RewardSample02::R4RewardSample02(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:AgentReward(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    pAvoid=getValueFromJsonKRD(modelConfig,"pAvoid",randomGen,0.0);
    pHitE_head=getValueFromJsonKRD(modelConfig,"pHitE_head",randomGen,0.0);
    pHitE_tail=getValueFromJsonKRD(modelConfig,"pHitE_tail",randomGen,0.0);
    pCrash=getValueFromJsonKRD(modelConfig,"pCrash",randomGen,0.0);
    pHit_head=getValueFromJsonKRD(modelConfig,"pHit_head",randomGen,0.0);
    pHit_tail=getValueFromJsonKRD(modelConfig,"pHit_tail",randomGen,0.0);
    pOut=getValueFromJsonKRD(modelConfig,"pOut",randomGen,0.0);
    pAlive=getValueFromJsonKRD(modelConfig,"pAlive",randomGen,0.0);
    pFuelShortage=getValueFromJsonKRD(modelConfig,"pFuelShortage",randomGen,0.0);
}
R4RewardSample02::~R4RewardSample02(){}
void R4RewardSample02::onEpisodeBegin(){
    j_target="All";
    this->AgentReward::onEpisodeBegin();
    auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
    auto o=ruler_->observables;
    westSider=o.at("westSider");
    eastSider=o.at("eastSider");
    dOut=o.at("dOut");
    dLine=o.at("dLine");
    forwardAx=o.at("forwardAx").get<std::map<std::string,Eigen::Vector2d>>();
    sideAx=o.at("sideAx").get<std::map<std::string,Eigen::Vector2d>>();
    fuelMargin=o.at("fuelMargin");
    distanceFromBase=o.at("distanceFromBase").get<std::map<std::string,double>>();
    manager->addEventHandler("Crash",[&](const nl::json& args){this->R4RewardSample02::onCrash(args);});//墜落数監視用
    manager->addEventHandler("Hit",[&](const nl::json& args){this->R4RewardSample02::onHit(args);});//撃墜数監視用
    parents.clear();
    missiles.clear();
    checked.clear();
    deadFighters.clear();
    assetToTargetAgent.clear();
    for(auto&& e:manager->getAssets()){
        if(isinstance<Missile>(e)){
            auto m=getShared<Missile>(e);
            missiles[m->getTeam()].push_back(m);
            checked[m->getFullName()]=false;
        }
    }
    for(auto& e:reward){
        std::string key=e.first;
        auto agent=getShared(manager->getAgent(key));
        for(auto& p:agent->parents){
            auto f=manager->getAsset<Fighter>(p.second->getFullName());
            parents[key].push_back(f);
            assetToTargetAgent[p.second->getFullName()]=key;
        }
    }
}
void R4RewardSample02::onCrash(const nl::json& args){
    std::lock_guard<std::mutex> lock(mtx);
    std::shared_ptr<PhysicalAsset> asset=args;
    std::string name=asset->getFullName();
    auto beg=deadFighters.begin();
    auto end=deadFighters.end();
    if(std::find(beg,end,name)==end){
        if(assetToTargetAgent.find(name)!=assetToTargetAgent.end()){
            reward[assetToTargetAgent[name]]+=pCrash;
        }
        deadFighters.push_back(name);
    }
}
void R4RewardSample02::onHit(const nl::json& args){//{"wpn":wpn,"tgt":tgt}
    std::lock_guard<std::mutex> lock(mtx);
    std::shared_ptr<PhysicalAsset> wpn=args.at("wpn");
    std::shared_ptr<PhysicalAsset> tgt=args.at("tgt");
    auto beg=deadFighters.begin();
    auto end=deadFighters.end();
    if(std::find(beg,end,tgt->getFullName())==end){
        Eigen::Vector3d vel=tgt->motion.vel.normalized();
        Eigen::Vector3d dir=(wpn->motion.pos-tgt->motion.pos).normalized();
        double aa=acos(vel.dot(dir));
        std::string name=tgt->getFullName();
        if(assetToTargetAgent.find(name)!=assetToTargetAgent.end()){
            reward[assetToTargetAgent[name]]+=((pHitE_head+pHitE_tail)+cos(aa)*(pHitE_head-pHitE_tail))/2;
        }
        name=wpn->parent.lock()->getFullName();
        if(assetToTargetAgent.find(name)!=assetToTargetAgent.end()){
            reward[assetToTargetAgent[name]]+=((pHit_head+pHit_tail)+cos(aa)*(pHit_head-pHit_tail))/2;
        }
        deadFighters.push_back(tgt->getFullName());
    }
}
void R4RewardSample02::onInnerStepEnd(){
    for(auto&& e:manager->getAssets()){
        if(isinstance<Missile>(e)){
            auto m=getShared<Missile>(e);
            if(!checked[m->getFullName()] && m->hasLaunched && !m->isAlive()){
                Track3D tgt=m->target;
                for(auto& t:reward){
                    std::string name=t.first;
                    for(auto& p:parents[name]){
                        if(tgt.isSame(p)){
                            reward[name]+=pAvoid;
                        }
                    }
                }
                checked[m->getFullName()]=true;
            }
        }else if(isinstance<Fighter>(e)){
            auto f=getShared<Fighter>(e);
            if(f->isAlive()){
                std::string team=f->getTeam();
                double outDist=std::max(0.0,abs(sideAx[team].dot(f->posI().block<2,1>(0,0,2,1)))-dOut);
                std::string name=f->getFullName();
                if(assetToTargetAgent.find(name)!=assetToTargetAgent.end()){
                    reward[assetToTargetAgent[name]]+=(outDist/1000.)*pOut*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep();

                    if(f->optCruiseFuelFlowRatePerDistance>0.0){
                        double maxReachableRange=f->fuelRemaining/f->optCruiseFuelFlowRatePerDistance;
                        double distanceFromLine=forwardAx[team].dot(f->posI().block<2,1>(0,0,2,1))+dLine;
	                    double excess=maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase[team]);
                        if(excess<0){
                            reward[assetToTargetAgent[name]]+=pFuelShortage*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep();
                        }
                    }
                }
            }
        }
    }
}
void R4RewardSample02::onStepEnd(){
    auto ruler=getShared<R4BVRRuler01>(manager->getRuler());
    if(ruler->endReason!=R4BVRRuler01::EndReason::NOTYET){
        for(auto&& e:manager->getAssets()){
            if(isinstance<Fighter>(e)){
                auto f=getShared<Fighter>(e);
                if(f->isAlive()){
                    std::string name=f->getFullName();
                    if(assetToTargetAgent.find(name)!=assetToTargetAgent.end()){
                        if(ruler->isReturnableToBase(f)){
                            //帰還可能なら生存点
                            reward[assetToTargetAgent[name]]+=pAlive;
                        }else{
                            //帰還不可能なら墜落ペナルティ
                            reward[assetToTargetAgent[name]]+=pCrash;
                        }
                    }
                }
            }
        }
    }
    this->AgentReward::onStepEnd();
}

void exportR4RewardSample02(py::module& m)
{
    using namespace pybind11::literals;
    EXPOSE_LOCAL_CLASS(R4RewardSample02)
    DEF_FUNC(R4RewardSample02,onCrash)
    .def("onCrash",[](R4RewardSample02& v,const py::object &args){
        return v.onCrash(args);
    })
    DEF_FUNC(R4RewardSample02,onHit)
    .def("onHit",[](R4RewardSample02& v,const py::object &args){
        return v.onHit(args);
    })
    DEF_FUNC(R4RewardSample02,onEpisodeBegin)
    DEF_FUNC(R4RewardSample02,onInnerStepEnd)
    DEF_FUNC(R4RewardSample02,onStepEnd)
    DEF_READWRITE(R4RewardSample02,pAvoid)
    DEF_READWRITE(R4RewardSample02,pHitE_head)
    DEF_READWRITE(R4RewardSample02,pHitE_tail)
    DEF_READWRITE(R4RewardSample02,pCrash)
    DEF_READWRITE(R4RewardSample02,pHit_head)
    DEF_READWRITE(R4RewardSample02,pHit_tail)
    DEF_READWRITE(R4RewardSample02,pOut)
    DEF_READWRITE(R4RewardSample02,pAlive)
    DEF_READWRITE(R4RewardSample02,westSider)
    DEF_READWRITE(R4RewardSample02,eastSider)
    DEF_READWRITE(R4RewardSample02,checked)
    DEF_READWRITE(R4RewardSample02,parents)
    DEF_READWRITE(R4RewardSample02,missiles)
    ;
}
