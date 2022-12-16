// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R4BVRRuler01.h"
#include <algorithm>
#include "Utility.h"
#include "SimulationManager.h"
#include "Asset.h"
#include "Fighter.h"
#include "Missile.h"
#include "Agent.h"
using namespace util;

R4BVRRuler01::R4BVRRuler01(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Ruler(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    debug=getValueFromJsonKRD(modelConfig,"debug",randomGen,false);
    minTime=getValueFromJsonKRD(modelConfig,"minTime",randomGen,300.0);
    dLine=getValueFromJsonKRD(modelConfig,"dLine",randomGen,100000.0);
    dOut=getValueFromJsonKRD(modelConfig,"dOut",randomGen,75000.0);
    hLim=getValueFromJsonKRD(modelConfig,"hLim",randomGen,20000.0);
    westSider=getValueFromJsonKRD<std::string>(modelConfig,"westSider",randomGen,"Red");
    eastSider=getValueFromJsonKRD<std::string>(modelConfig,"eastSider",randomGen,"Blue");
    pDisq=getValueFromJsonKRD(modelConfig,"pDisq",randomGen,-10.0);
    pBreak=getValueFromJsonKRD(modelConfig,"pBreak",randomGen,1.0);
    _setupPDownConfig(pHit,modelConfig,"pHit",1.0);
    _setupPDownConfig(pCrash,modelConfig,"pCrash",1.0);
    _setupPDownConfig(pAlive,modelConfig,"pAlive",1.0);
    pAdv=getValueFromJsonKRD(modelConfig,"pAdv",randomGen,0.01);
    pOut=getValueFromJsonKRD(modelConfig,"pOut",randomGen,0.01);
    pHitPerAircraft=getValueFromJsonKRD(modelConfig,"pHitPerAircraft",randomGen,true);
    pCrashPerAircraft=getValueFromJsonKRD(modelConfig,"pCrashPerAircraft",randomGen,true);
    pAlivePerAircraft=getValueFromJsonKRD(modelConfig,"pAlivePerAircraft",randomGen,true);
    enableAdditionalTime=getValueFromJsonKRD(modelConfig,"enableAdditionalTime",randomGen,true);
    terminalAtElimination=getValueFromJsonKRD(modelConfig,"terminalAtElimination",randomGen,true);
    terminalAtBreak=getValueFromJsonKRD(modelConfig,"terminalAtBreak",randomGen,true);
    considerFuelConsumption=getValueFromJsonKRD(modelConfig,"considerFuelConsumption",randomGen,true);
    fuelMargin=getValueFromJsonKRD(modelConfig,"fuelMargin",randomGen,0.1);
    distanceFromBase=getValueFromJsonKRD<std::map<std::string,double>>(
        modelConfig,"distanceFromBase",randomGen,{{"Default",100000.0}});
    if(distanceFromBase.size()==0){
        distanceFromBase["Default"]=100000.0;
    }
    if(distanceFromBase.find("Default")==distanceFromBase.end()){
        distanceFromBase["Default"]=distanceFromBase.begin()->second;
    }
    modelNamesToBeConsideredForBreak=getValueFromJsonKRD<std::map<std::string,std::vector<std::string>>>(
        modelConfig,"modelNamesToBeConsideredForBreak",randomGen,{{westSider,{"Any"}},{eastSider,{"Any"}}});
    modelNamesToBeExcludedForBreak=getValueFromJsonKRD<std::map<std::string,std::vector<std::string>>>(
        modelConfig,"modelNamesToBeExcludedForBreak",randomGen,{{westSider,{}},{eastSider,{}}});
    modelNamesToBeConsideredForElimination=getValueFromJsonKRD<std::map<std::string,std::vector<std::string>>>(
        modelConfig,"modelNamesToBeConsideredForElimination",randomGen,{{westSider,{"Any"}},{eastSider,{"Any"}}});
    modelNamesToBeExcludedForElimination=getValueFromJsonKRD<std::map<std::string,std::vector<std::string>>>(
        modelConfig,"modelNamesToBeExcludedForElimination",randomGen,{{westSider,{}},{eastSider,{}}});
    crashCount=std::map<std::string,std::map<std::string,int>>();
    hitCount=std::map<std::string,std::map<std::string,int>>();
    forwardAx=std::map<std::string,Eigen::Vector2d>();
    sideAx=std::map<std::string,Eigen::Vector2d>();
    pHitScale=std::map<std::string,std::map<std::string,double>>();
    pCrashScale=std::map<std::string,std::map<std::string,double>>();
    pAliveScale=std::map<std::string,std::map<std::string,double>>();
    endReason=EndReason::NOTYET;
    endReasonSub=EndReason::NOTYET;
}
R4BVRRuler01::~R4BVRRuler01(){}
void R4BVRRuler01::debugPrint(const std::string& reason,const std::string& team,double value){
    if(debug){
        std::cout<<"["<<getFactoryModelName()<<","<<manager->getTickCount()<<"] "<<reason<<", "<<team<<", "<<value<<std::endl;
    }
}
void R4BVRRuler01::onEpisodeBegin(){
    modelConfig["teams"]=nl::json::array({westSider,eastSider});
    this->Ruler::onEpisodeBegin();
    assert(score.size()==2);
    assert(score.count(westSider)==1 && score.count(eastSider)==1);
    manager->addEventHandler("Crash",[&](const nl::json& args){this->R4BVRRuler01::onCrash(args);});//墜落数監視用
    manager->addEventHandler("Hit",[&](const nl::json& args){this->R4BVRRuler01::onHit(args);});//撃墜数監視用
    crashCount.clear();
    hitCount.clear();
    leadRange.clear();
    lastDownPosition.clear();
    lastDownReason.clear();
    outDist.clear();
    breakTime.clear();
    disqTime.clear();
    forwardAx.clear();
    sideAx.clear();
    deadFighters.clear();
    _setupPDownScale(pHitScale,pHit,pHitPerAircraft);
    _setupPDownScale(pCrashScale,pCrash,pCrashPerAircraft);
    _setupPDownScale(pAliveScale,pAlive,pAlivePerAircraft);
    forwardAx[westSider]=Eigen::Vector2d(0.,1.);
    forwardAx[eastSider]=Eigen::Vector2d(0.,-1.);
    sideAx[westSider]=Eigen::Vector2d(-1.,0.);
    sideAx[eastSider]=Eigen::Vector2d(1.,0.);
    for(auto& team:teams){
        crashCount[team]=std::map<std::string,int>();
        hitCount[team]=std::map<std::string,int>();
        leadRange[team]=-dLine;
        outDist[team]=0;
        eliminatedTime[team]=-1;
        breakTime[team]=-1;
        disqTime[team]=-1;
    }
    endReason=EndReason::NOTYET;
    endReasonSub=EndReason::NOTYET;
    observables={
        {"maxTime",maxTime},
        {"minTime",minTime},
        {"eastSider",eastSider},
        {"westSider",westSider},
        {"dOut",dOut},
        {"dLine",dLine},
        {"hLim",hLim},
        {"distanceFromBase",distanceFromBase},
        {"fuelMargin",fuelMargin},
        {"forwardAx",forwardAx},
        {"sideAx",sideAx},
        {"endReason",enumToJson(endReason)}
    };
}
void R4BVRRuler01::onValidationEnd(){
    //残燃料の上書き(最適巡航の燃料消費はvalidateで計算されるためonEpisodeBeginでは上書きできない)
    for(auto&& team:teams){
        for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset);
        })){
            auto f=getShared<Fighter>(e);
            f->fuelRemaining-=f->optCruiseFuelFlowRatePerDistance*distanceFromBase[team]*(1+fuelMargin);
        }
    }

}
void R4BVRRuler01::onCrash(const nl::json& args){
    std::lock_guard<std::mutex> lock(mtx);
    std::shared_ptr<PhysicalAsset> asset=args;
    std::string team=asset->getTeam();
    std::string modelName=asset->getFactoryModelName();
    auto beg=deadFighters.begin();
    auto end=deadFighters.end();
    if(std::find(beg,end,asset->getFullName())==end){
        if(isToBeConsideredForElimination(team,modelName)){
            getCrashCount(team,modelName)+=1;
            lastDownPosition[team]=forwardAx[team].dot(asset->posI().block<2,1>(0,0,2,1));
            lastDownReason[team]=DownReason::CRASH;
        }
        deadFighters.push_back(asset->getFullName());
    }
}
void R4BVRRuler01::onHit(const nl::json& args){//{"wpn":wpn,"tgt":tgt}
    std::lock_guard<std::mutex> lock(mtx);
    std::shared_ptr<PhysicalAsset> wpn=args.at("wpn");
    std::shared_ptr<PhysicalAsset> tgt=args.at("tgt");
    std::string team=tgt->getTeam();
    std::string modelName=tgt->getFactoryModelName();
    auto beg=deadFighters.begin();
    auto end=deadFighters.end();
    if(std::find(beg,end,tgt->getFullName())==end){
        if(isToBeConsideredForElimination(team,modelName)){
            getHitCount(wpn->getTeam(),modelName)+=1;
            lastDownPosition[team]=forwardAx[team].dot(tgt->posI().block<2,1>(0,0,2,1));
            lastDownReason[team]=DownReason::HIT;
        }
        deadFighters.push_back(tgt->getFullName());
    }
}
void R4BVRRuler01::onInnerStepBegin(){
    std::vector<double> tmp;
    for(auto&& team:teams){
        crashCount[team]=std::map<std::string,int>();
        hitCount[team]=std::map<std::string,int>();
        if(manager->getTickCount()==0){
            tmp.clear();
            tmp.push_back(-dLine);
            for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
                return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && isToBeConsideredForBreak(team,asset->getFactoryModelName());
            })){
                auto f=getShared<Fighter>(e);
                tmp.push_back(forwardAx[team].dot(f->posI().block<2,1>(0,0,2,1)));
            }
            leadRange[team]=*std::max_element(tmp.begin(),tmp.end());
        }
    }
}
void R4BVRRuler01::onInnerStepEnd(){
    //生存数の監視
    std::map<std::string,int> aliveCount;
    for(auto& team:teams){
        aliveCount[team]=0;
        for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && isToBeConsideredForElimination(team,asset->getFactoryModelName());
        })){
            aliveCount[team]++;
        }
        if(aliveCount[team]==0 && eliminatedTime[team]<0){
            eliminatedTime[team]=manager->getTime();
        }
    }
    //防衛ラインの突破タイミングを監視
    std::vector<double> tmp;
    for(auto& team:teams){
        tmp.clear();
        tmp.push_back(-dLine);
        for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && isToBeConsideredForBreak(team,asset->getFactoryModelName());
        })){
            auto f=getShared<Fighter>(e);
            tmp.push_back(forwardAx[team].dot(f->posI().block<2,1>(0,0,2,1)));
        }
        leadRange[team]=*std::max_element(tmp.begin(),tmp.end());
        if(leadRange[team]>=dLine && breakTime[team]<0){
            breakTime[team]=manager->getTime();
            //得点計算 2：突破による加点(pBreak点)
            debugPrint("2. Break",team,pBreak);
            stepScore[team]+=pBreak;
        }
    }

    for(auto& team:teams){
        //得点計算 1：撃墜による加点(1機あたりpHit点)
        for(auto&& c:hitCount[team]){
            debugPrint("1. Hit("+c.first+")",team,c.second*getPHit(team,c.first));
    		stepScore[team]+=c.second*getPHit(team,c.first);
        }
	    //得点計算 6-(a)：墜落による減点(1機あたりpCrash点)
        for(auto&& c:crashCount[team]){
            debugPrint("6-(a). Crash("+c.first+")",team,-c.second*getPCrash(team,c.first));
		    stepScore[team]-=c.second*getPCrash(team,c.first);
        }
        //得点計算 6-(b)：場外に対する減点(1秒、1kmにつきpOut点)
        outDist[team]=0;
        for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset);
        })){
            auto f=getShared<Fighter>(e);
            outDist[team]+=std::max(0.0,abs(sideAx[team].dot(f->posI().block<2,1>(0,0,2,1)))-dOut);
        }
        if(outDist[team]>0.0){
            debugPrint("6-(b). Out",team,-(outDist[team]/1000.)*pOut*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep());
        }
        stepScore[team]-=(outDist[team]/1000.)*pOut*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep();
        if(score[team]+stepScore[team]<=pDisq && disqTime[team]<0){
            disqTime[team]=manager->getTime();
        }
    }
    if(endReasonSub==EndReason::NOTYET){
        if(terminalAtElimination &&
            (eliminatedTime[westSider]>=0 || eliminatedTime[eastSider]>=0)
        ){
            //終了条件(1)が有効な場合
            if(!(enableAdditionalTime && checkHitPossibility())){
                //相討ちを考慮する場合、以後の撃墜の可能性がある限り継続
                endReasonSub=EndReason::ELIMINATION;
            }
        }
        if(endReasonSub==EndReason::NOTYET){
            if(terminalAtBreak && 
                (breakTime[westSider]>=0 || breakTime[eastSider]>=0)
            ){
                //終了条件(2)が有効な場合
                if(!(enableAdditionalTime && checkHitPossibility())){
                    //相討ちを考慮する場合、以後の撃墜の可能性がある限り継続
                    endReasonSub=EndReason::BREAK;
                }
            }
            if(endReasonSub==EndReason::NOTYET){
                if(disqTime[westSider]>=0 || disqTime[eastSider]>=0){
                    //終了条件(5)の判定
                    endReasonSub=EndReason::PENALTY;
                }else{
                    //終了条件(3)の判定
                    bool withdrawal=manager->getTime()>=minTime;
                    for(auto& team:teams){
                        for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
                            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset);
                        })){
                            auto f=getShared<Fighter>(e);
                            if(forwardAx[team].dot(f->posI().block<2,1>(0,0,2,1))>=-dLine){
                                withdrawal=false;
                                break;
                            }
                        }
                        if(!withdrawal){
                            break;
                        }
                    }
                    if(withdrawal){
                        endReasonSub=EndReason::WITHDRAWAL;
                    }
                }
            }
        }
    }
}
void R4BVRRuler01::checkDone(){
	//終了判定
    dones.clear();
    for(auto&& e:manager->getAgents()){
        auto a=getShared(e);
        dones[a->getName()]=!a->isAlive();
    }
    bool considerAdvantage=false;
	//終了条件(1)：全機撃墜or墜落
    if(endReasonSub==EndReason::ELIMINATION){
        endReason=EndReason::ELIMINATION;
        if(breakTime[westSider]<0 && breakTime[eastSider]<0){
            //両者未突破の場合、得点計算3及び5の考慮が必要
            if(eliminatedTime[westSider]>=0 && eliminatedTime[eastSider]>=0){
                //両者全滅した場合、得点計算5に従い優勢度を考慮
                considerAdvantage=true;
                for(auto&& t:teams){
                    leadRange[t]=lastDownPosition[t];
                }
            }else{
                //一方が生存していた場合
                //得点計算 3：未突破の陣営に「そこから突破して更に帰還可能な」突破判定対象機が存在していれば+pBreak点
                if(eliminatedTime[eastSider]<0){//東が生存
                    bool chk=false;
                    for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
                        return asset->isAlive() && asset->getTeam()==eastSider && isinstance<Fighter>(asset) && isToBeConsideredForBreak(eastSider,asset->getFactoryModelName());
                    })){
                        if(isBreakableAndReturnableToBase(e.lock())){
                            chk=true;
                            break;
                        }
                    }
                    if(chk){
                        debugPrint("3. Break",eastSider,pBreak);
            		    stepScore[eastSider]+=pBreak;
	    	            score[eastSider]+=pBreak;
                    }
                }else{//西が生存
                    bool chk=false;
                    for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
                        return asset->isAlive() && asset->getTeam()==westSider && isinstance<Fighter>(asset) && isToBeConsideredForBreak(westSider,asset->getFactoryModelName());
                    })){
                        if(isBreakableAndReturnableToBase(e.lock())){
                            chk=true;
                            break;
                        }
                    }
                    if(chk){
                        debugPrint("3. Break",westSider,pBreak);
            		    stepScore[westSider]+=pBreak;
	    	            score[westSider]+=pBreak;
                    }
                }
            }
        }
    }
    //終了条件(2)：防衛ラインの突破
    else if(endReasonSub==EndReason::BREAK){
        endReason=EndReason::BREAK;
        //追加の得点増減はなし
    }
    //終了条件(5)：ペナルティによる敗北
    else if(endReasonSub==EndReason::PENALTY){
        endReason=EndReason::PENALTY;
        if(breakTime[westSider]<0 && breakTime[eastSider]<0){
            //両者未突破の場合、得点計算5に従い優勢度を計算
            considerAdvantage=true;
        }
    }
    //終了条件(3)：両者撤退による打ち切り
    else if(endReasonSub==EndReason::WITHDRAWAL){
        endReason=EndReason::WITHDRAWAL;
        //追加の得点増減はなし(優勢度も計算しない)
    }
    //終了条件(4)：時間切れ
    else if(manager->getTime()>=maxTime){
        endReason=EndReason::TIMEUP;
        if(breakTime[westSider]<0 && breakTime[eastSider]<0){
            //両者未突破の場合、得点計算5に従い優勢度を計算
            considerAdvantage=true;
        }
    }
    //得点計算 4：生存点(1機あたりpAlive点)
    //得点計算 6(a)：帰還不可による墜落ペナルティ(1機あたりpCrash点)
    if(endReason!=EndReason::NOTYET){
        for(auto& team:teams){
            for(auto&& e:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
                return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && isToBeConsideredForElimination(team,asset->getFactoryModelName());
            })){
                auto asset=e.lock();
                if(isReturnableToBase(asset)){
                    //帰還可能なら生存点
                    debugPrint("4. Alive("+asset->getFactoryModelName()+")",team,getPAlive(team,asset->getFactoryModelName()));
		            stepScore[team]+=getPAlive(team,asset->getFactoryModelName());
		            score[team]+=getPAlive(team,asset->getFactoryModelName());
                }else{
                    //帰還不可能なら墜落ペナルティ
                    debugPrint("6(a). NoReturn("+asset->getFactoryModelName()+")",team,-getPCrash(team,asset->getFactoryModelName()));
		            stepScore[team]-=getPCrash(team,asset->getFactoryModelName());
		            score[team]-=getPCrash(team,asset->getFactoryModelName());
                }
            }
        }
    }
    if(considerAdvantage){
        //終了条件(4)、(5)または両者全滅の場合で、両者未突破だった場合の優勢度加点
        //得点計算 5：進出度合いに対する加点(1kmにつきpAdv点)
        if(leadRange[westSider]>leadRange[eastSider]){
            double s=(leadRange[westSider]-leadRange[eastSider])/2./1000.*pAdv;
            debugPrint("5. Adv",westSider,s);
		    stepScore[westSider]+=s;
		    score[westSider]+=s;
        }else{
            double s=(leadRange[eastSider]-leadRange[westSider])/2./1000.*pAdv;
            debugPrint("5. Adv",eastSider,s);
		    stepScore[eastSider]+=s;
		    score[eastSider]+=s;
        }
    }
    //打ち切り対策で常時勝者を仮設定しておく
    if(score[westSider]>score[eastSider]){
        winner=westSider;
    }else if(score[westSider]<score[eastSider]){
        winner=eastSider;
    }else{
        winner="";//引き分けは空文字列
    }
    if(endReason!=EndReason::NOTYET){
        for(auto& e:dones){
            e.second=true;
        }
        dones["__all__"]=true;
    }else{
        dones["__all__"]=false;
    }
    observables["endReason"]=enumToJson(endReason);
}
void R4BVRRuler01::_setupPDownConfig(
    std::map<std::string,double>& _config,
    const nl::json& _modelConfig,
    const std::string& _key,
    double _defaultValue){
    _config=getValueFromJsonKRD<std::map<std::string,double>>(
        _modelConfig,_key,randomGen,{{"Default",_defaultValue}});
    if(_config.size()==0){
        _config["Default"]=_defaultValue;
    }
    if(_config.find("Default")==_config.end()){
        _config["Default"]=_config.begin()->second;
    }
}
void R4BVRRuler01::_setupPDownScale(
    std::map<std::string,std::map<std::string,double>>& _scale,
    const std::map<std::string,double>& _config,
    bool _perAircraft){
    _scale.clear();
    for(auto& team:teams){
        auto& sub=_scale[team];
        for(auto&& asset:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && isToBeConsideredForElimination(team,asset->getFactoryModelName());
        })){
            std::string mn=asset.lock()->getFactoryModelName();
            std::string cn;
            if(_config.find(mn)==_config.end()){
                cn="Default";
            }else{
                cn=mn;
            }
            auto found=sub.find(cn);
            if(found==sub.end()){
                sub.insert(std::make_pair(cn,1));
            }else{
                found->second+=1;
            }
        }
        for(auto& p:sub){
            if(_perAircraft){
                p.second=1.0;
            }else{
                p.second=1.0/p.second;
            }
        }
    }
}
double R4BVRRuler01::_getPDownImpl(
        const std::map<std::string,std::map<std::string,double>>& _scale,
        const std::map<std::string,double>& _config,
        const std::string& _team,
        const std::string& _modelName) const{
    std::string cn;
    double p;
    auto found=_config.find(_modelName);
    if(found==_config.end()){
        cn="Default";
        p=_config.at(cn);
    }else{
        cn=_modelName;
        p=found->second;
    }
    return p*_scale.at(_team).at(cn);
}
double R4BVRRuler01::getPHit(const std::string& team,const std::string& modelName) const{
    return _getPDownImpl(pHitScale,pHit,team,modelName);
}
double R4BVRRuler01::getPCrash(const std::string& team,const std::string& modelName) const{
    return _getPDownImpl(pCrashScale,pCrash,team,modelName);
}
double R4BVRRuler01::getPAlive(const std::string& team,const std::string& modelName) const{
    return _getPDownImpl(pAliveScale,pAlive,team,modelName);
}
int& R4BVRRuler01::getCrashCount(const std::string& team,const std::string& modelName){
    auto& count =crashCount[team];
    auto found=count.find(modelName);
    if(found==count.end()){
        count.insert(std::make_pair(modelName,0));
    }
    return count[modelName];
}
int& R4BVRRuler01::getHitCount(const std::string& team,const std::string& modelName){
    auto& count =hitCount[team];
    auto found=count.find(modelName);
    if(found==count.end()){
        count.insert(std::make_pair(modelName,0));
    }
    return count[modelName];
}
bool R4BVRRuler01::isToBeConsideredForBreak(const std::string& team,const std::string& modelName){
    auto c_begin=modelNamesToBeConsideredForBreak[team].begin();
    auto c_end=modelNamesToBeConsideredForBreak[team].end();
    auto e_begin=modelNamesToBeExcludedForBreak[team].begin();
    auto e_end=modelNamesToBeExcludedForBreak[team].end();
    bool isAny=std::find(c_begin,c_end,"Any")!=c_end;
    return  (isAny || std::find(c_begin,c_end,modelName)!=c_end) && std::find(e_begin,e_end,modelName)==e_end;
}
bool R4BVRRuler01::isToBeConsideredForElimination(const std::string& team,const std::string& modelName){
    auto c_begin=modelNamesToBeConsideredForElimination[team].begin();
    auto c_end=modelNamesToBeConsideredForElimination[team].end();
    auto e_begin=modelNamesToBeExcludedForElimination[team].begin();
    auto e_end=modelNamesToBeExcludedForElimination[team].end();
    bool isAny=std::find(c_begin,c_end,"Any")!=c_end;
    return  (isAny || std::find(c_begin,c_end,modelName)!=c_end) && std::find(e_begin,e_end,modelName)==e_end;
}
bool R4BVRRuler01::checkHitPossibility() const{
    //この先撃墜の可能性があるかどうかを返す
    if(eliminatedTime.at(westSider)<0){
        //西側が生存している場合、東側の航空機か誘導弾が残っていれば可能性ありとする
        for(auto&& asset:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==eastSider && isinstance<Fighter>(asset);
        })){
            return true;
        }
        for(auto&& asset:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==eastSider && isinstance<Missile>(asset);
        })){
            auto msl=getShared<Missile>(asset);
            if(msl->isAlive() && msl->hasLaunched){
                return true;
            }
        }
    }
    if(eliminatedTime.at(eastSider)<0){
        //東側が生存している場合、西側の航空機か誘導弾が残っていれば可能性ありとする
        for(auto&& asset:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==westSider && isinstance<Fighter>(asset);
        })){
            return true;
        }
        for(auto&& asset:manager->getAssets([&](std::shared_ptr<const Asset> asset)->bool{
            return asset->isAlive() && asset->getTeam()==westSider && isinstance<Missile>(asset);
        })){
            auto msl=getShared<Missile>(asset);
            if(msl->isAlive() && msl->hasLaunched){
                return true;
            }
        }
    }
    return false;
}
bool R4BVRRuler01::isReturnableToBase(const std::shared_ptr<PhysicalAsset>& asset) const{
    //現在位置から帰還可能かどうか
    if(!considerFuelConsumption){
        return true;
    }
    auto fgtr=getShared<Fighter>(asset);
    std::string team=fgtr->getTeam();
    double range=fgtr->getMaxReachableRange()/(1+fuelMargin);//今の残燃料で到達できる距離
    double distanceFromLine=forwardAx.at(team).dot(fgtr->posI().block<2,1>(0,0,2,1))+dLine;
    return distanceFromBase.at(team)+distanceFromLine<=range;
}
bool R4BVRRuler01::isBreakableAndReturnableToBase(const std::shared_ptr<PhysicalAsset>& asset) const{
    //現在位置から防衛ラインを突破して更に帰還可能かどうか
    if(!considerFuelConsumption){
        return true;
    }
    auto fgtr=getShared<Fighter>(asset);
    std::string team=fgtr->getTeam();
    double range=fgtr->getMaxReachableRange()/(1+fuelMargin);//今の残燃料で到達できる距離
    double distanceFromLine=forwardAx.at(team).dot(fgtr->posI().block<2,1>(0,0,2,1))+dLine;
    return distanceFromBase.at(team)+4*dLine-distanceFromLine<=range;
}

void exportR4BVRRuler01(py::module& m)
{
    using namespace pybind11::literals;
    BIND_MAP_NAME(std::string,R4BVRRuler01::DownReason,"std::map<std::string,R4BVRRuler01::DownReason>",false);

    auto cls=EXPOSE_CLASS(R4BVRRuler01);
    cls
    DEF_FUNC(R4BVRRuler01,onCrash)
    .def("onCrash",[](R4BVRRuler01& v,const py::object &args){
        return v.onCrash(args);
    })
    DEF_FUNC(R4BVRRuler01,onHit)
    .def("onHit",[](R4BVRRuler01& v,const py::object &args){
        return v.onHit(args);
    })
    DEF_FUNC(R4BVRRuler01,onEpisodeBegin)
    DEF_FUNC(R4BVRRuler01,onValidationEnd)
    DEF_FUNC(R4BVRRuler01,onInnerStepBegin)
    DEF_FUNC(R4BVRRuler01,onInnerStepEnd)
    DEF_FUNC(R4BVRRuler01,checkDone)
    DEF_FUNC(R4BVRRuler01,_setupPDownConfig)
    DEF_FUNC(R4BVRRuler01,_setupPDownScale)
    DEF_FUNC(R4BVRRuler01,_getPDownImpl)
    DEF_FUNC(R4BVRRuler01,isToBeConsideredForBreak)
    DEF_FUNC(R4BVRRuler01,isToBeConsideredForElimination)
    DEF_FUNC(R4BVRRuler01,checkHitPossibility)
    DEF_FUNC(R4BVRRuler01,isReturnableToBase)
    DEF_FUNC(R4BVRRuler01,isBreakableAndReturnableToBase)
    DEF_READWRITE(R4BVRRuler01,dLine)
    DEF_READWRITE(R4BVRRuler01,dOut)
    DEF_READWRITE(R4BVRRuler01,hLim)
    DEF_READWRITE(R4BVRRuler01,minTime)
    DEF_READWRITE(R4BVRRuler01,westSider)
    DEF_READWRITE(R4BVRRuler01,eastSider)
    DEF_READWRITE(R4BVRRuler01,pDisq)
    DEF_READWRITE(R4BVRRuler01,pBreak)
    DEF_READWRITE(R4BVRRuler01,modelNamesToBeConsideredForBreak)
    DEF_READWRITE(R4BVRRuler01,modelNamesToBeExcludedForBreak)
    DEF_READWRITE(R4BVRRuler01,pHit)
    DEF_READWRITE(R4BVRRuler01,pHitScale)
    DEF_READWRITE(R4BVRRuler01,pCrash)
    DEF_READWRITE(R4BVRRuler01,pCrashScale)
    DEF_READWRITE(R4BVRRuler01,pAlive)
    DEF_READWRITE(R4BVRRuler01,pAliveScale)
    DEF_READWRITE(R4BVRRuler01,modelNamesToBeConsideredForElimination)
    DEF_READWRITE(R4BVRRuler01,modelNamesToBeExcludedForElimination)
    DEF_READWRITE(R4BVRRuler01,pAdv)
    DEF_READWRITE(R4BVRRuler01,pOut)
    DEF_READWRITE(R4BVRRuler01,crashCount)
    DEF_READWRITE(R4BVRRuler01,hitCount)
    DEF_READWRITE(R4BVRRuler01,leadRange)
    DEF_READWRITE(R4BVRRuler01,lastDownPosition)
    DEF_READWRITE(R4BVRRuler01,lastDownReason)
    DEF_READWRITE(R4BVRRuler01,outDist)
    DEF_READWRITE(R4BVRRuler01,breakTime)
    DEF_READWRITE(R4BVRRuler01,disqTime)
    DEF_READWRITE(R4BVRRuler01,forwardAx)
    DEF_READWRITE(R4BVRRuler01,sideAx)
    DEF_READWRITE(R4BVRRuler01,endReason)
    DEF_READWRITE(R4BVRRuler01,endReasonSub)
    DEF_READWRITE(R4BVRRuler01,pHitPerAircraft)
    DEF_READWRITE(R4BVRRuler01,pCrashPerAircraft)
    DEF_READWRITE(R4BVRRuler01,pAlivePerAircraft)
    DEF_READWRITE(R4BVRRuler01,enableAdditionalTime)
    DEF_READWRITE(R4BVRRuler01,terminalAtElimination)
    DEF_READWRITE(R4BVRRuler01,terminalAtBreak)
    DEF_READWRITE(R4BVRRuler01,considerFuelConsumption)
    DEF_READWRITE(R4BVRRuler01,fuelMargin)
    DEF_READWRITE(R4BVRRuler01,distanceFromBase)
    ;
    py::enum_<R4BVRRuler01::DownReason>(cls,"DownReason")
    .value("CRASH",R4BVRRuler01::DownReason::CRASH)
    .value("HIT",R4BVRRuler01::DownReason::HIT)
    ;
    py::enum_<R4BVRRuler01::EndReason>(cls,"EndReason")
    .value("NOTYET",R4BVRRuler01::EndReason::NOTYET)
    .value("ELIMINATION",R4BVRRuler01::EndReason::ELIMINATION)
    .value("BREAK",R4BVRRuler01::EndReason::BREAK)
    .value("TIMEUP",R4BVRRuler01::EndReason::TIMEUP)
    .value("WITHDRAWAL",R4BVRRuler01::EndReason::WITHDRAWAL)
    .value("PENALTY",R4BVRRuler01::EndReason::PENALTY)
    ;
}
