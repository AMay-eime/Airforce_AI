// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Fighter.h"
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"
#include "Agent.h"
#include "Missile.h"
#include "Sensor.h"
#include <boost/uuid/nil_generator.hpp>
using namespace util;
Fighter::Fighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:PhysicalAsset(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    //modelConfigで指定するもの
    m=getValueFromJsonKRD(modelConfig.at("dynamics"),"m",randomGen,10000.0);
    if(modelConfig.contains("stealth")){
        rcsScale=getValueFromJsonKRD(modelConfig.at("stealth"),"rcsScale",randomGen,1.0);
    }else{
        rcsScale=1.0;
    }
    if(modelConfig.contains("weapon")){
        numMsls=getValueFromJsonKRD(modelConfig.at("weapon"),"numMsls",randomGen,10);
    }else{
        numMsls=10;
    }
    fuelCapacity=getValueFromJsonKRD(modelConfig.at("propulsion"),"fuelCapacity",randomGen,5000.0);
    fuelRemaining=fuelCapacity;
    //instanceConfigで指定するもの
    motion.pos=getValueFromJsonKRD(instanceConfig,"pos",randomGen,Eigen::Vector3d(0,0,0));
    double V=getValueFromJsonKRD(instanceConfig,"vel",randomGen,0.0);
    double azInDeg=getValueFromJsonKRD(instanceConfig,"heading",randomGen,0.0);
    while(azInDeg>180){azInDeg-=360;}
    while(azInDeg<=-180){azInDeg+=360;}
    motion.az=deg2rad(azInDeg);
    datalinkName=getValueFromJsonKRD<std::string>(instanceConfig,"datalinkName",randomGen,"");
    //その他の位置、姿勢等の運動状態に関する変数の初期化
    motion.el=0.0;
    double cs=cos(motion.az);
    double sn=sin(motion.az);
    double cs_2=cos(motion.az/2);
    double sn_2=sin(motion.az/2);
    if(azInDeg==0){
        cs=1;sn=0;
        cs_2=1;sn_2=0;
    }else if(azInDeg==90){
        cs=0;sn=1;
        cs_2=sqrt(2)/2;
        sn_2=sqrt(2)/2;
    }else if(azInDeg==180){
        cs=-1;sn=0;
        cs_2=0;sn_2=1;
    }else if(azInDeg==-90){
        cs=0;sn=-1;
        cs_2=sqrt(2)/2;
        sn_2=-sqrt(2)/2;
    }
    vel_prev=motion.vel=Eigen::Vector3d(V*cs,V*sn,0);
    pos_prev=motion.pos-motion.vel*(interval[SimPhase::BEHAVE]*manager->getBaseTimeStep());
    motion.q=Quaternion(cs_2,0,0,sn_2);
    motion.calcQh();
    motion.omega<<0,0,0;
    //その他の内部変数
    isDatalinkEnabled=true;
    track.clear();
    trackSource.clear();
    target=Track3D();targetID=-1;
    launchable=true;
    observables["motion"]=motion;
    observables["propulsion"]={
        {"fuelRemaining",fuelRemaining}
    };
    observables["spec"]={
        {"dynamics",{nl::json::object()}},
        {"weapon",{
            {"numMsls",numMsls}
        }},
        {"stealth",{
            {"rcsScale",rcsScale}
        }},
        {"sensor",nl::json::object()},
        {"propulsion",{
            {"fuelCapacity",fuelCapacity}
        }}
    };
    observables["shared"]={
        {"agent",nl::json::object()},
        {"fighter",nl::json::object()}
    };
}
Fighter::~Fighter(){
}
void Fighter::makeChildren(){
    //子要素のデフォルトのmodelConfigは、処理周期の指定を自身と同じとする
    nl::json defaultControllerModelConfig=nl::json::object();
    if(modelConfig.contains("firstTick")){
        defaultControllerModelConfig["firstTick"]=modelConfig.at("firstTick");
    }
    if(modelConfig.contains("interval")){
        defaultControllerModelConfig["interval"]=modelConfig.at("interval");
    }
    //子要素のinstanceConfig
    //sensors
    nl::json sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":Radar"},//PhysicalAsset
        {"parent",this->weak_from_this()},//PhysicalAsset
        {"isBound",true}//PhysicalAsset
    };
    radar=manager->generateAsset<AircraftRadar>(
        "PhysicalAsset",
        modelConfig.at("/sensor/radar"_json_pointer),
        sub
    );
    observables["spec"]["sensor"]["radar"]=radar.lock()->observables["spec"];
    sub["seed"]=randomGen();
    sub["fullName"]=fullName+":MWS";
    mws=manager->generateAsset<MWS>(
        "PhysicalAsset",
        modelConfig.at("/sensor/mws"_json_pointer),
        sub
    );
    observables["spec"]["sensor"]["mws"]=mws.lock()->observables["spec"];

    //propulsion
    sub["fullName"]=fullName+":Propulsion";
    engine=manager->generateAsset<Propulsion>(
        "PhysicalAsset",
        modelConfig.at("/propulsion/engine"_json_pointer),
        sub
    );

    //weapons
    missiles.clear();
    missileTargets.clear();
    nextMsl=0;
    remMsls=numMsls;
    sub["isBound"]=false;
    for(int i=0;i<numMsls;++i){
        sub["seed"]=randomGen();
        sub["fullName"]=fullName+":Missile"+std::to_string(i+1);
        missiles.push_back(
            manager->generateAsset<Missile>(
                "PhysicalAsset",
                modelConfig.at("/weapon/missile"_json_pointer),
                sub
            )
        );
        missileTargets.push_back(std::make_pair(Track3D(),false));
        manager->generateCommunicationBuffer(
            "MissileComm:"+sub["fullName"].get<std::string>(),
            nl::json::array({"PhysicalAsset:"+getFullName(),"PhysicalAsset:"+sub["fullName"].get<std::string>()}),
            nl::json::array({"PhysicalAsset:"+getTeam()+"/.+"})
        );
    }
    if(numMsls<=0){
        //初期弾数0のとき、射程計算のみ可能とするためにダミーインスタンスを生成
        sub["seed"]=randomGen();
        sub["fullName"]=fullName+":Missile(dummy)";
        dummyMissile=manager->generateAsset<Missile>(
                "PhysicalAsset",
                modelConfig.at("/weapon/missile"_json_pointer),
                sub
            );
        //CommunicationBufferは正規表現で対象Assetを特定するため、括弧はエスケープが必要
        std::string query=fullName+":Missile\\(dummy\\)";
        manager->generateCommunicationBuffer(
            "MissileComm:"+sub["fullName"].get<std::string>(),
            nl::json::array({"PhysicalAsset:"+getFullName(),"PhysicalAsset:"+query}),
            nl::json::array()
        );
    }else{
        dummyMissile=std::shared_ptr<Missile>(nullptr);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":SensorDataSharer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/sensorDataSharer"_json_pointer)){
        controllers["SensorDataSharer"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/controller/sensorDataSharer"_json_pointer),sub);
    }else{
        controllers["SensorDataSharer"]=manager->generateAssetByClassName<SensorDataSharer>("Controller","Fighter::SensorDataSharer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":SensorDataSanitizer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/sensorDataSanitizer"_json_pointer)){
        controllers["SensorDataSanitizer"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/controller/sensorDataSanitizer"_json_pointer),sub);
    }else{
        controllers["SensorDataSanitizer"]=manager->generateAssetByClassName<SensorDataSanitizer>("Controller","Fighter::SensorDataSanitizer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":OtherDataSharer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/otherDataSharer"_json_pointer)){
        controllers["OtherDataSharer"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/controller/otherDataSharer"_json_pointer),sub);
    }else{
        controllers["OtherDataSharer"]=manager->generateAssetByClassName<OtherDataSharer>("Controller","Fighter::OtherDataSharer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":OtherDataSanitizer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/otherDataSanitizer"_json_pointer)){
        controllers["OtherDataSanitizer"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/controller/otherDataSanitizer"_json_pointer),sub);
    }else{
        controllers["OtherDataSanitizer"]=manager->generateAssetByClassName<OtherDataSanitizer>("Controller","Fighter::OtherDataSanitizer",defaultControllerModelConfig,sub);
    }
    if(modelConfig.contains("/pilot/model"_json_pointer)){
        sub={
            {"seed",randomGen()},//Entity
            {"fullName",fullName+":HumanIntervention"},//Controller
            {"parent",this->weak_from_this()}//Controller
        };
        controllers["HumanIntervention"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/pilot/model"_json_pointer),sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":WeaponController"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/weapon/controller"_json_pointer)){
        controllers["WeaponController"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/weapon/controller"_json_pointer),sub);
    }else{
        controllers["WeaponController"]=manager->generateAssetByClassName<WeaponController>("Controller","Fighter::WeaponController",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"fullName",fullName+":FlightController"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    controllers["FlightController"]=manager->generateAsset<Controller>("Controller",modelConfig.at("/dynamics/controller"_json_pointer),sub);
}
void Fighter::validate(){
    isDatalinkEnabled = communicationBuffers.count(datalinkName)>0;
    //自機以外の味方機の誘導弾とのデータリンクに接続
    for(auto&& asset:manager->getAssets()){
        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset) && !this->isSame(asset.lock())){
            auto f=getShared<const Fighter>(asset);
            for(auto&& e:f->missiles){
                std::string bufferName="MissileComm:"+e.lock()->getFullName();
                manager->requestInvitationToCommunicationBuffer(bufferName,getShared<Asset>(this->shared_from_this()));
            }
        }
    }
    if(modelConfig.contains("/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer)){
        optCruiseFuelFlowRatePerDistance=modelConfig.at("/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer);
    }else{
        optCruiseFuelFlowRatePerDistance=calcOptimumCruiseFuelFlowRatePerDistance();
        std::cout<<"optCruiseFuelFlowRatePerDistance="<<optCruiseFuelFlowRatePerDistance<<" for \""<<getFactoryModelName()<<"\""<<std::endl;
        std::cout<<"maximumRange="<<getMaxReachableRange()<<std::endl;
    }
    observables["spec"]["propulsion"]["optCruiseFuelFlowRatePerDistance"]=optCruiseFuelFlowRatePerDistance;
}
void Fighter::setDependency(){
    //validate
    for(auto&& e:missiles){
        e.lock()->dependencyChecker->addDependency(SimPhase::VALIDATE,getShared<Asset>(this->shared_from_this()));
    }
    if(!dummyMissile.expired()){
        dummyMissile.lock()->dependencyChecker->addDependency(SimPhase::VALIDATE,getShared<Asset>(this->shared_from_this()));
    }
    //perceive
    controllers["SensorDataSharer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,radar.lock());
    controllers["SensorDataSharer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,mws.lock());
    for(auto&& asset:manager->getAssets()){
        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
            auto f=getShared<const Fighter>(asset);
            controllers["SensorDataSanitizer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,f->controllers.at("SensorDataSharer").lock());
        }
    }
    for(auto&& e:missiles){
        e.lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,controllers["SensorDataSanitizer"].lock());
        dependencyChecker->addDependency(SimPhase::PERCEIVE,e.lock());
    }
    if(!dummyMissile.expired()){
        dummyMissile.lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,controllers["SensorDataSanitizer"].lock());
        dependencyChecker->addDependency(SimPhase::PERCEIVE,dummyMissile.lock());
    }
    controllers["OtherDataSharer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,getShared<Asset>(this->shared_from_this()));
    controllers["OtherDataSharer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,agent.lock());
    for(auto&& asset:manager->getAssets()){
        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
            auto f=getShared<const Fighter>(asset);
            controllers["OtherDataSanitizer"].lock()->dependencyChecker->addDependency(SimPhase::PERCEIVE,f->controllers.at("OtherDataSharer").lock());
        }
    }
    //control
    controllers["OtherDataSharer"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,agent.lock());
    for(auto&& asset:manager->getAssets()){
        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
            auto f=getShared<const Fighter>(asset);
            controllers["OtherDataSanitizer"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,f->controllers.at("OtherDataSharer").lock());
        }
    }
    if(controllers.count("HumanIntervention")>0){
        controllers["HumanIntervention"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock());
        controllers["WeaponController"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["HumanIntervention"].lock());
        controllers["FlightController"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["HumanIntervention"].lock());
    }else{
        controllers["WeaponController"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock());
        controllers["FlightController"].lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock());
    }
    engine.lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["FlightController"].lock());
    for(auto&& e:missiles){
        e.lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["WeaponController"].lock());
    }
    if(!dummyMissile.expired()){
        dummyMissile.lock()->dependencyChecker->addDependency(SimPhase::CONTROL,controllers["WeaponController"].lock());
    }
    //behave
    engine.lock()->dependencyChecker->addDependency(SimPhase::BEHAVE,getShared<Asset>(this->shared_from_this()));
}
void Fighter::perceive(bool inReset){
    PhysicalAsset::perceive(inReset);
    observables["propulsion"]={
        {"fuelRemaining",fuelRemaining}
    };
    nl::json mslObs=nl::json::array();
    for(auto&& e:missiles){
        mslObs.push_back(e.lock()->observables);
    }
    observables["weapon"]={
        {"remMsls",remMsls},
        {"nextMsl",nextMsl},
        {"launchable",launchable},
        {"missiles",std::move(mslObs)}
    };
}
void Fighter::control(){
    //done by controllers
}
void Fighter::behave(){
    double dt=interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
    calcMotion(dt);
    fuelRemaining=std::max<double>(fuelRemaining-dt*engine.lock()->getFuelFlowRate(),0.0);
    if(motion.pos(2)>0){//墜落
        manager->triggerEvent("Crash",this->weak_from_this());
        manager->requestToKillAsset(getShared<Asset>(this->shared_from_this()));
    }
}
void Fighter::kill(){
    engine.lock()->kill();
    radar.lock()->kill();
    mws.lock()->kill();
    if(isDatalinkEnabled){
        //immidiate notification to friends. If you need more realistic behavior, other notification or estimation of aliveness scheme should be implemented.
        nl::json mslObs=nl::json::array();
        for(auto&& e:missiles){
            mslObs.push_back(e.lock()->observables);
        }
        communicationBuffers[datalinkName].lock()->send({
            {"fighterObservables",{{getFullName(),{
                {"isAlive",false},
                {"weapon",{{
                    {"remMsls",remMsls},
                    {"nextMsl",nextMsl},
                    {"launchable",false},
                    {"missiles",std::move(mslObs)}
                }}},
                {"time",(manager->getTickCount()+1)*manager->getBaseTimeStep()}
            }}}}
        },CommunicationBuffer::MERGE);
    }
    for(auto&& e:controllers){
        e.second.lock()->kill();
    }
    this->PhysicalAsset::kill();
}
double Fighter::getMaxReachableRange(){
    if(optCruiseFuelFlowRatePerDistance<=0.0){
        //no fuel consumption
        return std::numeric_limits<double>::infinity();
    }else{
        return fuelRemaining/optCruiseFuelFlowRatePerDistance;
    }
}
std::pair<bool,Track3D> Fighter::isTracking(std::weak_ptr<PhysicalAsset> target_){
    if(target_.expired()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.lock()->uuid);
    }
}
std::pair<bool,Track3D> Fighter::isTracking(const Track3D& target_){
    if(target_.is_none()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.truth);
    }
}
std::pair<bool,Track3D> Fighter::isTracking(const boost::uuids::uuid& target_){
    if(target_==boost::uuids::nil_uuid()){
        return std::make_pair(false,Track3D());
    }else{
        if(isAlive()){
            for(auto& t:track){
                if(t.isSame(target_)){
                    return std::make_pair(true,t);
                }
            }
        }
        return std::make_pair(false,Track3D());
    }
}
void Fighter::setFlightControllerMode(const std::string& ctrlName){
    std::dynamic_pointer_cast<FlightController>(controllers["FlightController"].lock())->setMode(ctrlName);
}
Eigen::Vector3d Fighter::toEulerAngle(){
    Eigen::Vector3d ex=relItoB(Eigen::Vector3d(1,0,0));
    Eigen::Vector3d ey=relItoB(Eigen::Vector3d(0,1,0));
    Eigen::Vector3d horizontalY=Eigen::Vector3d(0,0,1).cross(ex).normalized();
    double sinRoll=horizontalY.cross(ey).dot(ex);
    double cosRoll=horizontalY.dot(ey);
    double rollAtt=atan2(sinRoll,cosRoll);
    return Eigen::Vector3d(rollAtt,-motion.el,motion.az);
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt);
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt);
    }
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt,aa);
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt,aa);
    }
}
Eigen::Vector3d Fighter::relHtoI(const Eigen::Vector3d &v) const{
    return motion.relHtoP(v);
}
Eigen::Vector3d Fighter::relItoH(const Eigen::Vector3d &v) const{
    return motion.relPtoH(v);
}
Eigen::Vector3d Fighter::absHtoI(const Eigen::Vector3d &v) const{
    return motion.absHtoP(v);
}
Eigen::Vector3d Fighter::absItoH(const Eigen::Vector3d &v) const{
    return motion.absPtoH(v);
}
std::shared_ptr<AssetAccessor> Fighter::getAccessor(){
    if(!accessor){
        accessor=std::make_shared<FighterAccessor>(getShared<Fighter>(this->shared_from_this()));
    }
    return accessor;
}

Fighter::SensorDataSharer::SensorDataSharer(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
void Fighter::SensorDataSharer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(p->isDatalinkEnabled){
            p->communicationBuffers[p->datalinkName].lock()->send({
                {"fighterObservables",{{p->getFullName(),{
                    {"isAlive",p->isAlive()},
                    {"sensor",{
                        {"radar",p->radar.lock()->observables},
                        {"mws",p->mws.lock()->observables}
                    }},
                    {"time",manager->getTime()}
                }}}}
            },CommunicationBuffer::MERGE);
        }
    }
}
Fighter::SensorDataSanitizer::SensorDataSanitizer(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
void Fighter::SensorDataSanitizer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        p->track.clear();
        p->trackSource.clear();
        for(auto& e:p->radar.lock()->track){
            p->track.push_back(e.copy());
            p->trackSource.push_back({p->getFullName()});
        }
        nl::json sharedFighterObservable=nl::json::object();
        nl::json& sf=p->observables["shared"]["fighter"];
        sf=nl::json::object();
        if(p->isDatalinkEnabled){
            std::pair<double,nl::json> tmp=p->communicationBuffers[p->datalinkName].lock()->receive("fighterObservables");
            if(tmp.first>=0){//valid data
                sharedFighterObservable=tmp.second;
            }
            Track3D same;
            int sameID=-1;
            int idx=0;
            for(auto&& e:sharedFighterObservable.items()){
                sf[e.key()]["isAlive"]=e.value().at("isAlive");
                if(e.key()!=p->getFullName()){
                    double sent=e.value().at("time");
                    if(lastSharedTime.count(e.key())==0 || lastSharedTime[e.key()]<sent){
                        lastSharedTime[e.key()]=sent;
                        if(e.value()["isAlive"].get<bool>()){
                            std::vector<Track3D> shared=e.value().at("/sensor/radar/track"_json_pointer);
                            sf[e.key()]["sensor"]=e.value()["sensor"];
                            for(auto&& rhs:shared){
                                same=Track3D();
                                sameID=-1;
                                idx=0;
                                for(auto& lhs:p->track){
                                    if(lhs.isSame(rhs)){
                                        same=lhs;
                                        sameID=idx;
                                        idx++;
                                    }
                                }
                                if(same.is_none()){
                                    p->track.push_back(rhs);
                                    p->trackSource.push_back({e.key()});
                                }else{
                                    same.addBuffer(rhs);
                                    p->trackSource[sameID].push_back(e.key());
                                }
                            }
                        }
                    }
                }
            }
            for(auto& t:p->track){
                t.merge();
            }
        }
        p->observables["sensor"]={
            {"radar",p->radar.lock()->observables},
            {"mws",p->mws.lock()->observables}
        };
        nl::json j_track=nl::json::array();
        for(auto&& e:p->track){
            j_track.push_back(Track3D(e));
        }
        p->observables["sensor"]["track"]=j_track;
        p->observables["sensor"]["trackSource"]=p->trackSource;
        if(!p->target.is_none()){
            Track3D old=p->target;
            p->target=Track3D();
            p->targetID=-1;
            int i=0;
            for(auto& t:p->track){
                if(old.isSame(t)){
                    p->target=t;
                    p->targetID=i;
                }
                ++i;
            }
        }else{
            p->targetID=-1;
        }
        for(int mslId=0;mslId<p->nextMsl;++mslId){
            Track3D old=p->missileTargets[mslId].first;
            p->missileTargets[mslId].second=false;
            for(auto& t:p->track){
                if(old.isSame(t)){
                    p->missileTargets[mslId].first=t;
                    p->missileTargets[mslId].second=true;
                    break;
                }
            }
        }
    }else{
        p->track.clear();
    }
}
Fighter::OtherDataSharer::OtherDataSharer(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
void Fighter::OtherDataSharer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(p->isDatalinkEnabled){
            nl::json dataPacket={
                {"fighterObservables",{
                    {p->getFullName(),{
                        {"isAlive",p->isAlive()},
                        {"spec",p->observables["spec"]},
                        {"motion",p->observables["motion"]},
                        {"propulsion",p->observables["propulsion"]},
                        {"weapon",p->observables["weapon"]},
                        {"time",manager->getTime()}
                    }}
                }}
            };
            if(p->hasAgent){
                dataPacket["agentObservables"]=nl::json::object();
                nl::json& agObs=dataPacket["agentObservables"];
                for(auto&&e:p->agent.lock()->observables.items()){
                    agObs[e.key()]={
                        {"obs",e.value()},
                        {"time",manager->getTime()}
                    };
                }
            }
            //撃墜された味方の射撃済誘導弾に関する諸元更新を引き継ぎ
            bool isFirst=false;//自身が生存機の中で先頭かどうか
            for(auto&& asset:manager->getAssets()){
                if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                    auto f=getShared<const Fighter>(asset);
                    if(f->isAlive()){
                        isFirst = f->getFullName()==p->getFullName();
                        break;
                    }
                }
            }
            if(isFirst){
                for(auto&& asset:manager->getAssets()){
                    if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                        auto f=getShared<const Fighter>(asset);
                        if(!f->isAlive()){
                            nl::json mslObs=nl::json::array();
                            for(auto&& e:f->missiles){
                                mslObs.push_back(e.lock()->observables);
                            }
                            dataPacket["fighterObservables"][f->getFullName()]["weapon"]={{"missiles",std::move(mslObs)}};
                            dataPacket["fighterObservables"][f->getFullName()]["time"]=manager->getTime();
                        }
                    }
                }
            }
            //送信
            p->communicationBuffers[p->datalinkName].lock()->send(dataPacket,CommunicationBuffer::MERGE);
        }
    }
}
void Fighter::OtherDataSharer::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_ && p->isDatalinkEnabled){
        if(p->hasAgent){
            nl::json agCom=nl::json::object();
            for(auto&&e:p->agent.lock()->commands.items()){
                agCom[e.key()]={
                    {"com",e.value()},
                    {"time",manager->getTime()}
                };
            }
            p->communicationBuffers[p->datalinkName].lock()->send({
                {"agentCommands",agCom}
            },CommunicationBuffer::MERGE);
        }
    }
}
Fighter::OtherDataSanitizer::OtherDataSanitizer(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    lastSharedTimeOfAgentCommand=-1;
}
void Fighter::OtherDataSanitizer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        std::pair<double,nl::json> sharedAgentObservable={-1,nl::json::object()};
        std::pair<double,nl::json> sharedFighterObservable={-1,nl::json::object()};
        if(p->isDatalinkEnabled){
            sharedAgentObservable=p->communicationBuffers[p->datalinkName].lock()->receive("agentObservables");
            if(sharedAgentObservable.first<0){//invalid data
                sharedAgentObservable.second=nl::json::object();
            }
            sharedFighterObservable=p->communicationBuffers[p->datalinkName].lock()->receive("fighterObservables");
            if(sharedFighterObservable.first<0){//invalid data
                sharedFighterObservable.second=nl::json::object();
            }
        }
        nl::json& sa=p->observables["shared"]["agent"];
        sa=nl::json::object();
        for(auto&& e:sharedAgentObservable.second.items()){
            double sent=e.value().at("time");
            if(lastSharedTimeOfAgentObservable.count(e.key())==0 || lastSharedTimeOfAgentObservable[e.key()]<sent){
                lastSharedTimeOfAgentObservable[e.key()]=sent;
                sa[e.key()]=e.value().at("obs");
            }
        }
        if(p->hasAgent){
            sa.merge_patch(p->agent.lock()->observables);
        }
        nl::json& sf=p->observables["shared"]["fighter"];
        for(auto&& e:sharedFighterObservable.second.items()){
            double sent=e.value().at("time");
            sf[e.key()]["isAlive"]=e.value().at("isAlive");
            if(lastSharedTimeOfFighterObservable.count(e.key())==0 || lastSharedTimeOfFighterObservable[e.key()]<sent){
                lastSharedTimeOfFighterObservable[e.key()]=sent;
                sf[e.key()]["spec"]=e.value().at("spec");
                if(e.value().at("isAlive").get<bool>()){
                    sf[e.key()]["motion"]=e.value().at("motion");
                }
                sf[e.key()]["motion"]=e.value().at("motion");
                sf[e.key()]["propulsion"]=e.value().at("propulsion");
                sf[e.key()]["weapon"]=e.value().at("weapon");
            }
        }
    }
}
void Fighter::OtherDataSanitizer::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        nl::json agentCommand={
            {"motion",{
                {"roll",0.0},
                {"pitch",0.0},
                {"yaw",0.0},
                {"accel",0.0}
            }},
            {"weapon",{
                {"launch",false},
                {"target",Track3D()}
            }}
        };
        if(p->hasAgent){
            agentCommand=p->agent.lock()->commands.at(p->getFullName());
        }else{
            if(p->isDatalinkEnabled){
                auto tmp=p->communicationBuffers[p->datalinkName].lock()->receive("agentCommands");
                if(tmp.first>=0){//valid data
                    auto sharedAgentCommand=tmp.second;
                    if(sharedAgentCommand.contains(p->getFullName())){
                        double sent=sharedAgentCommand.at(p->getFullName()).at("time");
                        if(lastSharedTimeOfAgentCommand<0 || lastSharedTimeOfAgentCommand<sent){
                            lastSharedTimeOfAgentCommand=sent;
                            agentCommand=sharedAgentCommand.at(p->getFullName()).at("com");
                        }
                    }
                }
            }
        }
        p->commands["fromAgent"]=agentCommand;
        p->commands["motion"]=agentCommand["motion"];//もしHumanInterventionが無い場合はそのまま使われる
        p->commands["weapon"]=agentCommand["weapon"];//もしHumanInterventionが無い場合はそのまま使われる
    }
}
Fighter::HumanIntervention::HumanIntervention(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    capacity=getValueFromJsonKRD(modelConfig,"capacity",randomGen,1);
    delay=getValueFromJsonKRD(modelConfig,"delay",randomGen,3.0);
    cooldown=getValueFromJsonKRD(modelConfig,"cooldown",randomGen,0.999);
}
void Fighter::HumanIntervention::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        p->commands["weapon"]={
            {"launch",false},
            {"target",Track3D()}
        };
        if(p->launchable && p->commands["fromAgent"]["weapon"]["launch"]){
            if(recognizedShotCommands.size()<capacity
                && (recognizedShotCommands.size()==0 || manager->getTime()>=recognizedShotCommands.back().first+cooldown)
            ){
                recognizedShotCommands.push_back(std::make_pair(manager->getTime(),p->commands["fromAgent"]["weapon"]["target"]));
                if(recognizedShotCommands.size()>=capacity){
                    p->launchable=false;
                }
            }
        }
        if(recognizedShotCommands.size()>0){
            auto front=recognizedShotCommands.front();
            if(manager->getTime()>=front.first+delay){
                //承認
                p->commands["weapon"]={
                    {"launch",true},
                    {"target",front.second}
                };
                recognizedShotCommands.pop_front();
                if(recognizedShotCommands.size()<capacity){
                    p->launchable=true;
                }
            }else{
                //判断中
            }
        }
        p->commands["motion"]=p->commands["fromAgent"]["motion"];
    }
}
Fighter::WeaponController::WeaponController(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
void Fighter::WeaponController::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(!p->commands.contains("weapon")){
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
        }
        std::pair<bool,Track3D> trackingInfo=p->isTracking(p->commands["weapon"]["target"].get<Track3D>());
        p->target=trackingInfo.second;
        bool launchFlag=p->commands["weapon"]["launch"].get<bool>() && trackingInfo.first;
        if(p->remMsls==0){
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
            p->launchable=false;
        }else if(launchFlag){
            p->missileTargets[p->nextMsl]=std::make_pair(p->target,true);
            p->communicationBuffers["MissileComm:"+p->missiles[p->nextMsl].lock()
                ->getFullName()].lock()->send({
                    {"launch",true},
                    {"target",p->missileTargets[p->nextMsl]}
                },CommunicationBuffer::MERGE);
            p->nextMsl+=1;
            p->remMsls-=1;
            manager->triggerEvent("Shot",{this->weak_from_this()});
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
        }
        for(int mslId=0;mslId<p->nextMsl;++mslId){
            if(p->missileTargets[mslId].second){
                p->communicationBuffers["MissileComm:"+p->missiles[mslId].lock()
                ->getFullName()].lock()->send({
                    {"target",p->missileTargets[mslId].first}
                },CommunicationBuffer::MERGE);
            }
        }
        //撃墜された味方の射撃済誘導弾に対する制御を引き継ぎ
        nl::json& sf=p->observables["shared"]["fighter"];
        bool isFirst=false;//自身が生存機の中で先頭かどうか
        for(auto&& asset:manager->getAssets()){
            if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                auto f=getShared<const Fighter>(asset);
                if(f->isAlive()){
                    isFirst = f->getFullName()==p->getFullName();
                    break;
                }
            }
        }
        if(isFirst){
            for(auto&& asset:manager->getAssets()){
                if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                    auto f=getShared<const Fighter>(asset);
                    if(!f->isAlive()){
                        for(auto&& e:f->missiles){
                            auto msl=e.lock();
                            if(msl->isAlive() && msl->hasLaunched){
                                for(auto& t:p->track){
                                    if(msl->target.isSame(t)){
                                        p->communicationBuffers["MissileComm:"+msl->getFullName()].lock()->send({
                                            {"target",t}
                                        },CommunicationBuffer::MERGE);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
Fighter::FlightController::FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Controller(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    mode="direct";
}
Fighter::FlightController::~FlightController(){
}
void Fighter::FlightController::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(!p->commands.contains("motion")){
            p->commands["motion"]=getDefaultCommand();
        }
        commands["motion"]=calc(p->commands["motion"]);
    }
}
nl::json Fighter::FlightController::getDefaultCommand(){
    std::cout<<"Warning! Fighter::FlightController::getDefaultCommand() should be overridden."<<std::endl;
    return nl::json::object();
}
nl::json Fighter::FlightController::calc(const nl::json &cmd){
    std::cout<<"Warning! Fighter::FlightController::calc(const nl::json&) should be overridden."<<std::endl;
    return cmd;
}
void Fighter::FlightController::setMode(const std::string& ctrlName){
    mode=ctrlName;
}
FighterAccessor::FighterAccessor(std::shared_ptr<Fighter> a)
:PhysicalAssetAccessor(a){
    asset=a;
}
FighterAccessor::~FighterAccessor(){
}
void FighterAccessor::setFlightControllerMode(const std::string& ctrlName){
    asset.lock()->setFlightControllerMode(ctrlName);
}
double FighterAccessor::getMaxReachableRange(){
    return asset.lock()->getMaxReachableRange();
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    return asset.lock()->getRmax(rs,vs,rt,vt);
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa){
    return asset.lock()->getRmax(rs,vs,rt,vt,aa);
}

void exportFighter(py::module& m)
{
    using namespace pybind11::literals;
    auto clsObj=EXPOSE_CLASS(Fighter)
    DEF_FUNC(Fighter,makeChildren)
    DEF_FUNC(Fighter,validate)
    DEF_FUNC(Fighter,setDependency)
    DEF_FUNC(Fighter,perceive)
    DEF_FUNC(Fighter,control)
    DEF_FUNC(Fighter,behave)
    DEF_FUNC(Fighter,kill)
    DEF_FUNC(Fighter,calcOptimumCruiseFuelFlowRatePerDistance)
    DEF_FUNC(Fighter,getMaxReachableRange)
    .def("isTracking",py::overload_cast<std::weak_ptr<PhysicalAsset>>(&Fighter::isTracking))
    .def("isTracking",py::overload_cast<const Track3D&>(&Fighter::isTracking))
    .def("isTracking",py::overload_cast<const boost::uuids::uuid&>(&Fighter::isTracking))
    DEF_FUNC(Fighter,setFlightControllerMode)
    DEF_FUNC(Fighter,calcMotion)
    DEF_FUNC(Fighter,toEulerAngle)
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>(&Fighter::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&>(&Fighter::getRmax))
    DEF_FUNC(Fighter,relHtoI)
    DEF_FUNC(Fighter,relItoH)
    DEF_FUNC(Fighter,absHtoI)
    DEF_FUNC(Fighter,absItoH)
    DEF_FUNC(Fighter,getAccessor)
    DEF_READWRITE(Fighter,m)
    DEF_READWRITE(Fighter,rcsScale)
    DEF_READWRITE(Fighter,fuelCapacity)
    DEF_READWRITE(Fighter,engine)
    DEF_READWRITE(Fighter,radar)
    DEF_READWRITE(Fighter,mws)
    DEF_READWRITE(Fighter,missiles)
    DEF_READWRITE(Fighter,dummyMissile)
    DEF_READWRITE(Fighter,missileTargets)
    DEF_READWRITE(Fighter,nextMsl)
    DEF_READWRITE(Fighter,numMsls)
    DEF_READWRITE(Fighter,remMsls)
    DEF_READWRITE(Fighter,isDatalinkEnabled)
    DEF_READWRITE(Fighter,track)
    DEF_READWRITE(Fighter,trackSource)
    DEF_READWRITE(Fighter,datalinkName)
    DEF_READWRITE(Fighter,launchable)
    DEF_READWRITE(Fighter,target)
    DEF_READWRITE(Fighter,targetID)
    DEF_READWRITE(Fighter,fuelRemaining)
    DEF_READWRITE(Fighter,optCruiseFuelFlowRatePerDistance)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,SensorDataSharer)
    DEF_FUNC(Fighter::SensorDataSharer,perceive)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,SensorDataSanitizer)
    DEF_FUNC(Fighter::SensorDataSanitizer,perceive)
    DEF_READWRITE(Fighter::SensorDataSanitizer,lastSharedTime)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,OtherDataSharer)
    DEF_FUNC(Fighter::OtherDataSharer,perceive)
    DEF_FUNC(Fighter::OtherDataSharer,control)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,OtherDataSanitizer)
    DEF_FUNC(Fighter::OtherDataSanitizer,perceive)
    DEF_FUNC(Fighter::OtherDataSanitizer,control)
    DEF_READWRITE(Fighter::OtherDataSanitizer,lastSharedTimeOfAgentObservable)
    DEF_READWRITE(Fighter::OtherDataSanitizer,lastSharedTimeOfFighterObservable)
    DEF_READWRITE(Fighter::OtherDataSanitizer,lastSharedTimeOfAgentCommand)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,HumanIntervention)
    DEF_FUNC(Fighter::HumanIntervention,control)
    DEF_READWRITE(Fighter::HumanIntervention,capacity)
    DEF_READWRITE(Fighter::HumanIntervention,recognizedShotCommands)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,WeaponController)
    DEF_FUNC(Fighter::WeaponController,control)
    ;
    EXPOSE_INNER_CLASS(clsObj,Fighter,FlightController)
    DEF_FUNC(Fighter::FlightController,control)
    DEF_FUNC(Fighter::FlightController,getDefaultCommand)
    DEF_FUNC(Fighter::FlightController,calc)
    .def("calc",[](Fighter::FlightController& v,const py::object &cmd){
        return v.calc(cmd);
    })
    DEF_FUNC(Fighter::FlightController,setMode)
    ;
    EXPOSE_CLASS_WITHOUT_INIT(FighterAccessor)
    .def(py::init<std::shared_ptr<Fighter>>())
    DEF_FUNC(FighterAccessor,setFlightControllerMode)
    DEF_FUNC(FighterAccessor,getMaxReachableRange)
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>(&FighterAccessor::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&>(&FighterAccessor::getRmax))
    ;
}
