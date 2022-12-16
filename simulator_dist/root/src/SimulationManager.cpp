// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <queue>
#include <fstream>
#include <chrono>
#include <future>
#include <atomic>
#include <magic_enum/magic_enum.hpp>
#include "SimulationManager.h"
#include "Utility.h"
#include "Asset.h"
#include "PhysicalAsset.h"
#include "Agent.h"
#include "Controller.h"
#include "CommunicationBuffer.h"
#include "Callback.h"
#include "Ruler.h"
#include "Reward.h"
#include "Viewer.h"
using namespace util;

bool OrderComparer::operator()(const OrderComparer::Type& lhs,const OrderComparer::Type& rhs) const{
    //(nextTick,priority)
    if(lhs.first==rhs.first){
        return lhs.second<rhs.second;
    }else{
        return lhs.first<rhs.first;
    }
}
SimulationManager::SimulationManager()
:assetPhases{SimPhase::VALIDATE,SimPhase::PERCEIVE,SimPhase::CONTROL,SimPhase::BEHAVE},
callbackPhases{SimPhase::ON_GET_OBSERVATION_SPACE,SimPhase::ON_GET_ACTION_SPACE,SimPhase::ON_MAKE_OBS,SimPhase::ON_DEPLOY_ACTION,SimPhase::ON_EPISODE_BEGIN,SimPhase::ON_VALIDATION_END,SimPhase::ON_STEP_BEGIN,SimPhase::ON_INNERSTEP_BEGIN,SimPhase::ON_INNERSTEP_END,SimPhase::ON_STEP_END,SimPhase::ON_EPISODE_END},
pool(1){
    tickCount=0;
    baseTimeStep=0.1;
    randomGen=std::mt19937(std::random_device()());
}
SimulationManager::SimulationManager(const nl::json& config_,int worker_index_,int vector_index_,std::function<nl::json(const nl::json&,int,int)> overrider_)
:SimulationManager(){
    try{
        worker_index=worker_index_;
        vector_index=vector_index_;
        nl::json config=overrider_(parseConfig(config_),worker_index,vector_index);
        managerConfig=config.at("Manager");
        if(config.contains("Factory")){
            factory.addModelsFromJson(config.at("Factory"));
        }
    }catch(std::exception& ex){
        std::cout<<"In SimulationManager::ctor, parsing the config failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        std::cout<<"config_="<<config_<<std::endl;
        std::cout<<"worker_index_="<<worker_index_<<std::endl;
        std::cout<<"vector_index_="<<vector_index_<<std::endl;
        throw ex;
    }
}
SimulationManager::~SimulationManager(){
}
nl::json SimulationManager::parseConfig(const nl::json& config_){
    nl::json ret=nl::json::object();
    if(config_.is_object()){
        ret=config_;
    }else if(config_.is_array()){
        for(auto& e:config_){
            if(e.is_object()){
                ret.merge_patch(e);
            }else if(e.is_string()){
                std::ifstream ifs(e.get<std::string>());
                nl::json sub;
                ifs>>sub;
                ret.merge_patch(sub);
            }else{
                throw std::runtime_error("invalid config type.");
            }
        }
    }else{
        throw std::runtime_error("invalid config type.");
    }
    return ret;
}
void SimulationManager::setViewerType(const std::string& viewerType){
    auto mAcc=SimulationManagerAccessorForCallback::create(this->shared_from_this());
    viewer=factory.create<Viewer>("Viewer",viewerType,{
        {"manager",mAcc},
        {"baseTimeStep",baseTimeStep},
        {"name","Viewer"}
    });
}
void SimulationManager::seed(const unsigned int& seed_){
    seedValue=seed_;
    randomGen.seed(seed_);
    assetConfigDispatcher.seed(seed_);
    agentConfigDispatcher.seed(seed_);
}

void SimulationManager::configure(){
    tickCount=0;
    measureTime=getValueFromJsonKRD(managerConfig,"measureTime",randomGen,false);
    numThreads=getValueFromJsonKRD(managerConfig,"numThreads",randomGen,1);
    pool.reset(numThreads);
    exposeDeadAgentReward=getValueFromJsonKRD(managerConfig,"exposeDeadAgentReward",randomGen,true);
    baseTimeStep=managerConfig.at("/TimeStep/baseTimeStep"_json_pointer);
    agentInterval=managerConfig.at("/TimeStep/agentInterval"_json_pointer);
    try{
        if(managerConfig.contains("ViewerType")){
            setViewerType(managerConfig.at("ViewerType"));
        }else{
            setViewerType("None");
        }
    }catch(std::exception& ex){
        std::cout<<"setup of the Viewer failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    eventHandlers.clear();
    //callbacks.clear();
    //loggers.clear();
    auto mAcc=SimulationManagerAccessorForCallback::create(this->shared_from_this());
    try{
        ruler=factory.create<Ruler>("Ruler",managerConfig.at("Ruler"),{
            {"manager",mAcc},
            {"baseTimeStep",baseTimeStep},
            {"name","Ruler"}
        });   
    }catch(std::exception& ex){
        std::cout<<"setup of the Ruler failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    rewardGenerators.clear();
    nl::json subConfig;
    if(managerConfig.contains("Rewards")){
        subConfig=managerConfig.at("Rewards");
    }else{
        subConfig={{{"model","ScoreReward"},{"target","All"}}};
    }
    for(std::size_t i=0;i<subConfig.size();++i){
        nl::json elem=subConfig.at(i);
        try{
            mAcc=SimulationManagerAccessorForCallback::create(this->shared_from_this());
            rewardGenerators.push_back(factory.create<Reward>("Reward",elem.at("model"),{
                {"manager",mAcc},
                {"baseTimeStep",baseTimeStep},
                {"name","Reward"+std::to_string(i+1)},
                {"target",elem.at("target")}
            }));
        }catch(std::exception& ex){
            std::cout<<"Creation of a reward failed. config="<<elem<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    //origin=managerConfig.at("Origin");
    try{
        assetConfigDispatcher.initialize(managerConfig.at("AssetConfigDispatcher"));
    }catch(std::exception& ex){
        std::cout<<"initialization of assetConfigDispatcher failed."<<std::endl;
        std::cout<<"config="<<managerConfig.at("AssetConfigDispatcher")<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    try{
        agentConfigDispatcher.initialize(managerConfig.at("AgentConfigDispatcher"));
    }catch(std::exception& ex){
        std::cout<<"initialization of agentConfigDispatcher failed."<<std::endl;
        std::cout<<"config="<<managerConfig.at("AgentConfigDispatcher")<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    generateAssets();
    if(managerConfig.contains("Callbacks")){
        subConfig=managerConfig.at("Callbacks");
    }else{
        subConfig=nl::json::object();
    }
    for(auto& e:subConfig.items()){
        try{
            if(callbacks.count(e.key())==0 || callbacks[e.key()]->acceptReconfigure){
                mAcc=SimulationManagerAccessorForCallback::create(this->shared_from_this());
                if(e.value().contains("class")){
                    //classでの指定(configはmodelConfig)
                    callbacks[e.key()]=factory.createByClassName<Callback>("Callback",e.value()["class"],e.value()["config"],{
                        {"manager",mAcc},
                        {"baseTimeStep",baseTimeStep},
                        {"name",e.key()}
                    });
                }else if(e.value().contains("model")){
                    //modelでの指定(configはinstanceConfig)
                    nl::json ic=e.value()["config"];
                    ic.merge_patch({
                        {"manager",mAcc},
                        {"baseTimeStep",baseTimeStep},
                        {"name",e.key()}
                    });
                    callbacks[e.key()]=factory.create<Callback>("Callback",e.value()["model"],ic);
                }else{
                    throw std::runtime_error("A config for callbacks must contain 'class' or 'model' key.");
                }
            }
        }catch(std::exception& ex){
            std::cout<<"Creation of a callback failed. config={"<<e.key()<<":"<<e.value()<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    if(managerConfig.contains("Loggers")){
        subConfig=managerConfig.at("Loggers");
    }else{
        subConfig=nl::json::object();
    }
    for(auto& e:subConfig.items()){
        try{
            if(loggers.count(e.key())==0 || loggers[e.key()]->acceptReconfigure){
                mAcc=SimulationManagerAccessorForCallback::create(this->shared_from_this());
                if(e.value().contains("class")){
                    //classでの指定(configはmodelConfig)
                    loggers[e.key()]=factory.createByClassName<Callback>("Callback",e.value()["class"],e.value()["config"],{
                        {"manager",mAcc},
                        {"baseTimeStep",baseTimeStep},
                        {"name",e.key()}
                    });
                }else if(e.value().contains("model")){
                    //modelでの指定(configはinstanceConfig)
                    nl::json ic=e.value()["config"];
                    ic.merge_patch({
                        {"manager",mAcc},
                        {"baseTimeStep",baseTimeStep},
                        {"name",e.key()}
                    });
                    loggers[e.key()]=factory.create<Callback>("Callback",e.value()["model"],ic);
                }else{
                    throw std::runtime_error("A config for loggers must contain 'class' or 'model' key.");
                }
            }
        }catch(std::exception& ex){
            std::cout<<"Creation of a logger failed. config={"<<e.key()<<":"<<e.value()<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    resetCallbacks();
    get_observation_space();
    get_action_space();
    reconfigureRequested=false;
    reconfigureManagerReplacer=nl::json();
    reconfigureFactoryReplacer=nl::json();
}
void SimulationManager::checkOrder(){
    std::queue<std::shared_ptr<Asset>> q;
    for(auto& e:agents){
        e.second->dependencyChecker->clearReadiness();
        e.second->dependencyChecker->clearDependency();
    }
    for(auto& e:controllers){
        e.second->dependencyChecker->clearReadiness();
        e.second->dependencyChecker->clearDependency();
    }
    for(auto& e:assets){
        e.second->dependencyChecker->clearReadiness();
        e.second->dependencyChecker->clearDependency();
    }
    for(auto& e:agents){
        e.second->setDependency();
    }
    for(auto& e:controllers){
        e.second->setDependency();
    }
    for(auto& e:assets){
        e.second->setDependency();
    }
    orderedAssets.clear();
    indicedAssets.clear();
    assetDependsOn.clear();
    assetDependedOnBy.clear();
    for(auto& phase:assetPhases){
        std::size_t count=0;
        std::shared_ptr<Asset> now;
        try{
            for(auto& e:agents){
                q.push(e.second);
            }
            for(auto& e:controllers){
                q.push(e.second);
            }
            for(auto& e:assets){
                q.push(e.second);
            }
            while(!q.empty()){
                now=q.front();
                q.pop();
                if(now->dependencyChecker->checkReadiness(phase)){
                    std::uint64_t first=now->getFirstTick(phase);
                    if(phase==SimPhase::PERCEIVE && first==0){
                        first=now->getNextTick(phase,first);
                        assert(first>0);
                    }
                    while(first<0){
                        std::uint64_t tmp=now->getNextTick(phase,first);
                        assert(tmp>first);
                        first=tmp;
                    }
                    orderedAssets[phase][std::make_pair(first,count)]=now;
                    now->callOrderIndex[phase]=count;
                    indicedAssets[phase].push_back(std::make_pair(now,first));
                    for(auto&& d:now->dependencyChecker->dependency[phase]){
                        assetDependsOn[phase][count].push_back(d.lock()->callOrderIndex[phase]);
                        assetDependedOnBy[phase][d.lock()->callOrderIndex[phase]].push_back(count);
                    }
                    if(measureTime){
                        std::string phaseName=std::string(magic_enum::enum_name(phase));
                        maxProcessTime[phaseName][now->getFullName()]=0;
                        minProcessTime[phaseName][now->getFullName()]=std::numeric_limits<std::uint64_t>::max();
                        meanProcessTime[phaseName][now->getFullName()]=0;
                        processCount[phaseName][now->getFullName()]=0;
                    }
                    count++;
                }else{
                    q.push(now);
                }
            }
        }catch(std::exception& ex){
            std::cout<<"checkOrder failed. phase="<<magic_enum::enum_name(phase)<<", asset="<<now->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
        if(orderedAssets[phase].size()>0){
            nextTick[phase]=(*orderedAssets[phase].begin()).first.first;
        }else{
            nextTick[phase]=std::numeric_limits<std::uint64_t>::max();
        }
    }
}
void SimulationManager::printOrderedAssets(){
    for(auto& phase:assetPhases){
        std::cout<<"=========================="<<std::endl;
        std::cout<<"orderedAssets["<<magic_enum::enum_name(phase)<<"]="<<std::endl;
        for(auto&& e:orderedAssets[phase]){
            std::cout<<e.second->getFullName()<<std::endl;
            for(auto&& s:e.second->dependencyChecker->dependency[phase]){
                std::cout<<"    depends on "<<s.lock()->getFullName()<<std::endl;
            }
        }
    }
}
void SimulationManager::requestReconfigure(const nl::json& managerReplacer,const nl::json& factoryReplacer){
    reconfigureRequested=true;
    reconfigureManagerReplacer=managerReplacer;
    reconfigureFactoryReplacer=factoryReplacer;
}
void SimulationManager::resetCallbacks(){
    orderedAllCallbacks1.clear();
    orderedAllCallbacks2.clear();
    for(auto&&phase: callbackPhases){
        std::string phaseName=std::string(magic_enum::enum_name(phase));
        std::size_t count=0;
        orderedAllCallbacks1[phase][std::make_pair(ruler->getFirstTick(phase),count)]=ruler;
        if(measureTime){
            maxProcessTime[phaseName][ruler->getName()]=0;
            minProcessTime[phaseName][ruler->getName()]=std::numeric_limits<std::uint64_t>::max();
            meanProcessTime[phaseName][ruler->getName()]=0;
            processCount[phaseName][ruler->getName()]=0;
        }
        count++;
        for(auto& r:rewardGenerators){
            orderedAllCallbacks1[phase][std::make_pair(r->getFirstTick(phase),count)]=r;
            if(measureTime){
                maxProcessTime[phaseName][r->getName()]=0;
                minProcessTime[phaseName][r->getName()]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][r->getName()]=0;
                processCount[phaseName][r->getName()]=0;
            }
            count++;
        }
        for(auto& c:callbacks){
            orderedAllCallbacks2[phase][std::make_pair(c.second->getFirstTick(phase),count)]=c.second;
            if(measureTime){
                maxProcessTime[phaseName][c.second->getName()]=0;
                minProcessTime[phaseName][c.second->getName()]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][c.second->getName()]=0;
                processCount[phaseName][c.second->getName()]=0;
            }
            count++;
        }
        orderedAllCallbacks2[phase][std::make_pair(viewer->getFirstTick(phase),count)]=viewer;
        if(measureTime){
            maxProcessTime[phaseName][viewer->getName()]=0;
            minProcessTime[phaseName][viewer->getName()]=std::numeric_limits<std::uint64_t>::max();
            meanProcessTime[phaseName][viewer->getName()]=0;
            processCount[phaseName][viewer->getName()]=0;
        }
        count++;
        for(auto& l:loggers){
            orderedAllCallbacks2[phase][std::make_pair(l.second->getFirstTick(phase),count)]=l.second;
            if(measureTime){
                maxProcessTime[phaseName][l.second->getName()]=0;
                minProcessTime[phaseName][l.second->getName()]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][l.second->getName()]=0;
                processCount[phaseName][l.second->getName()]=0;
            }
            count++;
        }
    }
}
py::dict SimulationManager::reset(){
    tickCount=0;
    nextTick.clear();
    nextAgentTick=tickCount+agentInterval;
    eventHandlers.clear();
    communicationBuffers.clear();
    maxProcessTime.clear();
    minProcessTime.clear();
    meanProcessTime.clear();
    processCount.clear();
    if(reconfigureRequested){
        managerConfig.merge_patch(reconfigureManagerReplacer);
        factory.reconfigureModelConfig(reconfigureFactoryReplacer);
        configure();
        generateCommunicationBuffers();
    }else{
        generateAssets();
        generateCommunicationBuffers();
        resetCallbacks();
        get_observation_space();
        get_action_space();
    }
    py::gil_scoped_release release;
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_BEGIN,true);
    manualDone=false;
    dones.clear();
    gatherDones();
    scores=ruler->score;
    rewards.clear();
    gatherRewards();
    totalRewards.clear();
    gatherTotalRewards();
    for(auto&& comm:communicationBuffers){
        comm.second->validate();
    }
    std::chrono::system_clock::time_point before,after;
    std::chrono::microseconds difference;
    for(auto&& asset:indicedAssets[SimPhase::VALIDATE]){
        try{
            if(measureTime){
                before=std::chrono::system_clock::now();
            }
            asset.first->validate();
            if(measureTime){
                std::string name=asset.first->getFullName();
                after=std::chrono::system_clock::now();
                difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                std::string phaseName=std::string(magic_enum::enum_name(SimPhase::VALIDATE));
                maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                processCount[phaseName][name]++;
                meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
            }
        }catch(std::exception& ex){
            std::cout<<"asset.validate() failed. asset="<<asset.first->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_VALIDATION_END,true);
    runCallbackPhaseFunc(SimPhase::ON_VALIDATION_END,false);
    for(auto& asset:indicedAssets[SimPhase::PERCEIVE]){
        try{
            if(asset.first->isAlive()){
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                asset.first->perceive(true);
                if(measureTime){
                    std::string name=asset.first->getFullName();
                    after=std::chrono::system_clock::now();
                    std::string phaseName=std::string(magic_enum::enum_name(SimPhase::PERCEIVE));
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
            }
        }catch(std::exception& ex){
            std::cout<<"asset.perceive(true) failed. asset="<<asset.first->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    py::gil_scoped_acquire acquire;
    lastObservations.clear();
    auto obs=makeObs();
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_BEGIN,false);
    return obs;
}
py::tuple SimulationManager::step(const py::dict& action){
    deployAction(action);
    py::gil_scoped_release release;
    manualDone=false;
    runCallbackPhaseFunc(SimPhase::ON_STEP_BEGIN,true);
    runCallbackPhaseFunc(SimPhase::ON_STEP_BEGIN,false);
    while(tickCount<nextAgentTick){
        //std::cout<<"tickCount="<<tickCount<<",nextAgentTick="<<nextAgentTick<<std::endl;
        nextTickCB=std::min(nextTick[SimPhase::ON_INNERSTEP_BEGIN],std::min(nextTick[SimPhase::CONTROL],nextTick[SimPhase::BEHAVE]));
        nextTickP=std::min(nextTick[SimPhase::PERCEIVE],nextTick[SimPhase::ON_INNERSTEP_END]);
        innerStep();
    }
    runCallbackPhaseFunc(SimPhase::ON_STEP_END,true);
    py::gil_scoped_acquire acquire;
    auto obs=makeObs();
    scores=ruler->score;
    gatherRewards();
    gatherTotalRewards();
    gatherDones();
    py::dict infos,info;
    info["score"]=util::todict(ruler->score);
    info["w"]=worker_index;
    info["v"]=vector_index;
    if(dones["__all__"]){
        info["endReason"]=ruler->endReason;
    }
    for(auto&& e:obs){
        infos[e.first]=info;
    }
    runCallbackPhaseFunc(SimPhase::ON_STEP_END,false);
    if(dones["__all__"]){
        runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,true);
        runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,false);
        if(measureTime){
            nl::json tmp=minProcessTime;
            std::cout<<"minProcessTime="<<tmp.dump(1)<<std::endl;
            tmp=maxProcessTime;
            std::cout<<"maxProcessTime="<<tmp.dump(1)<<std::endl;
            tmp=meanProcessTime;
            std::cout<<"meanProcessTime="<<tmp.dump(1)<<std::endl;
        }
    }
    nextAgentTick+=agentInterval;
    return py::make_tuple(
        obs,
        util::todict(rewards),
        util::todict(dones),
        infos
    );
}
void SimulationManager::stopEpisodeExternally(void){
    dones["__all__"]=true;
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,true);
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,false);
}
void SimulationManager::runAssetPhaseFunc(const SimPhase& phase){
    //後の処理で各phaseごとの関数呼び出し形式を共通化しておく
    static const std::map<SimPhase,std::function<void(std::shared_ptr<Asset>)>> phaseFunc={
        {SimPhase::PERCEIVE,[](std::shared_ptr<Asset> a){a->perceive(false);}},
        {SimPhase::CONTROL,[](std::shared_ptr<Asset> a){a->control();}},
        {SimPhase::BEHAVE,[](std::shared_ptr<Asset> a){a->behave();}}
    };
    assert(phaseFunc.count(phase)>0);
    //Asset::kill()の待機リストを初期化
    killRequests.clear();
    //phaseFunc実行中のエラー監視用フラグ。子スレッドからの例外送出を拾わないため必要。
    std::atomic_bool success=true;

    //この時刻のこのphaseで処理すべきAssetの範囲を取得する
    auto oit=orderedAssets[phase].begin();
    bool finished=orderedAssets[phase].size()==0;//will be false
    std::vector<std::size_t> processed;
    while(!finished){
        if(oit==orderedAssets[phase].end()){
            break;
        }
        auto asset=*oit;
        if(asset.first.first>tickCount){
            finished=true;
        }else{
            oit=orderedAssets[phase].erase(oit);
            processed.push_back(asset.second->callOrderIndex[phase]); 
        }
    }
    if(numThreads==1){
        //シングルスレッドの場合、先頭から順番に処理すればよい
        std::chrono::system_clock::time_point before,after;
        std::chrono::microseconds difference;
        std::string phaseName=std::string(magic_enum::enum_name(phase));
        auto wit=processed.begin();
        while(wit!=processed.end()){
            std::size_t index=*wit;
            auto asset=indicedAssets[phase][index].first;
            std::string name=asset->getFullName();
            if(asset->isAlive()){
                try{
                    if(measureTime){
                        before=std::chrono::system_clock::now();
                    }
                    phaseFunc.at(phase)(asset);
                    if(measureTime){
                        after=std::chrono::system_clock::now();
                        difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                        maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                        minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                        processCount[phaseName][name]++;
                        meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                    }
                }catch(std::exception& ex){
                    std::cout<<"asset."<<phaseName<<"() failed. asset="<<name<<std::endl;
                    std::cout<<ex.what()<<std::endl;
                    std::cout<<"w="<<worker_index<<","<<"v="<<vector_index<<std::endl;
                    success=false;
                    throw ex;
                }
            }
            ++wit;
        }
    }else{
        //並列化する場合、処理できるようになったものから処理を実施
        std::vector<std::size_t> waitings=processed;
        //処理すべきAssetに依存するAssetも処理状況の監視が必要なため待機リストに追加
        auto wit=waitings.begin();
        while(wit!=waitings.end()){
            auto index=*wit;
        for(auto&& dependent:assetDependedOnBy[phase][index]){
            auto dst=std::lower_bound(waitings.begin(),waitings.end(),dependent);
            if(dst==waitings.end() || (*dst)>dependent){
                waitings.insert(dst,dependent);
            }
        }
        wit=std::find(waitings.begin(),waitings.end(),index);
            ++wit;
        }
        //処理状況監視用の変数の初期化
        std::map<std::size_t,std::atomic_bool> futures;//各Assetの処理状況を格納する。終了時にtrueとなる。
        std::map<std::size_t,std::vector<std::size_t>> checklists;//各Assetが依存するAssetのindexリスト。trueになった後に何度も確認してしまわないように要素を削除しながら回す。
        for(wit=waitings.begin();wit!=waitings.end();++wit){
            futures[*wit]=false;
            checklists[*wit]=assetDependsOn[phase][*wit];
        }
        nl::json tmp=waitings;
        tmp=processed;
        //処理を順番に実行する
        wit=waitings.begin();
        while(wit!=waitings.end()){
            std::size_t index=*wit;
        bool ready=true;
        auto dit=checklists[index].begin();
        while(dit!=checklists[index].end()){
            if(futures.find(*dit)==futures.end() || futures[*dit]){
                dit=checklists[index].erase(dit);
            }else{
                ready=false;
                dit=checklists[index].end();
            }
        }
            auto asset=indicedAssets[phase][index].first;
            std::string name=asset->getFullName();
        if(ready){
            if(indicedAssets[phase][index].second==tickCount){
                //この時刻に処理を実施するもの
                    if(asset->isAlive()){
                        int w=worker_index;
                        int v=vector_index;
                        pool.push_task([&futures,&success,phase,index,name,asset,w,v,this]{
                            std::chrono::system_clock::time_point before,after;
                            std::chrono::microseconds difference;
                            try{
                                if(measureTime){
                                    before=std::chrono::system_clock::now();
                                }
                                phaseFunc.at(phase)(asset);
                                if(measureTime){
                                    std::string phaseName=std::string(magic_enum::enum_name(phase));
                                    after=std::chrono::system_clock::now();
                                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                                    maxProcessTime.at(phaseName).at(name)=std::max<std::uint64_t>(maxProcessTime.at(phaseName).at(name),difference.count());
                                    minProcessTime.at(phaseName).at(name)=std::min<std::uint64_t>(minProcessTime.at(phaseName).at(name),difference.count());
                                    processCount.at(phaseName).at(name)++;
                                    meanProcessTime.at(phaseName).at(name)+=((double)(difference.count())-meanProcessTime.at(phaseName).at(name))/processCount.at(phaseName).at(name);
                                }
                            }catch(std::exception& ex){
                                std::cout<<"asset."<<magic_enum::enum_name(phase)<<"() failed. asset="<<name<<std::endl;
                                std::cout<<ex.what()<<std::endl;
                            std::cout<<"w="<<w<<","<<"v="<<v<<std::endl;
                            success=false;
                                throw ex;
                            }
                            futures[index]=true;
                        });
                    }else{
                        //not alive
                        futures[index]=true;
                    }
                }else{
                    //この時刻に処理を実施しないもの
                    futures[index]=true;
                }
                wit=waitings.erase(wit);
            }else{
                ++wit;
            }
            if(wit==waitings.end()){
                wit=waitings.begin();
        }
        if(!success){
            throw std::runtime_error("SimulationManager::runAssetPhaseFunc("+std::string(magic_enum::enum_name(phase))+") failed.");
            }
        }
        pool.wait_for_tasks();
    }
    for(auto&& e:killRequests){
        if(e->isAlive()){
            e->kill();
        }
    }
    //次回処理時刻の算出
    for(auto wit=processed.begin();wit!=processed.end();++wit){
        std::size_t index=*wit;
        auto asset=indicedAssets[phase][index].first;
        if(asset->isAlive()){
            auto next=asset->getNextTick(phase,tickCount);
            assert(next>tickCount);
            orderedAssets[phase][std::make_pair(next,index)]=asset;
            indicedAssets[phase][index]=std::make_pair(asset,next);
        }else{
            orderedAssets[phase][std::make_pair(std::numeric_limits<std::uint64_t>::max(),index)]=asset;
            indicedAssets[phase][index]=std::make_pair(asset,std::numeric_limits<std::uint64_t>::max());
        }
    }
    if(orderedAssets[phase].size()>0){
        nextTick[phase]=(*orderedAssets[phase].begin()).first.first;
    }else{
        nextTick[phase]=std::numeric_limits<std::uint64_t>::max();
    }
}
void SimulationManager::runCallbackPhaseFunc(const SimPhase& phase,bool group1){
    static const std::map<SimPhase,std::function<void(std::shared_ptr<Callback>)>> phaseFunc={
        {SimPhase::ON_GET_OBSERVATION_SPACE,[](std::shared_ptr<Callback> c){c->onGetObservationSpace();}},
        {SimPhase::ON_GET_ACTION_SPACE,[](std::shared_ptr<Callback> c){c->onGetActionSpace();}},
        {SimPhase::ON_MAKE_OBS,[](std::shared_ptr<Callback> c){c->onMakeObs();}},
        {SimPhase::ON_DEPLOY_ACTION,[](std::shared_ptr<Callback> c){c->onDeployAction();}},
        {SimPhase::ON_EPISODE_BEGIN,[](std::shared_ptr<Callback> c){c->onEpisodeBegin();}},
        {SimPhase::ON_VALIDATION_END,[](std::shared_ptr<Callback> c){c->onValidationEnd();}},
        {SimPhase::ON_STEP_BEGIN,[](std::shared_ptr<Callback> c){c->onStepBegin();}},
        {SimPhase::ON_INNERSTEP_BEGIN,[](std::shared_ptr<Callback> c){c->onInnerStepBegin();}},
        {SimPhase::ON_INNERSTEP_END,[](std::shared_ptr<Callback> c){c->onInnerStepEnd();}},
        {SimPhase::ON_STEP_END,[](std::shared_ptr<Callback> c){c->onStepEnd();}},
        {SimPhase::ON_EPISODE_END,[](std::shared_ptr<Callback> c){c->onEpisodeEnd();}}
    };
    assert(phaseFunc.count(phase)>0);
    auto& group = group1 ? orderedAllCallbacks1.at(phase) : orderedAllCallbacks2.at(phase);
    std::chrono::system_clock::time_point before,after;
    std::chrono::microseconds difference;
    killRequests.clear();
    if(phase==SimPhase::ON_GET_OBSERVATION_SPACE ||
        phase==SimPhase::ON_GET_ACTION_SPACE ||
        phase==SimPhase::ON_MAKE_OBS ||
        phase==SimPhase::ON_DEPLOY_ACTION
    ){
        for(auto& cb:group){
            try{
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                phaseFunc.at(phase)(cb.second);
                if(measureTime){
                    std::string phaseName=std::string(magic_enum::enum_name(phase));
                    std::string name=cb.second->getName();
                    after=std::chrono::system_clock::now();
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
            }catch(std::exception& ex){
                std::cout<<"callback."<<magic_enum::enum_name(phase)<<"() failed. callback="<<cb.second->getName()<<std::endl;
                std::cout<<ex.what()<<std::endl;
                std::cout<<"w="<<worker_index<<","<<"v="<<vector_index<<std::endl;
                throw ex;
            }
        }
    }else{
        bool finished=group.size()==0;//can be false
        while(!finished){
            auto cb=*group.begin();
            if(cb.first.first>tickCount){
                finished=true;
            }else{
                group.erase(group.begin());
                try{
                    if(measureTime){
                        before=std::chrono::system_clock::now();
                    }
                    phaseFunc.at(phase)(cb.second);
                    if(measureTime){
                        std::string phaseName=std::string(magic_enum::enum_name(phase));
                        std::string name=cb.second->getName();
                        after=std::chrono::system_clock::now();
                        difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                        maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                        minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                        processCount[phaseName][name]++;
                        meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                    }
                    auto next=cb.second->getNextTick(phase,tickCount);
                    assert(next>tickCount);
                    if(phase==SimPhase::ON_STEP_BEGIN || phase==SimPhase::ON_STEP_END){
                        assert(next%agentInterval==0);
                    }
                    group[std::make_pair(next,cb.first.second)]=cb.second;
                }catch(std::exception& ex){
                    std::cout<<"callback."<<magic_enum::enum_name(phase)<<"() failed. callback="<<cb.second->getName()<<std::endl;
                    std::cout<<ex.what()<<std::endl;
                    std::cout<<"w="<<worker_index<<","<<"v="<<vector_index<<std::endl;
                    finished=true;
                    throw ex;
                }
            }
        }
        if(group.size()>0){
            nextTick[phase]=(*group.begin()).first.first;
        }else{
            nextTick[phase]=std::numeric_limits<std::uint64_t>::max();
        }
    }
    for(auto&& e:killRequests){
        if(e->isAlive()){
            e->kill();
        }
    }
}
void SimulationManager::innerStep(){
    if(nextTickCB<std::min(nextTickP,nextAgentTick)){
        tickCount=nextTickCB;
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_BEGIN,true);
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_BEGIN,false);
        runAssetPhaseFunc(SimPhase::CONTROL);
        runAssetPhaseFunc(SimPhase::BEHAVE);
    }else if(nextTickP<=nextAgentTick){
        tickCount=nextTickP;
        runAssetPhaseFunc(SimPhase::PERCEIVE);
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_END,true);
        scores=ruler->score;
        gatherRewards();
        gatherTotalRewards();
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_END,false);
    }else{
        tickCount=nextAgentTick;
    }
}
py::dict SimulationManager::makeObs(){
    observation=py::dict();
    for(auto& agent:agents){
        try{
            if(agent.second->isAlive()){
                py::object obs=agent.second->makeObs();
                lastObservations[agent.second->getFullName().c_str()]=obs;
                observation[agent.second->getFullName().c_str()]=obs;
            }else if(!dones[agent.second->getFullName()]){
                observation[agent.second->getFullName().c_str()]=lastObservations[agent.second->getFullName().c_str()];
            }
        }catch(std::exception& ex){
            std::cout<<"agent.makeObs() failed. agent="<<agent.second->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            std::cout<<"w="<<worker_index<<","<<"v="<<vector_index<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_MAKE_OBS,true);
    runCallbackPhaseFunc(SimPhase::ON_MAKE_OBS,false);
    for(auto& agent:agents){
        if(observation.contains(agent.second->getFullName())){
            if(isinstance<ExpertWrapper>(agent.second)){
                auto ex=getShared<ExpertWrapper>(agent.second);
                ex->imitatorObs=py::cast<py::tuple>(observation[ex->getFullName().c_str()])[0];
                ex->expertObs=py::cast<py::tuple>(observation[ex->getFullName().c_str()])[1];
                if(ex->whichExpose=="Expert"){
                    observation[ex->getFullName().c_str()]=ex->expertObs;
                }else if(ex->whichExpose=="Imitator"){
                    observation[ex->getFullName().c_str()]=ex->imitatorObs;
                }else{//if(ex->whichExpose=="Both"){
                    //そのまま
                }
            }
        }
    }
    return observation;
}
void SimulationManager::deployAction(const py::dict& action_){
    action=action_;
    runCallbackPhaseFunc(SimPhase::ON_DEPLOY_ACTION,true);
    runCallbackPhaseFunc(SimPhase::ON_DEPLOY_ACTION,false);
    for(auto& agent:agents){
        try{
            if(agent.second->isAlive()){
                if(action.contains(agent.second->getFullName())){
                    agent.second->deploy(action[agent.second->getFullName().c_str()]);
                }else{
                    agent.second->deploy(py::none());
                }
            }
        }catch(std::exception& ex){
            std::cout<<"agent.deploy() failed. agent="<<agent.second->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            std::cout<<"w="<<worker_index<<","<<"v="<<vector_index<<std::endl;
            throw ex;
        }
    }
}
void SimulationManager::gatherDones(){
    for(auto& agent:agents){
        dones[agent.second->getFullName()]=!agent.second->isAlive() || ruler->dones[agent.second->getName()] || manualDone;
    }
    dones["__all__"]=ruler->dones["__all__"] || manualDone;
}
void SimulationManager::gatherRewards(){
    rewards.clear();
    for(auto& agent:agents){
        double val=0.0;
        for(auto& r:rewardGenerators){
            try{
                val+=r->getReward(agent.second);
            }catch(std::exception& ex){
                std::cout<<"reward.getReward(agent) failed. reward="<<r->getName()<<",agent="<<agent.second->getFullName()<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }
        if(exposeDeadAgentReward || (agent.second->isAlive() || !dones[agent.second->getFullName()])){
            rewards[agent.second->getFullName()]=val;
        }
    }
}
void SimulationManager::gatherTotalRewards(){
    for(auto& agent:agents){
        double val=0.0;
        for(auto& r:rewardGenerators){
            try{
                val+=r->getTotalReward(agent.second);
            }catch(std::exception& ex){
                std::cout<<"reward.getTotalReward(agent) failed. reward="<<r->getName()<<",agent="<<agent.second->getFullName()<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }
        totalRewards[agent.second->getFullName()]=val;
    }
}
py::dict SimulationManager::get_observation_space(){
    observation_space=py::dict();
    for(auto& agent:agents){
        try{
            observation_space[agent.second->getFullName().c_str()]=agent.second->observation_space();
        }catch(std::exception& ex){
            std::cout<<"agent.get_observation_space() failed. agent="<<agent.second->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_GET_OBSERVATION_SPACE,true);
    runCallbackPhaseFunc(SimPhase::ON_GET_OBSERVATION_SPACE,false);
    for(auto& agent:agents){
        if(isinstance<ExpertWrapper>(agent.second)){
            auto ex=getShared<ExpertWrapper>(agent.second);
            if(ex->whichExpose=="Imitator"){
                observation_space[ex->getFullName().c_str()]=py::cast<py::tuple>(observation_space[ex->getFullName().c_str()])[0];
            }else if(ex->whichExpose=="Expert"){
                observation_space[ex->getFullName().c_str()]=py::cast<py::tuple>(observation_space[ex->getFullName().c_str()])[1];
            }else{//if(ex->whichExpose=="Both"){
                //そのまま
            }
        }
    }
    return observation_space;
}
py::dict SimulationManager::get_action_space(){
    action_space=py::dict();
    for(auto& agent:agents){
        try{
            action_space[agent.second->getFullName().c_str()]=agent.second->action_space();
        }catch(std::exception& ex){
            std::cout<<"agent.get_action_space() failed. agent="<<agent.second->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_GET_ACTION_SPACE,true);
    runCallbackPhaseFunc(SimPhase::ON_GET_ACTION_SPACE,false);
    for(auto& agent:agents){
        if(isinstance<ExpertWrapper>(agent.second)){
            auto ex=getShared<ExpertWrapper>(agent.second);
            if(ex->whichExpose=="Imitator"){
                action_space[ex->getFullName().c_str()]=py::cast<py::tuple>(action_space[ex->getFullName().c_str()])[0];
            }else if(ex->whichExpose=="Expert"){
                action_space[ex->getFullName().c_str()]=py::cast<py::tuple>(action_space[ex->getFullName().c_str()])[1];
            }else{//if(ex->whichExpose=="Both"){
                //そのまま
            }
        }
    }
    return action_space;
}
void SimulationManager::addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler){
    eventHandlers[name].push_back(handler);
}
void SimulationManager::triggerEvent(const std::string& name, const nl::json& args){
    if(eventHandlers.count(name)>0){
        for(auto& e:eventHandlers[name]){
            try{
                e(args);
            }catch(std::exception& ex){
                std::cout<<"triggerEvent(args) failed. name="<<name<<", args="<<args<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }
    }else{
        //std::cout<<"no handler"<<std::endl;
    }
}
std::weak_ptr<Agent> SimulationManager::generateAgent(const nl::json& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents){
    nl::json instanceConfig=nl::json::object();
    try{
        if(agentConfig.contains("instanceConfig")){
            instanceConfig=agentConfig.at("instanceConfig");
        }
        auto mAcc=SimulationManagerAccessorForAgent::create(this->shared_from_this());
        auto dep=DependencyChecker::create(this->shared_from_this());
        instanceConfig["manager"]=mAcc;
        instanceConfig["baseTimeStep"]=baseTimeStep;
        instanceConfig["dependencyChecker"]=dep;
        instanceConfig["name"]=agentName;
        instanceConfig["parents"]=parents;
        instanceConfig["seed"]=randomGen();
        std::string type=agentConfig.at("type");
        std::string modelName;
        instanceConfig["type"]=type;
        if(agentConfig.contains("model")){
            modelName=instanceConfig["model"]=agentConfig.at("model");
        }else{
            assert(type=="ExpertE" || type=="ExpertI" || type=="ExpertBE" || type=="ExpertBI");
            instanceConfig["model"]=agentConfig.at("expertModel");
            modelName=type;
        }
        if(agentConfig.contains("policy")){
            instanceConfig["policy"]=agentConfig.at("policy");
        }else{
            instanceConfig["policy"]="Internal";
        }
        if(type=="Internal"){
            instanceConfig["policy"]="Internal";
        }else if(type=="External"){
            //nothing special
        }else if(type=="ExpertE" || type=="ExpertI" || type=="ExpertBE" || type=="ExpertBI"){
            instanceConfig["imitatorModelName"]=agentConfig.at("imitatorModel");
            instanceConfig["expertModelName"]=agentConfig.at("expertModel");
            instanceConfig["model"]=instanceConfig.at("expertModelName");
            if(agentConfig.contains("expertPolicy")){
                //external expert
                instanceConfig["expertPolicyName"]=agentConfig.at("expertPolicy");
                instanceConfig["policy"]=agentConfig.at("expertPolicy");
            }else{
                //internal expert
                instanceConfig["expertPolicyName"]=instanceConfig.at("policy");
                instanceConfig["policy"]="Internal";
            }
            if(agentConfig.contains("identifier")){
                instanceConfig["identifier"]=agentConfig.at("identifier");
            }else{
                instanceConfig["identifier"]=instanceConfig.at("imitatorModelName");
            }
        }
        std::shared_ptr<Agent>agent=factory.create<Agent>("Agent",modelName,instanceConfig);
        agents[agentName]=agent;
        return agent;
    }catch(std::exception& ex){
        std::cout<<"generateAgent(agentConfig,agentName,parents) failed."<<std::endl;
        std::cout<<"agentConfig="<<agentConfig<<std::endl;
        std::cout<<"agentName="<<agentName<<std::endl;
        std::cout<<"parents={"<<std::endl;
        for(auto&& p:parents){
            std::cout<<"  "<<p.first<<":"<<p.second->getFullName()<<","<<std::endl;
        }
        std::cout<<"}"<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
}
std::weak_ptr<Asset> SimulationManager::generateAsset(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig_){
    std::shared_ptr<Asset> asset;
    try{
        nl::json instanceConfig=instanceConfig_;
        auto mAcc=SimulationManagerAccessorForPhysicalAsset::create(this->shared_from_this());
        auto dep=DependencyChecker::create(this->shared_from_this());
        instanceConfig["manager"]=mAcc;
        instanceConfig["baseTimeStep"]=baseTimeStep;
        instanceConfig["dependencyChecker"]=dep;
        asset=factory.create<Asset>(baseName,modelName,instanceConfig);
    }catch(std::exception& ex){
        std::cout<<"creation of Asset failed. baseName="<<baseName<<", modelName="<<modelName<<std::endl;
        std::cout<<"instanceConfig_="<<instanceConfig_<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    if(isinstance<PhysicalAsset>(asset)){
        assets[asset->getFullName()]=std::dynamic_pointer_cast<PhysicalAsset>(asset);
    }else if(isinstance<Controller>(asset)){
        controllers[asset->getFullName()]=std::dynamic_pointer_cast<Controller>(asset);
    }else{
        throw std::runtime_error("Created Asset is neither PhysicalAsset nor Controller. fullName="+asset->getFullName()+", type="+typeid(asset).name());
    }
    try{
        asset->makeChildren();
    }catch(std::exception& ex){
        std::cout<<"asset.makeChildren() failed. asset="<<asset->getFullName()<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    return asset;
}
std::weak_ptr<Asset> SimulationManager::generateAssetByClassName(const std::string& baseName,const std::string& className,const nl::json& modelConfig_,const nl::json& instanceConfig_){
    std::shared_ptr<Asset> asset;
    try{
        nl::json instanceConfig=instanceConfig_;
        auto mAcc=SimulationManagerAccessorForPhysicalAsset::create(this->shared_from_this());
        auto dep=DependencyChecker::create(this->shared_from_this());
        instanceConfig["manager"]=mAcc;
        instanceConfig["baseTimeStep"]=baseTimeStep;
        instanceConfig["dependencyChecker"]=dep;
        asset=factory.createByClassName<Asset>(baseName,className,modelConfig_,instanceConfig);
    }catch(std::exception& ex){
        std::cout<<"creation of Asset failed. baseName="<<baseName<<", className="<<className<<std::endl;
        std::cout<<"modelConfig_="<<modelConfig_<<"instanceConfig_="<<instanceConfig_<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    if(isinstance<PhysicalAsset>(asset)){
        assets[asset->getFullName()]=std::dynamic_pointer_cast<PhysicalAsset>(asset);
    }else if(isinstance<Controller>(asset)){
        controllers[asset->getFullName()]=std::dynamic_pointer_cast<Controller>(asset);
    }else{
        throw std::runtime_error("Created Asset is neither PhysicalAsset nor Controller. fullName="+asset->getFullName()+", type="+typeid(asset).name());
    }
    try{
        asset->makeChildren();
    }catch(std::exception& ex){
        std::cout<<"asset.makeChildren() failed. asset="<<asset->getFullName()<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    return asset;
}
bool SimulationManager::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    if(communicationBuffers.count(name_)==0){
        communicationBuffers[name_]=CommunicationBuffer::create(this->shared_from_this(),name_,participants_,inviteOnRequest_);
        return true;
    }else{
        return false;
    }
}
bool SimulationManager::requestInvitationToCommunicationBuffer(const std::string& bufferName,std::shared_ptr<Asset> asset){
    if(communicationBuffers.count(bufferName)>0){
        return communicationBuffers[bufferName]->requestInvitation(asset);
    }else{
        return false;
    }
}
void SimulationManager::generateAssets(){
    assetConfigDispatcher.reset();
    agentConfigDispatcher.reset();
    assets.clear();
    agents.clear();
    controllers.clear();
    teams.clear();
    numExperts=0;
    experts.clear();
    nl::json agentBuffer=nl::json::object();
    std::map<std::string,nl::json> assetConfigs;
    try{
        assetConfigs=RecursiveJsonExtractor::run(
            assetConfigDispatcher.run(managerConfig.at("Assets")),
            [](const nl::json& node){
                if(node.is_object()){
                    if(node.contains("type")){
                        return node["type"]!="group" && node["type"]!="broadcast";
                    }
                }
                return false;
            }
        );
    }catch(std::exception& ex){
        std::cout<<"dispatch of asset config failed."<<std::endl;
        if(managerConfig.is_object() && managerConfig.contains("Assets")){
            std::cout<<"config="<<managerConfig.at("Assets")<<std::endl;
        }else{
            std::cout<<"config doesn't have 'Assets' as a key."<<std::endl;
        }
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    std::map<std::string,int> dummy;
    for(auto& e:assetConfigs){
        try{
            std::string assetName=e.first.substr(1);//remove "/" at head.
            std::string assetType=e.second.at("type");
            nl::json modelConfig=e.second.at("model");
            nl::json instanceConfig=e.second.at("instanceConfig");
            auto mAcc=SimulationManagerAccessorForPhysicalAsset::create(this->shared_from_this());
            auto dep=DependencyChecker::create(this->shared_from_this());
            instanceConfig.merge_patch({
                {"manager",mAcc},
                {"dependencyChecker",dep},
                {"fullName",assetName},
                {"seed",randomGen()}
            });
            std::shared_ptr<Asset> asset=generateAsset(assetType,modelConfig,instanceConfig).lock();
            dummy[asset->getTeam()]=0;
            if(e.second.contains("Agent")){
                nl::json agentConfig=agentConfigDispatcher.run(e.second["Agent"]);
                std::string agentName=agentConfig["name"];
                std::string agentPort;
                if(agentConfig.contains("port")){
                    agentPort=agentConfig["port"];
                }else{
                    agentPort="0";
                }
                if(agentBuffer.contains(agentName)){
                    if(agentBuffer[agentName]["parents"].contains(agentPort)){
                        throw std::runtime_error("Duplicated agent port: agent="+agentName+", port="+agentPort);
                    }else{
                        agentBuffer[agentName]["parents"][agentPort]=asset->getAccessor();
                    }
                }else{
                    agentBuffer[agentName]={
                        {"config",agentConfig},
                        {"parents",{{agentPort,asset->getAccessor()}}}
                    };
                }
            }
        }catch(std::exception& ex){
            std::cout<<"creation of assets failed."<<std::endl;
            std::cout<<"assetConfig={"<<e.first<<":"<<e.second<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    for(auto& e:dummy){
        teams.push_back(e.first);
    }
    for(auto& e:agentBuffer.items()){
        std::string agentName=e.key();
        std::shared_ptr<Agent> agent=generateAgent(e.value()["config"],agentName,e.value()["parents"]).lock();
        for(auto& p:agent->parents){
            p.second->asset.lock()->setAgent(agent,p.first);
        }
        if(agent->type=="ExpertE" || agent->type=="ExpertI" || agent->type=="ExpertBE" || agent->type=="ExpertBI"){
            numExperts++;
            experts[agentName]=agent;
        }
    }
    checkOrder();
}
void SimulationManager::generateCommunicationBuffers(){
    if(!managerConfig.contains("CommunicationBuffers")){
        return;
    }
    assert(managerConfig.at("CommunicationBuffers").is_object());
    for(auto&& e:managerConfig.at("CommunicationBuffers").items()){
        try{
            nl::json participants=e.value().contains("participants") ? e.value().at("participants") : nl::json::array();
            nl::json inviteOnRequest=e.value().contains("inviteOnRequest") ? e.value().at("inviteOnRequest") : nl::json::array();
            generateCommunicationBuffer(e.key(),participants,inviteOnRequest);
        }catch(std::exception& ex){
            std::cout<<"generateCommunicationBuffer() failed."<<std::endl;
            std::cout<<"name="<<e.key()<<std::endl;
            std::cout<<"participants="<<e.value()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
}
void SimulationManager::requestToKillAsset(std::shared_ptr<Asset> asset){
    killRequests.push_back(asset);
}
double SimulationManager::getTime() const{
    return baseTimeStep*tickCount;
}
std::uint64_t SimulationManager::getTickCount() const{
    return tickCount;
}
std::uint64_t SimulationManager::getAgentInterval() const{
    return agentInterval;
}
std::uint64_t SimulationManager::getStepCount() const{
    return tickCount/agentInterval;
}
std::vector<std::string> SimulationManager::getTeams() const{
    return teams;
}
nl::json SimulationManager::getManagerConfig() const{
    return managerConfig;
}
nl::json SimulationManager::getFactoryModelConfig() const{
    return factory.getModelConfig();
}
nl::json SimulationManager::to_json_ref(){
    nl::json ret=this->shared_from_this();
    return std::move(ret);
}

DependencyChecker::DependencyChecker(std::shared_ptr<SimulationManager> manager_){
    manager=manager_;
}
DependencyChecker::~DependencyChecker(){}
void DependencyChecker::clearReadiness(){
    for(auto& e:manager.lock()->assetPhases){
        readiness[e]=false;
    }
}
bool DependencyChecker::checkReadiness(const SimPhase &phase){
    if(readiness[phase]){
        return true;
    }
    for(auto&& e:dependency[phase]){
        if(e.expired()){
            throw std::runtime_error("One of dependency is expired!");
        }
        if(!e.lock()->dependencyChecker->readiness[phase]){
            return false;
        }
    }
    readiness[phase]=true;
    return true;
}
void DependencyChecker::clearDependency(){
    for(auto& e:manager.lock()->assetPhases){
        dependency[e].clear();
    }
}
void DependencyChecker::addDependency(const SimPhase& phase,const std::string& fullName_){
    dependency[phase].push_back(manager.lock()->getAsset(fullName_));
}
void DependencyChecker::addDependency(const SimPhase& phase,std::shared_ptr<Asset> asset){
    dependency[phase].push_back(asset);
}
nl::json DependencyChecker::to_json_ref(){
    return this->shared_from_this();
}
std::shared_ptr<DependencyChecker> DependencyChecker::create(std::shared_ptr<SimulationManager> manager_){
    return std::make_shared<DependencyChecker>(manager_);
}
std::shared_ptr<DependencyChecker> DependencyChecker::create(std::shared_ptr<DependencyChecker> checker_){
    return std::make_shared<DependencyChecker>(checker_->manager.lock());
}
std::shared_ptr<DependencyChecker> DependencyChecker::from_json_ref(const nl::json& j){
    return j;
}
std::weak_ptr<DependencyChecker> DependencyChecker::from_json_weakref(const nl::json& j){
    return j;
}

bool SimulationManagerAccessorBase::expired() const noexcept{
    return manager.expired();
}
double SimulationManagerAccessorBase::getTime() const{
    return manager.lock()->getTime();
}
double SimulationManagerAccessorBase::getBaseTimeStep() const{
    return manager.lock()->baseTimeStep;
}
std::uint64_t SimulationManagerAccessorBase::getTickCount() const{
    return manager.lock()->tickCount;
}
std::uint64_t SimulationManagerAccessorBase::getAgentInterval() const{
    return manager.lock()->agentInterval;
}
std::uint64_t SimulationManagerAccessorBase::getStepCount() const{
    return manager.lock()->getStepCount();
}
std::vector<std::string> SimulationManagerAccessorBase::getTeams() const{
    return std::move(manager.lock()->getTeams());
}
SimulationManagerAccessorBase::SimulationManagerAccessorBase(std::shared_ptr<SimulationManager> manager_):manager(manager_){}
SimulationManagerAccessorBase::~SimulationManagerAccessorBase(){}
nl::json SimulationManagerAccessorBase::to_json_ref(){
    return this->shared_from_this();
}
std::shared_ptr<SimulationManagerAccessorBase> SimulationManagerAccessorBase::from_json_ref(const nl::json& j){
    return j;
}
std::weak_ptr<SimulationManagerAccessorBase> SimulationManagerAccessorBase::from_json_weakref(const nl::json& j){
    return j;
}
std::shared_ptr<SimulationManagerAccessorBase> SimulationManagerAccessorBase::create(std::shared_ptr<SimulationManager> manager_){
    return std::make_shared<SimulationManagerAccessorBase>(manager_);
}
std::shared_ptr<SimulationManagerAccessorBase> SimulationManagerAccessorBase::create(std::shared_ptr<SimulationManagerAccessorBase> original_){
    return std::make_shared<SimulationManagerAccessorBase>(original_->manager.lock());
}
std::shared_ptr<SimulationManagerAccessorForCallback> SimulationManagerAccessorForCallback::copy(){
    return SimulationManagerAccessorForCallback::create(manager.lock());
}
void SimulationManagerAccessorForCallback::requestReconfigure(const nl::json& managerReplacer,const nl::json& factoryReplacer){
    manager.lock()->requestReconfigure(managerReplacer,factoryReplacer);
}
void SimulationManagerAccessorForCallback::addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler){
    manager.lock()->addEventHandler(name,handler);
}
void SimulationManagerAccessorForCallback::triggerEvent(const std::string& name, const nl::json& args){
    manager.lock()->triggerEvent(name,args);
}
void SimulationManagerAccessorForCallback::requestToKillAsset(std::shared_ptr<Asset> asset){
    manager.lock()->requestToKillAsset(asset);
}
py::dict& SimulationManagerAccessorForCallback::observation_space() const{
    return manager.lock()->observation_space;
}
py::dict& SimulationManagerAccessorForCallback::action_space() const{
    return manager.lock()->action_space;
}
py::dict& SimulationManagerAccessorForCallback::observation() const{
    return manager.lock()->observation;
}
py::dict& SimulationManagerAccessorForCallback::action() const{
    return manager.lock()->action;
}
std::map<std::string,bool>& SimulationManagerAccessorForCallback::dones() const{
    return manager.lock()->dones;
}
std::map<std::string,double>& SimulationManagerAccessorForCallback::scores() const{
    return manager.lock()->scores;
}
std::map<std::string,double>& SimulationManagerAccessorForCallback::rewards() const{
    return manager.lock()->rewards;
}
std::map<std::string,double>& SimulationManagerAccessorForCallback::totalRewards() const{
    return manager.lock()->totalRewards;
}
const std::map<std::string,std::weak_ptr<Agent>>& SimulationManagerAccessorForCallback::experts() const{
    return manager.lock()->experts;
}
int SimulationManagerAccessorForCallback::worker_index() const{
    return manager.lock()->worker_index;
}
int SimulationManagerAccessorForCallback::vector_index() const{
    return manager.lock()->vector_index;
}
bool& SimulationManagerAccessorForCallback::manualDone(){
    return manager.lock()->manualDone;
}
void SimulationManagerAccessorForCallback::setManualDone(const bool& b){
    manager.lock()->manualDone=b;
}
nl::json SimulationManagerAccessorForCallback::getManagerConfig() const{
    return manager.lock()->getManagerConfig();
}
nl::json SimulationManagerAccessorForCallback::getFactoryModelConfig() const{
    return manager.lock()->getFactoryModelConfig();
}
nl::json SimulationManagerAccessorForCallback::to_json_ref(){
    return this->shared_from_this();
}
std::shared_ptr<SimulationManagerAccessorForCallback> SimulationManagerAccessorForCallback::from_json_ref(const nl::json& j){
    return j;
}
std::weak_ptr<SimulationManagerAccessorForCallback> SimulationManagerAccessorForCallback::from_json_weakref(const nl::json& j){
    return j;
}
SimulationManagerAccessorForCallback::SimulationManagerAccessorForCallback(std::shared_ptr<SimulationManager> manager_):SimulationManagerAccessorBase(manager_){}
SimulationManagerAccessorForCallback::~SimulationManagerAccessorForCallback(){}
std::shared_ptr<SimulationManagerAccessorForCallback> SimulationManagerAccessorForCallback::create(std::shared_ptr<SimulationManager> manager_){
    return std::make_shared<SimulationManagerAccessorForCallback>(manager_);
}
std::shared_ptr<SimulationManagerAccessorForCallback> SimulationManagerAccessorForCallback::create(std::shared_ptr<SimulationManagerAccessorForCallback> original_){
    return std::make_shared<SimulationManagerAccessorForCallback>(original_->manager.lock());
}
std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> SimulationManagerAccessorForPhysicalAsset::copy(){
    return SimulationManagerAccessorForPhysicalAsset::create(manager.lock());
}
void SimulationManagerAccessorForPhysicalAsset::triggerEvent(const std::string& name, const nl::json& args){
    manager.lock()->triggerEvent(name,args);
}
void SimulationManagerAccessorForPhysicalAsset::requestToKillAsset(std::shared_ptr<Asset> asset){
    manager.lock()->requestToKillAsset(asset);
}
bool SimulationManagerAccessorForPhysicalAsset::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    return manager.lock()->generateCommunicationBuffer(name_,participants_,inviteOnRequest_);
}
bool SimulationManagerAccessorForPhysicalAsset::requestInvitationToCommunicationBuffer(const std::string& bufferName,std::shared_ptr<Asset> asset){
    return manager.lock()->requestInvitationToCommunicationBuffer(bufferName,asset);
}
nl::json SimulationManagerAccessorForPhysicalAsset::to_json_ref(){
    return this->shared_from_this();
}
std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> SimulationManagerAccessorForPhysicalAsset::from_json_ref(const nl::json& j){
    return j;
}
std::weak_ptr<SimulationManagerAccessorForPhysicalAsset> SimulationManagerAccessorForPhysicalAsset::from_json_weakref(const nl::json& j){
    return j;
}
SimulationManagerAccessorForPhysicalAsset::SimulationManagerAccessorForPhysicalAsset(std::shared_ptr<SimulationManager> manager_):SimulationManagerAccessorBase(manager_){}
SimulationManagerAccessorForPhysicalAsset::~SimulationManagerAccessorForPhysicalAsset(){}
std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> SimulationManagerAccessorForPhysicalAsset::create(std::shared_ptr<SimulationManager> manager_){
    return std::make_shared<SimulationManagerAccessorForPhysicalAsset>(manager_);
}
std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> SimulationManagerAccessorForPhysicalAsset::create(std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> original_){
    return std::make_shared<SimulationManagerAccessorForPhysicalAsset>(original_->manager.lock());
}
std::shared_ptr<SimulationManagerAccessorForAgent> SimulationManagerAccessorForAgent::copy(){
    return SimulationManagerAccessorForAgent::create(manager.lock());
}
bool SimulationManagerAccessorForAgent::requestInvitationToCommunicationBuffer(const std::string& bufferName,std::shared_ptr<Asset> asset){
    return manager.lock()->requestInvitationToCommunicationBuffer(bufferName,asset);
}
std::weak_ptr<RulerAccessor> SimulationManagerAccessorForAgent::getRuler() const{
    return manager.lock()->ruler->getAccessor();
}
nl::json SimulationManagerAccessorForAgent::to_json_ref(){
    return this->shared_from_this();
}
std::shared_ptr<SimulationManagerAccessorForAgent> SimulationManagerAccessorForAgent::from_json_ref(const nl::json& j){
    return j;
}
std::weak_ptr<SimulationManagerAccessorForAgent> SimulationManagerAccessorForAgent::from_json_weakref(const nl::json& j){
    return j;
}
SimulationManagerAccessorForAgent::SimulationManagerAccessorForAgent(std::shared_ptr<SimulationManager> manager_):SimulationManagerAccessorBase(manager_){}
SimulationManagerAccessorForAgent::~SimulationManagerAccessorForAgent(){}
std::shared_ptr<SimulationManagerAccessorForAgent> SimulationManagerAccessorForAgent::create(std::shared_ptr<SimulationManager> manager_){
    return std::make_shared<SimulationManagerAccessorForAgent>(manager_);
}
std::shared_ptr<SimulationManagerAccessorForAgent> SimulationManagerAccessorForAgent::create(std::shared_ptr<SimulationManagerAccessorForAgent> original_){
    return std::make_shared<SimulationManagerAccessorForAgent>(original_->manager.lock());
}

void exportSimulationManager(py::module &m)
{
    using namespace pybind11::literals;
    py::enum_<SimPhase>(m,"SimPhase")
    .value("VALIDATE",SimPhase::VALIDATE)
    .value("PERCEIVE",SimPhase::PERCEIVE)
    .value("CONTROL",SimPhase::CONTROL)
    .value("BEHAVE",SimPhase::BEHAVE)
    .value("ON_GET_OBSERVATION_SPACE",SimPhase::ON_GET_OBSERVATION_SPACE)
    .value("ON_GET_ACTION_SPACE",SimPhase::ON_GET_ACTION_SPACE)
    .value("ON_MAKE_OBS",SimPhase::ON_MAKE_OBS)
    .value("ON_EPISODE_BEGIN",SimPhase::ON_EPISODE_BEGIN)
    .value("ON_VALIDATION_END",SimPhase::ON_VALIDATION_END)
    .value("ON_STEP_BEGIN",SimPhase::ON_STEP_BEGIN)
    .value("ON_INNERSTEP_BEGIN",SimPhase::ON_INNERSTEP_BEGIN)
    .value("ON_INNERSTEP_END",SimPhase::ON_INNERSTEP_END)
    .value("ON_STEP_END",SimPhase::ON_STEP_END)
    .value("ON_EPISODE_END",SimPhase::ON_EPISODE_END)
    ;
    py::class_<MapIterable<PhysicalAsset,PhysicalAsset>>(m,"MapIterable<PhysicalAsset>")
    .def("__iter__",&MapIterable<PhysicalAsset,PhysicalAsset>::iter,py::keep_alive<0,1>())
    .def("__next__",[](MapIterable<PhysicalAsset,PhysicalAsset>& v){return v.next().lock()->restoreOriginalClassForPY(true);})
    ;
    py::class_<MapIterable<Controller,Controller>>(m,"MapIterable<Controller>")
    .def("__iter__",&MapIterable<Controller,Controller>::iter,py::keep_alive<0,1>())
    .def("__next__",[](MapIterable<Controller,Controller>& v){return v.next().lock()->restoreOriginalClassForPY(true);})
    ;
    py::class_<MapIterable<Agent,Agent>>(m,"MapIterable<Agent>")
    .def("__iter__",&MapIterable<Agent,Agent>::iter,py::keep_alive<0,1>())
    .def("__next__",[](MapIterable<Agent,Agent>& v){return v.next().lock()->restoreOriginalClassForPY(true);})
    ;
    py::class_<VectorIterable<Reward,Reward>>(m,"VectorIterable<Reward>")
    .def("__iter__",&VectorIterable<Reward,Reward>::iter,py::keep_alive<0,1>())
    .def("__next__",[](VectorIterable<Reward,Reward>& v){return v.next().lock()->restoreOriginalClassForPY(true);})
    ;
    py::class_<MapIterable<Callback,Callback>>(m,"VectorIterable<Callback>")
    .def("__iter__",&MapIterable<Callback,Callback>::iter,py::keep_alive<0,1>())
    .def("__next__",[](MapIterable<Callback,Callback>& v){return v.next().lock()->restoreOriginalClassForPY(true);})
    ;
    py::class_<SimulationManager,std::shared_ptr<SimulationManager>,SimulationManagerWrap<>>(m,"SimulationManager")
    .def(py::init([](const py::object& config_,int worker_index_,int vector_index_,const py::object& overrider_){
        if(overrider_.is_none()){
            return SimulationManager::create<SimulationManagerWrap<>>(config_,worker_index_,vector_index_);
        }else{
            std::function<nl::json(const nl::json&,int,int)> overrider=[&overrider_](const nl::json& c,int w,int v){
                auto py_overrider=py::cast<std::function<py::object(const py::object&,int,int)>>(overrider_);
                return py_overrider(c.get<py::object>(),w,v);
            };
            return SimulationManager::create<SimulationManagerWrap<>>(config_,worker_index_,vector_index_,overrider);
        }
    }),"config_"_a,"worker_index_"_a=0,"vector_index"_a=0,"overrider"_a=py::none())
    .def_static("parseConfig",[](const py::object& config_){return SimulationManager::parseConfig(config_);})
    DEF_FUNC(SimulationManager,get_observation_space)
    DEF_FUNC(SimulationManager,get_action_space)
    DEF_FUNC(SimulationManager,seed)
    DEF_FUNC(SimulationManager,reset)
    DEF_FUNC(SimulationManager,step)
    DEF_FUNC(SimulationManager,stopEpisodeExternally)
    DEF_FUNC(SimulationManager,getTime)
    DEF_FUNC(SimulationManager,getTickCount)
    DEF_FUNC(SimulationManager,getAgentInterval)
    DEF_FUNC(SimulationManager,getStepCount)
    DEF_FUNC(SimulationManager,getTeams)
    DEF_READONLY(SimulationManager,dones)
    DEF_READONLY(SimulationManager,scores)
    DEF_READONLY(SimulationManager,rewards)
    DEF_READONLY(SimulationManager,totalRewards)
    DEF_READONLY(SimulationManager,experts)
    DEF_READWRITE(SimulationManager,manualDone)
    .def("getAsset",[](const SimulationManager& v,const std::string& fullName_){return v.getAsset(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAssets",py::overload_cast<>(&SimulationManager::getAssets<>,py::const_))
    .def("getAssets",py::overload_cast<MapIterable<PhysicalAsset,PhysicalAsset>::MatcherType>(&SimulationManager::getAssets<>,py::const_),py::keep_alive<0,2>())
    .def("getAgent",[](const SimulationManager& v,const std::string& fullName_){return v.getAgent(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAgents",py::overload_cast<>(&SimulationManager::getAgents<>,py::const_))
    .def("getAgents",py::overload_cast<MapIterable<Agent,Agent>::MatcherType>(&SimulationManager::getAgents<>,py::const_),py::keep_alive<0,2>())
    .def("getController",[](const SimulationManager& v,const std::string& fullName_){return v.getController(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getControllers",py::overload_cast<>(&SimulationManager::getControllers<>,py::const_))
    .def("getControllers",py::overload_cast<MapIterable<Controller,Controller>::MatcherType>(&SimulationManager::getControllers<>,py::const_),py::keep_alive<0,2>())
    .def("getRuler",[](SimulationManager& v){return v.getRuler().lock()->restoreOriginalClassForPY(true);})
    .def("getViewer",[](SimulationManager& v){return v.getViewer().lock()->restoreOriginalClassForPY(true);})
    .def("getRewardGenerator",[](const SimulationManager& v,const int& idx){return v.getRewardGenerator(idx).lock()->restoreOriginalClassForPY(true);})
    .def("getRewardGenerators",py::overload_cast<>(&SimulationManager::getRewardGenerators<>,py::const_))
    .def("getRewardGenerators",py::overload_cast<VectorIterable<Reward,Reward>::MatcherType>(&SimulationManager::getRewardGenerators<>,py::const_),py::keep_alive<0,2>())
    .def("getCallback",[](const SimulationManager& v,const std::string& name_){return v.getCallback(name_).lock()->restoreOriginalClassForPY(true);})
    .def("getCallbacks",py::overload_cast<>(&SimulationManager::getCallbacks<>,py::const_))
    .def("getCallbacks",py::overload_cast<MapIterable<Callback,Callback>::MatcherType>(&SimulationManager::getCallbacks<>,py::const_),py::keep_alive<0,2>())
    .def("getLogger",[](const SimulationManager& v,const std::string& name_){return v.getLogger(name_).lock()->restoreOriginalClassForPY(true);})
    .def("getLoggers",py::overload_cast<>(&SimulationManager::getLoggers<>,py::const_))
    .def("getLoggers",py::overload_cast<MapIterable<Callback,Callback>::MatcherType>(&SimulationManager::getLoggers<>,py::const_),py::keep_alive<0,2>())
    DEF_FUNC(SimulationManager,getManagerConfig)
    DEF_FUNC(SimulationManager,getFactoryModelConfig)
    DEF_FUNC(SimulationManager,setViewerType)
    DEF_FUNC(SimulationManager,requestReconfigure)
    .def("requestReconfigure",[](SimulationManager& v,const py::object& managerReplacer,const py::object& factoryReplacer){v.requestReconfigure(managerReplacer,factoryReplacer);})
    DEF_FUNC(SimulationManager,to_json_ref)
    DEF_FUNC(SimulationManager,printOrderedAssets)
    DEF_READONLY(SimulationManager,worker_index)
    DEF_READONLY(SimulationManager,vector_index)
    DEF_READONLY(SimulationManager,observation_space)
    DEF_READONLY(SimulationManager,action_space)
    ;
    py::class_<DependencyChecker,std::shared_ptr<DependencyChecker>>(m,"DependencyChecker")
    .def(py::init<>(py::overload_cast<std::shared_ptr<SimulationManager>>(&DependencyChecker::create)))
    .def(py::init<>(py::overload_cast<std::shared_ptr<DependencyChecker>>(&DependencyChecker::create)))
    .def("addDependency",py::overload_cast<const SimPhase&,const std::string&>(&DependencyChecker::addDependency))
    .def("addDependency",py::overload_cast<const SimPhase&,std::shared_ptr<Asset>>(&DependencyChecker::addDependency))
    DEF_FUNC(DependencyChecker,to_json_ref)
    DEF_STATIC_FUNC(DependencyChecker,from_json_ref)
    .def_static("from_json_ref",[](const py::object& obj){return DependencyChecker::from_json_ref(obj);})
    DEF_STATIC_FUNC(DependencyChecker,from_json_weakref)
    .def_static("from_json_weakref",[](const py::object& obj){return DependencyChecker::from_json_weakref(obj);})
    ;
    py::class_<SimulationManagerAccessorBase,std::shared_ptr<SimulationManagerAccessorBase>>(m,"SimulationManagerAccessorBase")
    DEF_FUNC(SimulationManagerAccessorBase,expired)
    DEF_FUNC(SimulationManagerAccessorBase,getTime)
    DEF_FUNC(SimulationManagerAccessorBase,getBaseTimeStep)
    DEF_FUNC(SimulationManagerAccessorBase,getTickCount)
    DEF_FUNC(SimulationManagerAccessorBase,getAgentInterval)
    DEF_FUNC(SimulationManagerAccessorBase,getStepCount)
    DEF_FUNC(SimulationManagerAccessorBase,getTeams)
    ;
    py::class_<SimulationManagerAccessorForCallback,SimulationManagerAccessorBase,std::shared_ptr<SimulationManagerAccessorForCallback>>(m,"SimulationManagerAccessorForCallback")
    DEF_FUNC(SimulationManagerAccessorForCallback,copy)
    DEF_FUNC(SimulationManagerAccessorForCallback,requestReconfigure)
    .def("requestReconfigure",[](SimulationManagerAccessorForCallback& v,const py::object& managerReplacer,const py::object& factoryReplacer){
        v.requestReconfigure(managerReplacer,factoryReplacer);
    })
    DEF_FUNC(SimulationManagerAccessorForCallback,addEventHandler)
    .def("addEventHandler",[](SimulationManagerAccessorForCallback& v,const std::string& name,std::function<void(const py::object&)> handler){
        v.addEventHandler(name,handler);
    })
    DEF_FUNC(SimulationManagerAccessorForCallback,triggerEvent)
    .def("triggerEvent",[](SimulationManagerAccessorForCallback& v,const std::string& name, const py::object& args){
        v.triggerEvent(name,args);
    })
    .def_property("observation_space",&SimulationManagerAccessorForCallback::observation_space,&SimulationManagerAccessorForCallback::observation_space)
    .def_property("action_space",&SimulationManagerAccessorForCallback::action_space,&SimulationManagerAccessorForCallback::action_space)
    .def_property("observation",&SimulationManagerAccessorForCallback::observation,&SimulationManagerAccessorForCallback::observation)
    .def_property("action",&SimulationManagerAccessorForCallback::action,&SimulationManagerAccessorForCallback::action)
    .def_property("dones",&SimulationManagerAccessorForCallback::dones,&SimulationManagerAccessorForCallback::dones)
    .def_property("scores",&SimulationManagerAccessorForCallback::scores,&SimulationManagerAccessorForCallback::scores)
    .def_property("rewards",&SimulationManagerAccessorForCallback::rewards,&SimulationManagerAccessorForCallback::rewards)
    .def_property("totalRewards",&SimulationManagerAccessorForCallback::totalRewards,&SimulationManagerAccessorForCallback::totalRewards)
    .def_property_readonly("experts",&SimulationManagerAccessorForCallback::experts)
    .def_property_readonly("worker_index",&SimulationManagerAccessorForCallback::worker_index)
    .def_property_readonly("vector_index",&SimulationManagerAccessorForCallback::vector_index)
    .def_property("manualDone",&SimulationManagerAccessorForCallback::manualDone,&SimulationManagerAccessorForCallback::setManualDone)
    .def("getAsset",[](SimulationManagerAccessorForCallback& v,const std::string& fullName_){return v.getAsset(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAssets",py::overload_cast<>(&SimulationManagerAccessorForCallback::getAssets<>,py::const_))
    .def("getAssets",py::overload_cast<MapIterable<PhysicalAsset,PhysicalAsset>::MatcherType>(&SimulationManagerAccessorForCallback::getAssets<>,py::const_),py::keep_alive<0,2>())
    .def("getAgent",[](SimulationManagerAccessorForCallback& v,const std::string& fullName_){return v.getAgent(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAgents",py::overload_cast<>(&SimulationManagerAccessorForCallback::getAgents<>,py::const_))
    .def("getAgents",py::overload_cast<MapIterable<Agent,Agent>::MatcherType>(&SimulationManagerAccessorForCallback::getAgents<>,py::const_),py::keep_alive<0,2>())
    .def("getController",[](SimulationManagerAccessorForCallback& v,const std::string& fullName_){return v.getController(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getControllers",py::overload_cast<>(&SimulationManagerAccessorForCallback::getControllers<>,py::const_))
    .def("getControllers",py::overload_cast<MapIterable<Controller,Controller>::MatcherType>(&SimulationManagerAccessorForCallback::getControllers<>,py::const_),py::keep_alive<0,2>())
    .def("getRuler",[](SimulationManagerAccessorForCallback& v){return v.getRuler().lock()->restoreOriginalClassForPY(true);})
    .def("getViewer",[](SimulationManagerAccessorForCallback& v){return v.getViewer().lock()->restoreOriginalClassForPY(true);})
    .def("getRewardGenerator",[](SimulationManagerAccessorForCallback& v,const int& idx){return v.getRewardGenerator(idx).lock()->restoreOriginalClassForPY(true);})
    .def("getRewardGenerators",py::overload_cast<>(&SimulationManagerAccessorForCallback::getRewardGenerators<>,py::const_))
    .def("getRewardGenerators",py::overload_cast<VectorIterable<Reward,Reward>::MatcherType>(&SimulationManagerAccessorForCallback::getRewardGenerators<>,py::const_),py::keep_alive<0,2>())
    .def("getCallback",[](SimulationManagerAccessorForCallback& v,const std::string& name_){return v.getCallback(name_).lock()->restoreOriginalClassForPY(true);})
    .def("getCallbacks",py::overload_cast<>(&SimulationManagerAccessorForCallback::getCallbacks<>,py::const_))
    .def("getCallbacks",py::overload_cast<MapIterable<Callback,Callback>::MatcherType>(&SimulationManagerAccessorForCallback::getCallbacks<>,py::const_),py::keep_alive<0,2>())
    .def("getLogger",[](SimulationManagerAccessorForCallback& v,const std::string& name_){return v.getLogger(name_).lock()->restoreOriginalClassForPY(true);})
    .def("getLoggers",py::overload_cast<>(&SimulationManagerAccessorForCallback::getLoggers<>,py::const_))
    .def("getLoggers",py::overload_cast<MapIterable<Callback,Callback>::MatcherType>(&SimulationManagerAccessorForCallback::getLoggers<>,py::const_),py::keep_alive<0,2>())
    DEF_FUNC(SimulationManagerAccessorForCallback,getManagerConfig)
    DEF_FUNC(SimulationManagerAccessorForCallback,getFactoryModelConfig)
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForCallback& v,const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig_){
        return v.generateUnmanagedChild<Callback>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForCallback& v,const std::string& baseName,const std::string& modelName,const py::object& instanceConfig_){
        return v.generateUnmanagedChild<Callback>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChildByClassName",[](SimulationManagerAccessorForCallback& v,const std::string& baseName,const std::string& className,const nl::json& modelConfig_,const nl::json& instanceConfig_){
        return v.generateUnmanagedChildByClassName<Callback>(baseName,className,modelConfig_,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChildByClassName",[](SimulationManagerAccessorForCallback& v,const std::string& baseName,const std::string& className,const py::object& modelConfig_,const py::object& instanceConfig_){
        return v.generateUnmanagedChildByClassName<Callback>(baseName,className,modelConfig_,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    DEF_FUNC(SimulationManagerAccessorForCallback,to_json_ref)
    DEF_STATIC_FUNC(SimulationManagerAccessorForCallback,from_json_ref)
    .def_static("from_json_ref",[](const py::object& obj){return SimulationManagerAccessorForCallback::from_json_ref(obj);})
    DEF_STATIC_FUNC(SimulationManagerAccessorForCallback,from_json_weakref)
    .def_static("from_json_weakref",[](const py::object& obj){return SimulationManagerAccessorForCallback::from_json_weakref(obj);})
    ;
    py::class_<SimulationManagerAccessorForPhysicalAsset,SimulationManagerAccessorBase,std::shared_ptr<SimulationManagerAccessorForPhysicalAsset>>(m,"SimulationManagerAccessorForPhysicalAsset")
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,copy)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,triggerEvent)
    .def("triggerEvent",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& name, const py::object& args){
        v.triggerEvent(name,args);
    })
    .def("getAsset",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& fullName_){return v.getAsset(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAssets",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getAssets<>,py::const_))
    .def("getAssets",py::overload_cast<MapIterable<PhysicalAsset,PhysicalAsset>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getAssets<>,py::const_),py::keep_alive<0,2>())
    .def("getAgent",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& fullName_){return v.getAgent(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getAgents",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getAgents<>,py::const_))
    .def("getAgents",py::overload_cast<MapIterable<Agent,Agent>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getAgents<>,py::const_),py::keep_alive<0,2>())
    .def("getController",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& fullName_){return v.getController(fullName_).lock()->restoreOriginalClassForPY(true);})
    .def("getControllers",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getControllers<>,py::const_))
    .def("getControllers",py::overload_cast<MapIterable<Controller,Controller>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getControllers<>,py::const_),py::keep_alive<0,2>())
    .def("getRuler",[](SimulationManagerAccessorForPhysicalAsset& v){return v.getRuler().lock()->restoreOriginalClassForPY(true);})
    .def("generateAgent",[](SimulationManagerAccessorForPhysicalAsset& v,const nl::json& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents){
        return v.generateAgent(agentConfig,agentName,parents).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateAgent",[](SimulationManagerAccessorForPhysicalAsset& v,const py::object& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents){
        return v.generateAgent(agentConfig,agentName,parents).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateAsset",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig_){
        return v.generateAsset(baseName,modelName,instanceConfig_).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateAsset",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& modelName,const py::object& instanceConfig_){
        return v.generateAsset(baseName,modelName,instanceConfig_).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateAssetByClassName",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& className,const nl::json& modelConfig_,const nl::json& instanceConfig_){
        return v.generateAssetByClassName(baseName,className,modelConfig_,instanceConfig_).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateAssetByClassName",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& className,const py::object& modelConfig_,const py::object& instanceConfig_){
        return v.generateAssetByClassName(baseName,className,modelConfig_,instanceConfig_).lock()->restoreOriginalClassForPY(true);
    })
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig_){
        return v.generateUnmanagedChild<Asset>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& modelName,const py::object& instanceConfig_){
        return v.generateUnmanagedChild<Asset>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChildByClassName",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& className,const nl::json& modelConfig_,const nl::json& instanceConfig_){
        return v.generateUnmanagedChildByClassName<Asset>(baseName,className,modelConfig_,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChildByClassName",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& baseName,const std::string& className,const py::object& modelConfig_,const py::object& instanceConfig_){
        return v.generateUnmanagedChildByClassName<Asset>(baseName,className,modelConfig_,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,requestInvitationToCommunicationBuffer)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,generateCommunicationBuffer)
    .def("generateCommunicationBuffer",[](SimulationManagerAccessorForPhysicalAsset& v,const std::string& name_,const py::object& participants_,const py::object& inviteOnRequest_){
        return v.generateCommunicationBuffer(name_,participants_,inviteOnRequest_);
    })
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,to_json_ref)
    DEF_STATIC_FUNC(SimulationManagerAccessorForPhysicalAsset,from_json_ref)
    .def_static("from_json_ref",[](const py::object& obj){return SimulationManagerAccessorForPhysicalAsset::from_json_ref(obj);})
    DEF_STATIC_FUNC(SimulationManagerAccessorForPhysicalAsset,from_json_weakref)
    .def_static("from_json_weakref",[](const py::object& obj){return SimulationManagerAccessorForPhysicalAsset::from_json_weakref(obj);})
    ;
    py::class_<SimulationManagerAccessorForAgent,SimulationManagerAccessorBase,std::shared_ptr<SimulationManagerAccessorForAgent>>(m,"SimulationManagerAccessorForAgent")
    DEF_FUNC(SimulationManagerAccessorForAgent,copy)
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForAgent& v,const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig_){
        return v.generateUnmanagedChild<Agent>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    .def("generateUnmanagedChild",[](SimulationManagerAccessorForAgent& v,const std::string& baseName,const std::string& modelName,const py::object& instanceConfig_){
        return v.generateUnmanagedChild<Agent>(baseName,modelName,instanceConfig_)->restoreOriginalClassForPY(false);
    })
    DEF_FUNC(SimulationManagerAccessorForAgent,requestInvitationToCommunicationBuffer)
    .def("getRuler",[](SimulationManagerAccessorForAgent& v){return v.getRuler().lock()->restoreOriginalClassForPY(true);})
    DEF_FUNC(SimulationManagerAccessorForAgent,to_json_ref)
    DEF_STATIC_FUNC(SimulationManagerAccessorForAgent,from_json_ref)
    .def_static("from_json_ref",[](const py::object& obj){return SimulationManagerAccessorForAgent::from_json_ref(obj);})
    DEF_STATIC_FUNC(SimulationManagerAccessorForAgent,from_json_weakref)
    .def_static("from_json_weakref",[](const py::object& obj){return SimulationManagerAccessorForAgent::from_json_weakref(obj);})
    ;
}

