// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Callback.h"
#include "Utility.h"
#include "SimulationManager.h"
using namespace util;

Callback::Callback(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Entity(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    manager=instanceConfig.at("manager");
    name=instanceConfig.at("name");
    try{
        acceptReconfigure=modelConfig.at("acceptReconfigure");
    }catch(...){
        acceptReconfigure=false;
    }
    firstTick[SimPhase::ON_EPISODE_BEGIN]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_VALIDATION_END]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_STEP_BEGIN]=0;
    if(instanceConfig.contains("firstTick")){
        nl::json firstTicks=instanceConfig.at("firstTick");
        std::string firstTickUnit="tick";
        if(firstTicks.contains("unit")){
            firstTickUnit=firstTicks.at("unit");
        }
        if(firstTickUnit=="time"){
            double dt=instanceConfig.at("baseTimeStep");
            firstTick[SimPhase::ON_INNERSTEP_BEGIN]=std::lround(getValueFromJsonKRD(firstTicks,"value",randomGen,0.0)/dt);
        }else if(firstTickUnit=="tick"){
            firstTick[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(firstTicks,"value",randomGen,0);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of firstTick."+firstTickUnit);
        }
    }else if(modelConfig.contains("firstTick")){
        nl::json firstTicks=modelConfig.at("firstTick");
        std::string firstTickUnit="tick";
        if(firstTicks.contains("unit")){
            firstTickUnit=firstTicks.at("unit");
        }
        if(firstTickUnit=="time"){
            double dt=instanceConfig.at("baseTimeStep");
            firstTick[SimPhase::ON_INNERSTEP_BEGIN]=std::lround(getValueFromJsonKRD(firstTicks,"value",randomGen,0.0)/dt);
        }else if(firstTickUnit=="tick"){
            firstTick[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(firstTicks,"value",randomGen,0);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of firstTick."+firstTickUnit);
        }
    }else{
        firstTick[SimPhase::ON_INNERSTEP_BEGIN]=0;
    }
    firstTick[SimPhase::ON_INNERSTEP_END]=firstTick[SimPhase::ON_INNERSTEP_BEGIN]+1;
    firstTick[SimPhase::ON_STEP_END]=manager->getAgentInterval();
    firstTick[SimPhase::ON_EPISODE_END]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_GET_OBSERVATION_SPACE]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_GET_ACTION_SPACE]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_MAKE_OBS]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_DEPLOY_ACTION]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_EPISODE_BEGIN]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_VALIDATION_END]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_STEP_BEGIN]=manager->getAgentInterval();
    if(instanceConfig.contains("interval")){
        nl::json intervals=instanceConfig.at("interval");
        std::string intervalUnit="tick";
        if(intervals.contains("unit")){
            intervalUnit=intervals.at("unit");
        }
        if(intervalUnit=="time"){
            double dt=instanceConfig.at("baseTimeStep");
            interval[SimPhase::ON_INNERSTEP_BEGIN]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"value",randomGen,dt)/dt));
        }else if(intervalUnit=="tick"){
            interval[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(intervals,"value",randomGen,1);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of interval."+intervalUnit);
        }
    }else if(modelConfig.contains("interval")){
        nl::json intervals=modelConfig.at("interval");
        std::string intervalUnit="tick";
        if(intervals.contains("unit")){
            intervalUnit=intervals.at("unit");
        }
        if(intervalUnit=="time"){
            double dt=instanceConfig.at("baseTimeStep");
            interval[SimPhase::ON_INNERSTEP_BEGIN]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"value",randomGen,dt)/dt));
        }else if(intervalUnit=="tick"){
            interval[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(intervals,"value",randomGen,1);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of interval."+intervalUnit);
        }
    }else{
        interval[SimPhase::ON_INNERSTEP_BEGIN]=1;
    }
    interval[SimPhase::ON_INNERSTEP_END]=interval[SimPhase::ON_INNERSTEP_BEGIN];
    interval[SimPhase::ON_STEP_END]=manager->getAgentInterval();
    interval[SimPhase::ON_EPISODE_END]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_GET_OBSERVATION_SPACE]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_GET_ACTION_SPACE]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_MAKE_OBS]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_DEPLOY_ACTION]=0;//meaningless but defined for same interface
}
Callback::~Callback(){}
std::string Callback::getName() const{
    return name;
}
void Callback::onGetObservationSpace(){
}
void Callback::onGetActionSpace(){
}
void Callback::onMakeObs(){
}
void Callback::onDeployAction(){
}
void Callback::onEpisodeBegin(){
}
void Callback::onValidationEnd(){
}
void Callback::onStepBegin(){
}
void Callback::onInnerStepBegin(){
}
void Callback::onInnerStepEnd(){
}
void Callback::onStepEnd(){
}
void Callback::onEpisodeEnd(){
}

void exportCallback(py::module &m)
{
    using namespace pybind11::literals;
    EXPOSE_CLASS(Callback)
    DEF_FUNC(Callback,getName)
    DEF_FUNC(Callback,onGetObservationSpace)
    DEF_FUNC(Callback,onGetActionSpace)
    DEF_FUNC(Callback,onMakeObs)
    DEF_FUNC(Callback,onDeployAction)
    DEF_FUNC(Callback,onEpisodeBegin)
    DEF_FUNC(Callback,onValidationEnd)
    DEF_FUNC(Callback,onStepBegin)
    DEF_FUNC(Callback,onInnerStepBegin)
    DEF_FUNC(Callback,onInnerStepEnd)
    DEF_FUNC(Callback,onStepEnd)
    DEF_FUNC(Callback,onEpisodeEnd)
    DEF_READWRITE(Callback,manager)
    DEF_READWRITE(Callback,name)
    DEF_READWRITE(Callback,acceptReconfigure)
    ;
}