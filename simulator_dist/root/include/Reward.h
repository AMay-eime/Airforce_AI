// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include "Callback.h"

namespace py=pybind11;
namespace nl=nlohmann;

class Agent;
class Asset;

DECLARE_CLASS_WITH_TRAMPOLINE(Reward,Callback)
    public:
    nl::json j_target;
    std::vector<std::string> target;
    std::map<std::string,double> reward,totalReward;
    //constructors & destructor
    Reward(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~Reward();
    //functions
    virtual void onEpisodeBegin();
    virtual void onStepBegin();
    virtual void onStepEnd();
    virtual void setupTarget(const std::string& query,bool asTeam);
    virtual double getReward(const std::string &key);
    virtual double getTotalReward(const std::string &key);
    virtual double getReward(const std::shared_ptr<Agent> key);
    virtual double getTotalReward(const std::shared_ptr<Agent> key);
};
DECLARE_TRAMPOLINE(Reward)
    //virtual functions
    virtual void setupTarget(const std::string& query,bool asTeam) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,setupTarget,query,asTeam);
    }
    virtual double getReward(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getReward,key);
    }
    virtual double getTotalReward(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getTotalReward,key);
    }
    virtual double getReward(const std::shared_ptr<Agent> key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getReward,key);
    }
    virtual double getTotalReward(const std::shared_ptr<Agent> key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getTotalReward,key);
    }
};
DECLARE_CLASS_WITHOUT_TRAMPOLINE(AgentReward,Reward)
    public:
    //constructors & destructor
    AgentReward(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~AgentReward();
    //functions
    virtual void onEpisodeBegin();
    virtual double getReward(const std::string &key);
    virtual double getTotalReward(const std::string &key);
    virtual double getReward(const std::shared_ptr<Agent> key);
    virtual double getTotalReward(const std::shared_ptr<Agent> key);
};
DECLARE_CLASS_WITHOUT_TRAMPOLINE(TeamReward,Reward)
    public:
    //constructors & destructor
    TeamReward(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~TeamReward();
    //functions
    virtual void onEpisodeBegin();
    virtual double getReward(const std::string &key);
    virtual double getTotalReward(const std::string &key);
    virtual double getReward(const std::shared_ptr<Agent> key);
    virtual double getTotalReward(const std::shared_ptr<Agent> key);
};
DECLARE_CLASS_WITHOUT_TRAMPOLINE(ScoreReward,TeamReward)
    public:
    //constructors & destructor
    ScoreReward(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~ScoreReward();
    //functions
    virtual void onStepEnd();
};
void exportReward(py::module &m);



