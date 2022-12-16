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
class RulerAccessor;

DECLARE_CLASS_WITH_TRAMPOLINE(Ruler,Callback)
    public:
    enum class EndReason{
        TIMEUP,
        NO_ONE_ALIVE,
        NOTYET
    };
    std::map<std::string,bool> dones;
    double maxTime;
    std::vector<std::string> teams;
    std::string winner;
    EndReason endReason;
    std::map<std::string,double> score,stepScore;
    nl::json observables;
    //constructors & destructor
    Ruler(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~Ruler();
    //functions
    virtual void onEpisodeBegin();
    virtual void onStepBegin();
    virtual void onInnerStepBegin();
    virtual void onInnerStepEnd();
    virtual void onStepEnd();
    virtual void onEpisodeEnd();
    virtual void checkDone();
    virtual double getStepScore(const std::string &key);
    virtual double getScore(const std::string &key);
    virtual double getStepScore(const std::shared_ptr<Agent> key);
    virtual double getScore(const std::shared_ptr<Agent> key);
    virtual std::shared_ptr<RulerAccessor> getAccessor();
    protected:
    std::shared_ptr<RulerAccessor> accessor;
};

DECLARE_BASE_CLASS(RulerAccessor)
    //Accessor for Agent
    public:
    RulerAccessor(std::shared_ptr<Ruler> r);
    virtual ~RulerAccessor();
    virtual std::string getFactoryBaseName() const;//FactoryにおけるbaseNameを返す。
    virtual std::string getFactoryClassName() const;//Factoryにおけるクラス名を返す。
    virtual std::string getFactoryModelName() const;//Factoryにおけるモデル名を返す。
    virtual double getStepScore(const std::string &key);
    virtual double getScore(const std::string &key);
    virtual double getStepScore(const std::shared_ptr<Agent> key);
    virtual double getScore(const std::shared_ptr<Agent> key);
    const nl::json& observables;
    template<class T>
    bool isinstance(){
        return util::isinstance<T>(ruler);
    }
    virtual bool isinstancePY(py::object cls);
    protected:
    std::weak_ptr<Ruler> ruler;
    static nl::json dummy;
};

DECLARE_TRAMPOLINE(Ruler)
    virtual void checkDone() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,checkDone);
    }
    virtual double getStepScore(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getStepScore,key);
    }
    virtual double getScore(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getScore,key);
    }
    virtual double getStepScore(const std::shared_ptr<Agent> agent) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getStepScore,agent);
    }
    virtual double getScore(const std::shared_ptr<Agent> agent) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getScore,agent);
    }
    virtual std::shared_ptr<RulerAccessor> getAccessor() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(std::shared_ptr<RulerAccessor>,Base,getAccessor);
    }
};
DECLARE_BASE_TRAMPOLINE(RulerAccessor)
    virtual std::string getFactoryBaseName() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(std::string,Base,getFactoryBaseName);
    }
    virtual std::string getFactoryClassName() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(std::string,Base,getFactoryClassName);
    }
    virtual std::string getFactoryModelName() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(std::string,Base,getFactoryModelName);
    }
    virtual double getStepScore(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getStepScore,key);
    }
    virtual double getScore(const std::string &key) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getScore,key);
    }
    virtual double getStepScore(const std::shared_ptr<Agent> agent) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getStepScore,agent);
    }
    virtual double getScore(const std::shared_ptr<Agent> agent) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getScore,agent);
    }
    virtual bool isinstancePY(py::object cls) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isinstancePY,cls);
    }
};

void exportRuler(py::module &m);


