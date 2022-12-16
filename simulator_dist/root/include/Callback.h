// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <mutex>
#include <pybind11/pybind11.h>
#include "Entity.h"
class SimulationManager;
class SimulationManagerAccessorForCallback;

namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITH_TRAMPOLINE(Callback,Entity)
    friend class SimulationManager;
    public:
    std::mutex mtx;
    std::shared_ptr<SimulationManagerAccessorForCallback> manager;
    bool acceptReconfigure;
    std::string name;
    public:
    //constructors & destructor
    Callback(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~Callback();
    //functions
    virtual std::string getName() const;
    virtual void onGetObservationSpace();
    virtual void onGetActionSpace();
    virtual void onMakeObs();
    virtual void onDeployAction();
    virtual void onEpisodeBegin();
    virtual void onValidationEnd();
    virtual void onStepBegin();
    virtual void onInnerStepBegin();
    virtual void onInnerStepEnd();
    virtual void onStepEnd();
    virtual void onEpisodeEnd();
};

DECLARE_TRAMPOLINE(Callback)
    virtual std::string getName() const{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(std::string,Base,getName);
    }
    virtual void onGetObservationSpace() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onGetObservationSpace);
    }
    virtual void onGetActionSpace() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onGetActionSpace);
    }
    virtual void onMakeObs() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onMakeObs);
    }
    virtual void onDeployAction() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onDeployAction);
    }
    virtual void onEpisodeBegin() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onEpisodeBegin);
    }
    virtual void onValidationEnd() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onValidationEnd);
    }
    virtual void onStepBegin() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onStepBegin);
    }
    virtual void onInnerStepBegin() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onInnerStepBegin);
    }
    virtual void onInnerStepEnd() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onInnerStepEnd);
    }
    virtual void onStepEnd() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onStepEnd);
    }
    virtual void onEpisodeEnd() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,onEpisodeEnd);
    }
};

void exportCallback(py::module &m);
