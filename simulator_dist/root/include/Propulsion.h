// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "PhysicalAsset.h"
namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITH_TRAMPOLINE(Propulsion,PhysicalAsset)
	/*
	Generic Propulsion base class.
    */
    public:
    Propulsion(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~Propulsion();
    virtual double getFuelFlowRate()=0;//In [kg/s]
    virtual double getThrust()=0;// In [N]
    virtual double calcFuelFlowRate(const nl::json& args)=0;//as generic interface
    virtual double calcThrust(const nl::json& args)=0;//as generic interface
    virtual void setPowerCommand(const double& pCmd_)=0;//Normalized command in [0,1]
    virtual void control();
};

DECLARE_TRAMPOLINE(Propulsion)
    virtual double getFuelFlowRate() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(double,Base,getFuelFlowRate);
    }
    virtual double getThrust() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(double,Base,getThrust);
    }
    virtual double calcFuelFlowRate(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(double,Base,calcFuelFlowRate,args);
    }
    virtual double calcThrust(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(double,Base,calcThrust,args);
    }
    virtual void setPowerCommand(const double& pCmd_) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(void,Base,setPowerCommand,pCmd_);
    }
};

void exportPropulsion(py::module& m);