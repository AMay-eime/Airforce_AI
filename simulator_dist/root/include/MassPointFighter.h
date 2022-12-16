// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <deque>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "MathUtility.h"
#include "Fighter.h"
#include "Controller.h"
#include "Track.h"
namespace py=pybind11;
namespace nl=nlohmann;
class Missile;
class FighterSensor;

DECLARE_CLASS_WITH_TRAMPOLINE(IdealDirectPropulsion,Propulsion)
	/*
	An ideal, simplest propulsion model linear acceleration without fuel consumption
    */
    public:
    double tMin,tMax;//Min. & Max. thrust in [N]
    //engine states
    double thrust;
    double pCmd;

    IdealDirectPropulsion(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    ~IdealDirectPropulsion();
    virtual double getFuelFlowRate();
    virtual double getThrust();
    virtual double calcFuelFlowRate(const nl::json& args);//generic interface
    virtual double calcThrust(const nl::json& args);//generic interface
    virtual void setPowerCommand(const double& pCmd_);
};
DECLARE_TRAMPOLINE(IdealDirectPropulsion)
    virtual double getFuelFlowRate() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getFuelFlowRate);
    }
    virtual double getThrust() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,getThrust);
    }
    virtual double calcFuelFlowRate(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,calcFuelFlowRate,args);
    }
    virtual double calcThrust(const nl::json& args) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,calcThrust,args);
    }
    virtual void setPowerCommand(const double& pCmd_) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,setPowerCommand,pCmd_);
    }
};

DECLARE_CLASS_WITH_TRAMPOLINE(MassPointFighter,Fighter)
    public:
    //model parameters
    double vMin,vMax,rollMax,pitchMax,yawMax;
    public:
    //constructors & destructor
    MassPointFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~MassPointFighter();
    //functions
    virtual double calcOptimumCruiseFuelFlowRatePerDistance();//Minimum fuel flow rate (per distance) in level steady flight
    virtual void calcMotion(double dt) override;
    DECLARE_CLASS_WITHOUT_TRAMPOLINE(FlightController,Fighter::FlightController)
        public:
        //constructors & destructor
        FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_);
        virtual ~FlightController();
        virtual nl::json getDefaultCommand() override;
        virtual nl::json calc(const nl::json &cmd) override;
    };
};
DECLARE_TRAMPOLINE(MassPointFighter)
    virtual double calcOptimumCruiseFuelFlowRatePerDistance() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,calcOptimumCruiseFuelFlowRatePerDistance);
    }
    virtual void calcMotion(double dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,calcMotion,dt);
    }
};

void exportMassPointFighter(py::module& m);