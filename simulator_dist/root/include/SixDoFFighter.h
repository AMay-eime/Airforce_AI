// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "MathUtility.h"
#include "FlightControllerUtility.h"
#include "CoordinatedFighter.h"
#include "Utility.h"
namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITH_TRAMPOLINE(SixDoFFighter,Fighter)
	/*
	6 degree-of-freedom aircraft model.
	Aerodynamic coefficients and engine characteristics are based on public released F-16 information listed below.
    Most of the model is based on [1], wave drag is based on [3], and supersonic CD0 is based on [4].
    Flight controller model uses a simple combination of dynamic inversion and LQR.
    Control variables for attitude control are based on [5].
	References:
	[1] Stevens, Brian L., et al. "Aircraft control and simulation: dynamics, controls design, and autonomous systems." John Wiley & Sons, 2015.
	[3] Krus, P., et al. "Modelling of Transonic and Supersonic Aerodynamics for Conceptual Design and Flight Simulation." Proceedings of the 10th Aerospace Technology Congress, 2019.
    [4] Webb, T. S., et al. "Correlation of F-16 Aerodynamics and Performance Predictions with Early Flight Test Results." Agard Conference Proceedings, N242, 1977.
    [5] Heidlauf, P., et al. "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers." ARCH18. 5th International Workshop on Applied Verification of Countinuous and Hybrid Systems, 2018.
	*/
    //protected:
    public:
    //model parameters
    double S,Le,b,mac,XcgR,Xcg;
    Eigen::Matrix3d I,Iinv;//慣性テンソル
    double deLimit,daLimit,drLimit,deMaxRate,daMaxRate,drMaxRate,deTimeConstant,daTimeConstant,drTimeConstant;
    Eigen::VectorXd cdwTable;
    double de,da,dr,T;
    public:
    //constructors & destructor
    SixDoFFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~SixDoFFighter();
    //functions
    virtual void validate();
    virtual double calcOptimumCruiseFuelFlowRatePerDistance();//Minimum fuel flow rate (per distance) in level steady flight
    virtual Eigen::VectorXd trim6(double alt,double V,double tgtBank);
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives);
    virtual Eigen::VectorXd calcDerivative(const Eigen::VectorXd& x,const Eigen::VectorXd& u);
    virtual void calcMotion(double dt) override;
    DECLARE_CLASS_WITHOUT_TRAMPOLINE(FlightController,Fighter::FlightController)
        public:
        double PositiveNzLimit,NegativeNzLimit;
        double maxNyrCmd,maxPsCmd;
        double kPsCmdP,kPsCmdD,kNzCmdP,kNzCmdD,kNyrCmdP,kNyrCmdD;
        AltitudeKeeper altitudeKeeper;
        std::vector<Eigen::MatrixXd> Ak,Bk,Qk,Rk;
        Eigen::MatrixXd P,K;
        nl::json buffer;
        std::map<std::string,Eigen::VectorXd> integrals;
        //constructors & destructor
        FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_);
        virtual nl::json getDefaultCommand() override;
        virtual nl::json calc(const nl::json &cmd) override;
        nl::json calcDirect(const nl::json &cmd);
        nl::json calcFromManualInput(const nl::json &cmd);
        nl::json calcFromDirAndVel(const nl::json &cmd);
        Eigen::VectorXd calcU(const nl::json &cmd,const Eigen::VectorXd& x);
    };
};

DECLARE_TRAMPOLINE(SixDoFFighter)
    virtual double calcOptimumCruiseFuelFlowRatePerDistance() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(double,Base,calcOptimumCruiseFuelFlowRatePerDistance);
    }
    virtual void calcMotion(double dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,calcMotion,dt);
    }
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives) override{
        typedef std::map<std::string,double> retType;
        PYBIND11_OVERRIDE(retType,Base,calcAeroV,alpha_,beta_,de_,da_,dr_,mach_,withDerivatives);
    }
};

DECLARE_CLASS_WITHOUT_TRAMPOLINE(StevensFighter,SixDoFFighter)
	/*
	6 degree-of-freedom aircraft model with aerodynamic coefficients written in [1].
	References:
	[1] Stevens, Brian L., et al. "Aircraft control and simulation: dynamics, controls design, and autonomous systems." John Wiley & Sons, 2015.
	*/
    Eigen::Tensor<double,2> dampTable,cxTable,clTable,cmTable,cnTable,dldaTable,dldrTable,dndaTable,dndrTable;
    Eigen::Tensor<double,1> czTable;
    Eigen::VectorXd alphaTable,betaSymTable,betaAsymTable,deTable;
    double dCydB,dCydda,dCyddr,dCzdde;
    StevensFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~StevensFighter();
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives);
};

DECLARE_CLASS_WITHOUT_TRAMPOLINE(MorelliFighter,SixDoFFighter)
	/*
	6 degree-of-freedom aircraft model variant with aerodynamic coefficients written in [5].
	References:
	[5] Morelli, E. A. "Global Nonlinear Parametric Modeling with Application to F-16 Aerodynamics." Proceedings of the 1998 American Control Conference, 1998.
	*/
    std::map<std::string,Eigen::VectorXd> aeroC;
    MorelliFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~MorelliFighter();
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives);
};

void exportSixDoFFighter(py::module& m);