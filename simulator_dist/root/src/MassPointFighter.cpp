// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "MassPointFighter.h"
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"
using namespace util;

IdealDirectPropulsion::IdealDirectPropulsion(const nl::json& modelConfig_,const nl::json& instanceConfig_)
    :Propulsion(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    tMin=getValueFromJsonKRD(modelConfig.at("propulsion"),"tMin",randomGen,-gravity*10000.0);
    tMax=getValueFromJsonKRD(modelConfig.at("propulsion"),"tMax",randomGen,gravity*10000.0);
    pCmd=0.0;
    thrust=0.0;
}
IdealDirectPropulsion::~IdealDirectPropulsion(){}
double IdealDirectPropulsion::getFuelFlowRate(){
    return 0.0;
}
double IdealDirectPropulsion::getThrust(){
    return thrust;
}
double IdealDirectPropulsion::calcFuelFlowRate(const nl::json& args){
    return 0.0;
}
double IdealDirectPropulsion::calcThrust(const nl::json& args){
    double arg_pCmd=args.at("pCmd");
    return tMin+arg_pCmd*(tMax-tMin);
}
void IdealDirectPropulsion::setPowerCommand(const double& pCmd_){
    pCmd=pCmd_;
    thrust=tMin+pCmd*(tMax-tMin);
}

MassPointFighter::MassPointFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    //modelConfigで指定するもの
    vMin=getValueFromJsonKRD(modelConfig.at("dynamics"),"vMin",randomGen,150.0);
    vMax=getValueFromJsonKRD(modelConfig.at("dynamics"),"vMax",randomGen,450.0);
    rollMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"rollMax",randomGen,180.0));
    pitchMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"pitchMax",randomGen,30.0));
    yawMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"yawMax",randomGen,30.0));
    observables["spec"].merge_patch({
        {"dynamics",{
            {"vMin",vMin},
            {"vMax",vMax},
            {"rollMax",rollMax},
            {"pitchMax",pitchMax},
            {"yawMax",yawMax}
        }}
    });
}
MassPointFighter::~MassPointFighter(){
}
double MassPointFighter::calcOptimumCruiseFuelFlowRatePerDistance(){
    return 0.0;//no fuel consideration
}
void MassPointFighter::calcMotion(double dt){
    vel_prev=motion.vel;
    pos_prev=motion.pos;
    double V=motion.vel.norm();
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    Eigen::Vector3d omegaB=ctrl.at("omegaB");
    double accel=engine.lock()->getThrust()/m;
    motion.omega=relBtoI(omegaB);
    double theta=motion.omega.norm()*dt;
    if(theta>1e-6){
        Eigen::Vector3d ax=motion.omega.normalized();
        Quaternion dq=Quaternion::fromAngle(ax,theta);
        motion.q=dq*motion.q;
    }else{
        Eigen::VectorXd dq=motion.q.dqdwi()*motion.omega*dt;
        motion.q=(motion.q+Quaternion(dq)).normalized();
    }
    double Vaft=std::max(vMin,std::min(vMax,V+accel*dt));
    motion.vel=relBtoI(Eigen::Vector3d(Vaft,0,0));
    motion.pos+=motion.vel*dt;
    motion.time=manager->getTime()+dt;
    motion.calcQh();
}
MassPointFighter::FlightController::FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter::FlightController(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
MassPointFighter::FlightController::~FlightController(){
}

nl::json MassPointFighter::FlightController::getDefaultCommand(){
    return {
        {"roll",0.0},
        {"pitch",0.0},
        {"yaw",0.0},
        {"accel",0.0}
    };
}
nl::json MassPointFighter::FlightController::calc(const nl::json &cmd){
    auto p=getShared<MassPointFighter>(parent);
    auto roll=std::clamp(cmd.at("roll").get<double>(),-1.,1.)*p->rollMax;
    auto pitch=std::clamp(cmd.at("pitch").get<double>(),-1.,1.)*p->pitchMax;
    auto yaw=std::clamp(cmd.at("yaw").get<double>(),-1.,1.)*p->yawMax;
    double pCmd;
    if(cmd.contains("throttle")){//0〜1
        pCmd=std::clamp(cmd.at("throttle").get<double>(),0.,1.);
    }else if(cmd.contains("accel")){//-1〜+1
        pCmd=(1+std::clamp(cmd.at("accel").get<double>(),-1.,1.))*0.5;
    }else{
        throw std::runtime_error("Either 'throttle' or 'accel' must be in cmd keys.");
    }
    return {
        {"omegaB",Eigen::Vector3d(roll,pitch,yaw)},
        {"pCmd",pCmd}
    };
}

void exportMassPointFighter(py::module& m)
{
    using namespace pybind11::literals;
    EXPOSE_CLASS(IdealDirectPropulsion)
    DEF_READWRITE(IdealDirectPropulsion,tMin)
    DEF_READWRITE(IdealDirectPropulsion,tMax)
    DEF_READWRITE(IdealDirectPropulsion,thrust)
    DEF_READWRITE(IdealDirectPropulsion,pCmd)
    DEF_FUNC(IdealDirectPropulsion,getFuelFlowRate)
    DEF_FUNC(IdealDirectPropulsion,getThrust)
    .def("calcFuelFlowRate",[](IdealDirectPropulsion& v,const py::object &args){
        return v.calcFuelFlowRate(args);
    })
    .def("calcThrust",[](IdealDirectPropulsion& v,const py::object &args){
        return v.calcThrust(args);
    })
    DEF_FUNC(IdealDirectPropulsion,setPowerCommand)
    ;

    auto clsObj=EXPOSE_CLASS(MassPointFighter)
    DEF_FUNC(MassPointFighter,calcOptimumCruiseFuelFlowRatePerDistance)
    DEF_FUNC(MassPointFighter,calcMotion)
    DEF_READWRITE(MassPointFighter,vMin)
    DEF_READWRITE(MassPointFighter,vMax)
    DEF_READWRITE(MassPointFighter,rollMax)
    DEF_READWRITE(MassPointFighter,pitchMax)
    DEF_READWRITE(MassPointFighter,yawMax)
    ;
    EXPOSE_INNER_CLASS(clsObj,MassPointFighter,FlightController)
    DEF_FUNC(MassPointFighter::FlightController,getDefaultCommand)
    DEF_FUNC(MassPointFighter::FlightController,calc)
    .def("calc",[](MassPointFighter::FlightController& v,const py::object &cmd){
        return v.calc(cmd);
    })
    ;
}
