// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Propulsion.h"
#include "Fighter.h"
using namespace util;

Propulsion::Propulsion(const nl::json& modelConfig_,const nl::json& instanceConfig_)
    :PhysicalAsset(modelConfig_,instanceConfig_){
    if(isDummy){return;}
}
Propulsion::~Propulsion(){}
void Propulsion::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive()){
        nl::json ctrl=p->controllers["FlightController"].lock()->commands["motion"];
        setPowerCommand(ctrl.at("pCmd"));
    }
}

void exportPropulsion(py::module& m)
{
    using namespace pybind11::literals;
    EXPOSE_CLASS(Propulsion)
    DEF_FUNC(Propulsion,getFuelFlowRate)
    DEF_FUNC(Propulsion,getThrust)
    .def("calcFuelFlowRate",[](Propulsion& v,const py::object &args){
        return v.calcFuelFlowRate(args);
    })
    .def("calcThrust",[](Propulsion& v,const py::object &args){
        return v.calcThrust(args);
    })
    DEF_FUNC(Propulsion,setPowerCommand)
    DEF_FUNC(Propulsion,control)
    ;
}
