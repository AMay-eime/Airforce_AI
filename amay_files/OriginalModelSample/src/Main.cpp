// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#define MY_MODULE_NAME libOriginalModelSample
#include <ASRCAISim1/Factory.h>
#include "R4AgentSample01S.h"
#include "R4AgentSample01M.h"
#include "R4RewardSample01.h"
#include "R4RewardSample02.h"
#include <iostream>
namespace py=pybind11;


PYBIND11_MODULE(MY_MODULE_NAME,m)
{    
    using namespace pybind11::literals;
    m.doc()="OriginalModelSample";
    exportR4AgentSample01S(m);
    exportR4AgentSample01M(m);
    exportR4RewardSample01(m);
    exportR4RewardSample02(m);
    FACTORY_ADD_CLASS(Agent,R4AgentSample01S)
    FACTORY_ADD_CLASS(Agent,R4AgentSample01M)
    FACTORY_ADD_CLASS(Reward,R4RewardSample01)
    FACTORY_ADD_CLASS(Reward,R4RewardSample02)
}
