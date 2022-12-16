// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "MathUtility.h"
#include "Utility.h"
#include "Reward.h"
#include "Fighter.h"
#include "Missile.h"
namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITHOUT_TRAMPOLINE(WinLoseReward,TeamReward)
	/*得点差や途中経過によらず、終了時の勝ち負けのみによる報酬を与える例
	*/
    public:
    //parameters
    double win,lose,draw;
    //internal variables
    std::string westSider,eastSider;
    //constructors & destructor
    WinLoseReward(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~WinLoseReward();
    //functions
    virtual void onEpisodeBegin();
    virtual void onStepEnd();
};
void exportWinLoseReward(py::module& m);
