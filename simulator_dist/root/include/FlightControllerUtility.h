// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "MotionState.h"

namespace py=pybind11;

Eigen::Vector3d PYBIND11_EXPORT pitchLimitter(const MotionState& motion, const Eigen::Vector3d& dstDir,const double& pitchLimit);

class  PYBIND11_EXPORT AltitudeKeeper{
    public:
    AltitudeKeeper();
    AltitudeKeeper(const nl::json& config);
    Eigen::Vector3d operator()(const MotionState& motion, const Eigen::Vector3d& dstDir, const double& dstAlt);
    double getDstPitch(const MotionState& motion, const double& dstAlt);
    double inverse(const MotionState& motion, const double& dstPitch);
    public:
    double pGain;//目標高度との差に対するゲイン。負の値。
    double dGain;//高度変化に対するゲイン。正の値。
    double minPitch;//ピッチ角の下限
    double maxPitch;//ピッチ角の上限
};

void exportFlightControllerUtility(py::module &m);
