// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "FlightControllerUtility.h"
#include "MathUtility.h"
#include "Utility.h"
#include "Units.h"

namespace py=pybind11;
using namespace util;

Eigen::Vector3d pitchLimitter(const MotionState& motion, const Eigen::Vector3d& dstDir,const double& pitchLimit){
    Eigen::Vector3d t=motion.vel.normalized();//現在の進行方向
    Eigen::Vector3d dd=dstDir.normalized();//目標方向
    Eigen::Vector3d b=t.cross(dd);//最小回転量で到達できる回転軸(=「当初回転軸」)
	double snTheta=std::min(1.0,b.norm());
    double csTheta=std::clamp(t.dot(dd),-1.0,1.0);
    double eTheta=atan2(snTheta,csTheta);
    b/=snTheta;
    Eigen::Vector3d pz(0,0,1);//+z方向
    double sint=abs(t.dot(pz));//現在のピッチ角の正弦
    double sindd=abs(dd.dot(pz));//目標方向のピッチ角の正弦
    double sing=std::max(sin(pitchLimit),std::max(sint,sindd));//制限ピッチ角の正弦
    double sinu=pz.cross(b).norm();//b軸まわりの回転面上におけるピッチ角(の絶対値)の最大値
    if(sinu>sing){
        //回転面が制限範囲を通過する場合
        Eigen::Vector3d vert=b.cross(pz.cross(b)).normalized();//ピッチ角(の絶対値)が最大となる方向
        if(vert.dot(dd+t)<0){
            vert=-vert;
        }
        if(b.dot(t.cross(vert))>=0 && b.dot(vert.cross(dd))>=0){
            //最短経路が垂直面を通過する場合、ピッチ角が制限値以内に収まるように回転軸を補正する。
            //補正後回転軸の候補は4種類あるため、もとの軸に最も近いものを選択する。
            Eigen::Vector3d hh=t.cross(pz);
            if(hh.norm()<1e-6){
                hh=(b.cross(pz)).cross(pz).normalized();
            }else{
                hh.normalize();
            }
            Eigen::Vector3d vv=hh.cross(t).normalized();
            double sinRot=1;
            if(sing>sint){
                double cosg=sqrt(std::max(0.0,1-sing*sing));
                sinRot=cosg/sqrt(std::max(0.0,1-sint+sint));
            }
            double cosRot=sqrt(std::max(0.0,1-sinRot*sinRot));
            Eigen::Vector3d bb1=hh*cosRot+vv*sinRot;
            Eigen::Vector3d bb2=hh*cosRot-vv*sinRot;
            Eigen::Vector3d bb3=-hh*cosRot+vv*sinRot;
            Eigen::Vector3d bb4=-hh*cosRot-vv*sinRot;
            Eigen::Vector3d bb12,bb34;
            if(b.dot(bb1)>b.dot(bb2)){
                bb12=bb1.normalized();
            }else{
                bb12=bb2.normalized();
            }
            if(b.dot(bb3)>b.dot(bb4)){
                bb34=bb3.normalized();
            }else{
                bb34=bb4.normalized();
            }
            if(b.dot(bb12)>b.dot(bb34)){
                b=bb12.normalized();
            }else{
                b=bb34.normalized();
            }
        }
    }
    return b;
}
AltitudeKeeper::AltitudeKeeper(){
    pGain=-1.0/3000.0;
    dGain=-1.0/1000.0;
    minPitch=deg2rad(-45.0);
    maxPitch=deg2rad(30.0);
}
AltitudeKeeper::AltitudeKeeper(const nl::json& config):AltitudeKeeper(){
    pGain=getValueFromJsonKD(config,"pGain",pGain);
    dGain=getValueFromJsonKD(config,"dGain",dGain);
    minPitch=getValueFromJsonKD(config,"minPitch",minPitch);
    maxPitch=getValueFromJsonKD(config,"maxPitch",maxPitch);
}
Eigen::Vector3d AltitudeKeeper::operator()(const MotionState& motion, const Eigen::Vector3d& dstDir, const double& dstAlt){
    double eAlt=-motion.pos(2) - dstAlt;
    double vAlt=-motion.vel(2);
    double V=motion.vel.norm();
    Eigen::Vector3d ret(dstDir(0),dstDir(1),0);
    ret.normalize();
    double dstPitch=asin(std::clamp<double>((pGain*eAlt+dGain*vAlt)/V,-1,1));//上向き正の目標ピッチ角
    dstPitch=std::clamp<double>(dstPitch,minPitch,maxPitch);
    double cs=cos(dstPitch);
    double sn=sin(dstPitch);
    ret<<cs*ret(0),cs*ret(1),-sn;
    return ret;
}
double AltitudeKeeper::getDstPitch(const MotionState& motion, const double& dstAlt){
    double eAlt=-motion.pos(2) - dstAlt;
    double vAlt=-motion.vel(2);
    double V=motion.vel.norm();
    double dstPitch=asin(std::clamp<double>((pGain*eAlt+dGain*vAlt)/V,-1,1));//上向き正の目標ピッチ角
    return std::clamp<double>(dstPitch,minPitch,maxPitch);
}
double AltitudeKeeper::inverse(const MotionState& motion, const double& dstPitch){
    double alt=-motion.pos(2);
    double vAlt=-motion.vel(2);
    double V=motion.vel.norm();
    double eAlt=(V*sin(dstPitch)-dGain*vAlt)/pGain;
    return alt-eAlt;
}

void exportFlightControllerUtility(py::module &m)
{
    using namespace pybind11::literals;
    m.def("pitchLimitter",&pitchLimitter);
    py::class_<AltitudeKeeper>(m,"AltitudeKeeper")
    .def(py::init<>())
    .def(py::init<const nl::json&>())
    .def(py::init([](const py::object& config_){return AltitudeKeeper(config_);}))
    .def("__call__",&AltitudeKeeper::operator())
    DEF_FUNC(AltitudeKeeper,getDstPitch)
    DEF_FUNC(AltitudeKeeper,inverse)
    DEF_READWRITE(AltitudeKeeper,pGain)
    DEF_READWRITE(AltitudeKeeper,dGain)
    DEF_READWRITE(AltitudeKeeper,minPitch)
    DEF_READWRITE(AltitudeKeeper,maxPitch)
    ;
}
