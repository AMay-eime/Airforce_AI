// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "SixDoFFighter.h"
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <boost/math/tools/roots.hpp>
#include "nlopt.hpp"
#include "Utility.h"
#include "MathUtility.h"
#include "Units.h"
#include "Agent.h"
#include "SimulationManager.h"
#include "Missile.h"
#include "Sensor.h"
using namespace util;

SixDoFFighter::SixDoFFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    m=getValueFromJsonKRD(modelConfig.at("dynamics"),"m",randomGen,lb2kg(20500.0));
    S=getValueFromJsonKRD(modelConfig.at("dynamics"),"S",randomGen,ft2m(ft2m(300.0)));
    Le=getValueFromJsonKRD(modelConfig.at("dynamics"),"Le",randomGen,slugft22kgm2(160.0));
    b=getValueFromJsonKRD(modelConfig.at("dynamics"),"b",randomGen,ft2m(30.0));
    mac=getValueFromJsonKRD(modelConfig.at("dynamics"),"mac",randomGen,ft2m(11.32));
    XcgR=getValueFromJsonKRD(modelConfig.at("dynamics"),"XcgR",randomGen,0.35);
    Xcg=getValueFromJsonKRD(modelConfig.at("dynamics"),"Xcg",randomGen,0.35);
    I=getValueFromJsonKRD(modelConfig.at("dynamics"),"I",randomGen,Eigen::Matrix3d({
        {slugft22kgm2(9496.0),0.0,slugft22kgm2(982.0)},
        {0.0,slugft22kgm2(55814.0),0.0},
        {slugft22kgm2(982.0),0.0,slugft22kgm2(63100.0)}
    }));
    Iinv=I.inverse();
    de=0;
    da=0;
    dr=0;
    deLimit=getValueFromJsonKRD(modelConfig.at("dynamics"),"deLimit",randomGen,deg2rad(25.0));
    daLimit=getValueFromJsonKRD(modelConfig.at("dynamics"),"daLimit",randomGen,deg2rad(21.0));
    drLimit=getValueFromJsonKRD(modelConfig.at("dynamics"),"drLimit",randomGen,deg2rad(30.0));
    deMaxRate=getValueFromJsonKRD(modelConfig.at("dynamics"),"deMaxRate",randomGen,deg2rad(60.0));
    daMaxRate=getValueFromJsonKRD(modelConfig.at("dynamics"),"daMaxRate",randomGen,deg2rad(80.0));
    drMaxRate=getValueFromJsonKRD(modelConfig.at("dynamics"),"drMaxRate",randomGen,deg2rad(120.0));
    deTimeConstant=getValueFromJsonKRD(modelConfig.at("dynamics"),"deTimeConstant",randomGen,0.0495);
    daTimeConstant=getValueFromJsonKRD(modelConfig.at("dynamics"),"daTimeConstant",randomGen,0.0495);
    drTimeConstant=getValueFromJsonKRD(modelConfig.at("dynamics"),"drTimeConstant",randomGen,0.0495);
    cdwTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"cdw",randomGen,
        (Eigen::Matrix<double,4,1>()<<2.76e-2,1.95e-3,1.49e0,1.31e-1).finished()
    );

    motion.omega=getValueFromJsonKRD(instanceConfig,"omega",randomGen,Eigen::Vector3d(0,0,0));

}
SixDoFFighter::~SixDoFFighter(){
}
void SixDoFFighter::validate(){
    Fighter::validate();
    double alt=-motion.pos(2);
    double V=motion.vel.norm();
    double tgtBank=getValueFromJsonKRD(modelConfig.at("dynamics"),"tgtTrimBank",randomGen,deg2rad(0));
    Eigen::VectorXd trimmed=trim6(alt,V,tgtBank);
    double alpha_=trimmed[0];
    double beta_=trimmed[1];
    Quaternion q1=Quaternion::fromAngle(Eigen::Vector3d(0,0,1),motion.az);
    Quaternion q2=Quaternion::fromAngle(Eigen::Vector3d(1,0,0),tgtBank);
    Quaternion q3=Quaternion::fromAngle(Eigen::Vector3d(0,0,1),-beta_);
    Quaternion q4=Quaternion::fromAngle(Eigen::Vector3d(0,1,0),alpha_);
    motion.q=q1*q2*q3*q4;
    motion.calcQh();
    de=trimmed[2];
    da=trimmed[3];
    dr=trimmed[4];
    auto eng=getShared<SimpleFighterJetEngine>(engine);
    eng->pCmd=trimmed[5];
    eng->power=trimmed[5];
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    eng->thrust=eng->calcThrust(trimmed[5],alt,Mach);
}

double SixDoFFighter::calcOptimumCruiseFuelFlowRatePerDistance(){
    //high computational cost, so once you obtained the value, should write it in json modelConfig.
    double ffr=engine.lock()->getFuelFlowRate();
    if(ffr<=0.0){
        //no fuel consumption model
        return 0.0;
    }else{
        std::function<double(unsigned int,const double*,double*,void*)> obj=[&](unsigned int n,const double* x,double* grad,void* data){
            //x=[alt,V] in [m,m/s]
            double alt=x[0];
            double V=x[1];
            std::map<std::string,double> atm=atmosphere(alt);
            double Mach=V/atm["a"];
            Eigen::VectorXd trimmed=trim6(alt,V,0.0);//trim for level flight
            double fuelFlowRate=engine.lock()->calcFuelFlowRate({{"power",trimmed[5]}});
            if(trimmed[6]==0){//infeasible to keep steady level flight
                return engine.lock()->calcFuelFlowRate({{"power",1.0}});
            }
            return fuelFlowRate/V;//fuel consumption per distance in [kg/m]
        };
        nlopt_delegator obj_d(&obj,nullptr);
	    std::vector<double> x={10000.0,300.0};
        std::vector<double> lb={0.0,1.0};
        std::vector<double> ub={20000.0,600.0};
	    nlopt::opt opt=nlopt::opt(nlopt::algorithm::LN_NELDERMEAD,2);
        opt.set_min_objective(nlopt_delegatee,&obj_d);
        opt.set_maxeval(1e5);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        auto result=opt.optimize(x,ffr);
        std::map<std::string,double> atm=atmosphere(x[0]);
        double Mach=x[1]/atm["a"];
        std::cout<<"optimize cruise: alt="<<x[0]<<"[m], V="<<x[1]<<"[m/s] (M="<<Mach<<")"<<", ffr="<<ffr<<"[kg/m] ("<<ffr*x[1]<<"[kg/s])"<<std::endl;
        return ffr;
    }
}
Eigen::VectorXd SixDoFFighter::trim6(double alt,double V,double tgtBank){
    auto eng=getShared<SimpleFighterJetEngine>(engine);
    std::function<double(unsigned int,const double*,double*,void*)> obj=[&](unsigned int n,const double* x,double* grad,void* data){
        //x=[alpha,beta,de,da,dr,P]
        double alpha_=x[0];
        double beta_=x[1];
        double de_=x[2];
        double da_=x[3];
        double dr_=x[4];
        if(tgtBank==0){
            beta_=0;
            da_=0;
            dr_=0;
        }
        std::map<std::string,double> atm=atmosphere(alt);
        double Mach=V/atm["a"];
        double dp=atm["rho"]*V*V/2;
        double thrust_=eng->calcThrust(x[5],alt,Mach);
        Eigen::Vector3d vi(V,0,0);
        Eigen::Vector3d omegai(0,0,0);
        Quaternion q1=Quaternion::fromAngle(Eigen::Vector3d(0,0,1),0);
        Quaternion q2=Quaternion::fromAngle(Eigen::Vector3d(1,0,0),tgtBank);
        Quaternion q3=Quaternion::fromAngle(Eigen::Vector3d(0,0,1),-beta_);
        Quaternion q4=Quaternion::fromAngle(Eigen::Vector3d(0,1,0),alpha_);
        Quaternion q=q1*q2*q3*q4;
        Eigen::Vector3d ex=q.transformVector(Eigen::Vector3d(1,0,0));
        Eigen::Vector3d ey=q.transformVector(Eigen::Vector3d(0,1,0));
        Eigen::Vector3d ez=q.transformVector(Eigen::Vector3d(0,0,1));
        Eigen::Vector3d vb=q.conjugate().transformVector(vi);
        Eigen::Vector3d omegab=q.conjugate().transformVector(omegai);
        std::map<std::string,double> aero=calcAeroV(alpha_,beta_,de_,da_,dr_,Mach,false);
        double pd=omegab(0)*b/(2*V);
        double qd=omegab(1)*mac/(2*V);
        double rd=omegab(2)*b/(2*V);
        double CX=aero["Cx"]+aero["Cxq"]*qd;
        double CY=aero["Cy"]+aero["Cyp"]*pd+aero["Cyr"]*rd;
        double CZ=aero["Cz"]+aero["Czq"]*qd;
        double CL=aero["Cl"]+aero["Clp"]*pd+aero["Clr"]*rd+aero["Clda"]*da_+aero["Cldr"]*dr_;
        double CM=aero["Cm"]+aero["Cmq"]*qd+CZ*(XcgR-Xcg);
        double CN=aero["Cn"]+aero["Cnp"]*pd+aero["Cnr"]*rd+aero["Cnda"]*da_+aero["Cndr"]*dr_-CY*(XcgR-Xcg)*(mac/b);
        Eigen::Vector3d Fab=(Eigen::Vector3d(CX,CY,CZ)-aero["Cdw"]*vb/V)*dp*S;
	    Eigen::Vector3d Na=Eigen::Vector3d(CL*b,CM*mac,CN*b)*dp*S;
        Eigen::Vector3d Ftb=Eigen::Vector3d(thrust_,0,0);
        Eigen::Vector3d Fb=Fab+Ftb+m*gravity*Eigen::Vector3d(ex(2),ey(2),ez(2));
        Eigen::Vector3d omegaDotB=Iinv*(Na-omegab.cross(I*omegab+Eigen::Vector3d(Le,0.,0.)));
        Eigen::Vector3d vbDot=Fb/m-omegab.cross(vb);
        double VDot=vb.dot(vbDot)/V;
        double Vxz=sqrt(vb(0)*vb(0)+vb(2)*vb(2));
        double alphaDot=(vb(0)*vbDot(2)-vb(2)*vbDot(0))/(Vxz*Vxz);
        double betaDot=(V*vbDot(1)-vb(1)*VDot)/(V*Vxz);
        return (VDot*VDot)/(V*V)+alphaDot*alphaDot+betaDot*betaDot+omegaDotB.squaredNorm();
    };
    nlopt_delegator obj_d(&obj,nullptr);
	std::vector<double> trimmed={0,0,0,0,0,0.5};
    std::vector<double> lb={-deg2rad(10),-deg2rad(30),-deLimit,-daLimit,-drLimit,0};
    std::vector<double> ub={deg2rad(45),deg2rad(30),deLimit,daLimit,drLimit,1};
	nlopt::opt opt=nlopt::opt(nlopt::algorithm::LN_BOBYQA,6);
    opt.set_min_objective(nlopt_delegatee,&obj_d);
    opt.set_stopval(1e-10);
    opt.set_maxeval(1e4);
    double minf;
    try{
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        auto result=opt.optimize(trimmed,minf);
    }catch(...){
        if(std::isnan(trimmed[0])){trimmed[0]=0;}
        if(std::isnan(trimmed[1])){trimmed[1]=0;}
        if(std::isnan(trimmed[2])){trimmed[2]=0;}
        if(std::isnan(trimmed[3])){trimmed[3]=0;}
        if(std::isnan(trimmed[4])){trimmed[4]=0;}
        if(std::isnan(trimmed[5])){trimmed[5]=0.5;}
        minf=std::numeric_limits<double>::infinity();
    }
    Eigen::VectorXd ret(7);
    if(tgtBank==0){
        trimmed[1]=0;//beta
        trimmed[3]=0;//da
        trimmed[4]=0;//dr
    }
    ret<<trimmed[0],trimmed[1],trimmed[2],trimmed[3],trimmed[4],trimmed[5],(minf<1e-10 ? 1.0 : 0.0);
    return ret;
}
std::map<std::string,double> SixDoFFighter::calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives){
    std::map<std::string,double> ret;
    throw std::runtime_error("You must override SixDofFighter::calcAeroV in your derived class.");
    return std::move(ret);
}
Eigen::VectorXd SixDoFFighter::calcDerivative(const Eigen::VectorXd& x,const Eigen::VectorXd& u){
    //x,uの読み込み
    MotionState tmp_motion(
        x.block(0,0,3,1),
        x.block(3,0,3,1),
        x.block(10,0,3,1),
        Quaternion(x(6),x(7),x(8),x(9)).normalized(),
        0.0
    );
    double tmp_de=std::clamp<double>(x(13),-deLimit,deLimit);
    double tmp_da=std::clamp<double>(x(14),-daLimit,daLimit);
    double tmp_dr=std::clamp<double>(x(15),-drLimit,drLimit);
    double tmp_power=std::clamp<double>(x(16),0,1);
    double ue=u(0);
    double ua=u(1);
    double ur=u(2);
    double pCmd=u(3);

    //基本の状態量に関する値
    Eigen::Vector3d ri=tmp_motion.pos;
    Eigen::Vector3d vi=tmp_motion.vel;
    Eigen::Vector3d vb=tmp_motion.relPtoB(vi);
    double V=tmp_motion.vel.norm();
    Eigen::Vector3d vni=vi/V;
    Eigen::Vector3d omegai=tmp_motion.omega;
    Eigen::Matrix<double,4,3> Jq=tmp_motion.q.dqdwi();
    //大気,動圧に関する値
    double alt=-ri(2);
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    double dp=atm["rho"]*V*V/2;
    double K=dp*S;
    //α,βに関する値
    double alpha=atan2(vb(2),vb(0));
    double beta=asin(vb(1)/V);
    //力、モーメントに関する値
    //空気力
    double alphaClamp=std::clamp<double>(alpha,-deg2rad(10),deg2rad(45));
    double betaClamp=std::clamp<double>(beta,-deg2rad(30),deg2rad(30));
    std::map<std::string,double> aero=calcAeroV(alphaClamp,betaClamp,tmp_de,tmp_da,tmp_dr,Mach,false);
    Eigen::Vector3d omegaB=tmp_motion.relPtoB(omegai);
    double pd=omegaB(0)*b/(2*V);
    double qd=omegaB(1)*mac/(2*V);
    double rd=omegaB(2)*b/(2*V);
    double CX=aero["Cx"]+aero["Cxq"]*qd;
    double CY=aero["Cy"]+aero["Cyp"]*pd+aero["Cyr"]*rd;
    double CZ=aero["Cz"]+aero["Czq"]*qd;
    double CL=aero["Cl"]+aero["Clp"]*pd+aero["Clr"]*rd+aero["Clda"]*tmp_da+aero["Cldr"]*tmp_dr;
    double CM=aero["Cm"]+aero["Cmq"]*qd+CZ*(XcgR-Xcg);
    double CN=aero["Cn"]+aero["Cnp"]*pd+aero["Cnr"]*rd+aero["Cnda"]*tmp_da+aero["Cndr"]*tmp_dr-CY*(XcgR-Xcg)*(mac/b);
    Eigen::Vector3d CFa(CX,CY,CZ);
    Eigen::Vector3d CNa(CL,CM,CN);
    Eigen::Matrix<double,3,3> bcbMat=Eigen::Vector3d(b,mac,b).asDiagonal().toDenseMatrix();
    //推力
    auto eng=getShared<SimpleFighterJetEngine>(engine);
    double thrust=eng->calcThrust(tmp_power,alt,Mach);
    //状態方程式
    Eigen::Vector3d Fi=(tmp_motion.relBtoP(CFa)-aero["Cdw"]*vni)*K+tmp_motion.relBtoP(Eigen::Vector3d(thrust,0,0))+Eigen::Vector3d(0,0,m*gravity);
    Eigen::Vector3d Nb=bcbMat*CNa*K;
    Eigen::Vector3d Lb=I*omegaB+Eigen::Vector3d(Le,0.,0.);
    Eigen::Vector3d omegaDotB=Iinv*(Nb-omegaB.cross(Lb));
    Eigen::Vector3d omegaDotI=tmp_motion.relBtoP(omegaDotB);
    Eigen::VectorXd f=Eigen::VectorXd::Zero(17);
    f.block(0,0,3,1)=vi;
    f.block(3,0,3,1)=Fi/m;
    f.block(6,0,4,1)=Jq*omegai;
    f.block(10,0,3,1)=omegaDotI;
    f(13,0)=std::clamp<double>((ue*deLimit-tmp_de)/deTimeConstant,-deMaxRate,deMaxRate);
    f(14,0)=std::clamp<double>((ua*daLimit-tmp_da)/daTimeConstant,-daMaxRate,daMaxRate);
    f(15,0)=std::clamp<double>((ur*drLimit-tmp_dr)/drTimeConstant,-drMaxRate,drMaxRate);
    if(eng->enableInstantResponse){
        f(16,0)=0;
    }else{
        f(16,0)=eng->pdot(tmp_power,pCmd);
    }
    return f;
}
void SixDoFFighter::calcMotion(double dt){
    vel_prev=motion.vel;
    pos_prev=motion.pos;
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    auto eng=getShared<SimpleFighterJetEngine>(engine);
    Eigen::VectorXd x0=Eigen::VectorXd::Zero(17);
    x0.block(0,0,3,1)=motion.pos;
    x0.block(3,0,3,1)=motion.vel;
    x0.block(6,0,4,1)=motion.q.toArray();
    x0.block(10,0,3,1)=motion.omega;
    x0(13)=de;
    x0(14)=da;
    x0(15)=dr;
    x0(16)=eng->power;
    Eigen::VectorXd u=Eigen::VectorXd::Zero(4);
    u(0)=ctrl.at("ue").get<double>();
    u(1)=ctrl.at("ua").get<double>();
    u(2)=ctrl.at("ur").get<double>();
    u(3)=ctrl.at("pCmd").get<double>();
    std::function<Eigen::VectorXd (const Eigen::VectorXd& x)> deriv=[&](const Eigen::VectorXd& x){
        return calcDerivative(x,u);
    };
    Eigen::VectorXd nextX=RK4(x0,dt,deriv);
    motion.pos=nextX.block(0,0,3,1);
    motion.vel=nextX.block(3,0,3,1);
    motion.omega=nextX.block(10,0,3,1);
    motion.q=Quaternion(nextX(6),nextX(7),nextX(8),nextX(9)).normalized();
    de=std::clamp<double>(nextX(13),-deLimit,deLimit);
    da=std::clamp<double>(nextX(14),-daLimit,daLimit);
    dr=std::clamp<double>(nextX(15),-drLimit,drLimit);
    motion.time=manager->getTime()+dt;
    motion.calcQh();
    //α、βが許容範囲外に出た場合も墜落として扱う
    Eigen::Vector3d vb=motion.relPtoB(motion.vel);
    double V=vb.norm();
    double viex=vb(0);
    double viey=vb(1);
    double viez=vb(2);
    double alpha=atan2(viez,viex);
    double beta=asin(viey/V);
    if(alpha<deg2rad(-10.0) || 
        alpha>deg2rad(45.0) || 
        beta<deg2rad(-30.0) || 
        beta>deg2rad(30.0)
    ){
        if(motion.pos(2)<=0){
            //高度が0以上の場合のみここで通知(高度0未満の場合は親クラスで通知されるため)
            manager->triggerEvent("Crash",this->weak_from_this());
            manager->requestToKillAsset(getShared<Asset>(this->shared_from_this()));
            //std::cout<<"Crashed! alpha="<<rad2deg(alpha)<<", beta="<<rad2deg(beta)<<std::endl;
        }
    }
}

SixDoFFighter::FlightController::FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter::FlightController(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    mode="fromDirAndVel";
    PositiveNzLimit=getValueFromJsonKRD(modelConfig,"PositiveNzLimit",randomGen,10.0);
    NegativeNzLimit=getValueFromJsonKRD(modelConfig,"NegativeNzLimit",randomGen,2.0);
    maxNyrCmd=getValueFromJsonKRD(modelConfig,"maxNyrCmd",randomGen,2.0);
    maxPsCmd=getValueFromJsonKRD(modelConfig,"maxPsCmd",randomGen,2.0);
    kPsCmdP=getValueFromJsonKRD(modelConfig,"kPsCmdP",randomGen,0.5);
    kPsCmdD=getValueFromJsonKRD(modelConfig,"kPsCmdP",randomGen,-2.5);
    kNzCmdP=getValueFromJsonKRD(modelConfig,"kNzCmdP",randomGen,10.0);
    kNzCmdD=getValueFromJsonKRD(modelConfig,"kNzCmdD",randomGen,-2.5);
    kNyrCmdP=getValueFromJsonKRD(modelConfig,"kNyrCmdP",randomGen,0.5);
    kNyrCmdD=getValueFromJsonKRD(modelConfig,"kNyrCmdD",randomGen,-0.5);
    if(modelConfig.contains("altitudeKeeper")){
        nl::json sub=modelConfig.at("altitudeKeeper");
        altitudeKeeper=AltitudeKeeper(sub);
    }
}
nl::json SixDoFFighter::FlightController::getDefaultCommand(){
    return {
        {"ue",0.0},//-1〜1
        {"ua",0.0},//-1〜1
        {"ur",0.0},//-1〜1
        {"pCmd",0.0}//0〜1
    };
}
nl::json SixDoFFighter::FlightController::calc(const nl::json &cmd){
    if(mode=="fromDirAndVel"){
        return calcFromDirAndVel(cmd);
    }else if(mode=="fromManualInput"){
        return calcFromManualInput(cmd);
    }else{
        return calcDirect(cmd);
    }
}
nl::json SixDoFFighter::FlightController::calcDirect(const nl::json &cmd){
    //操舵量を直接指定する場合
    auto p=getShared<SixDoFFighter>(parent);
    double alt=-p->motion.pos(2);
    double V=p->motion.vel.norm();
    double tgtBank=0;
    Eigen::VectorXd u;
    Eigen::VectorXd trimmed=p->trim6(alt,V,tgtBank);
    u=Eigen::VectorXd::Zero(4);
    double roll_=0.0;
    if(cmd.contains("roll")){
        roll_=std::clamp(cmd.at("roll").get<double>(),-1.,1.);
    }
    double pitch_=0.0;
    if(cmd.contains("pitch")){
        pitch_=std::clamp(cmd.at("pitch").get<double>(),-1.,1.);
    }
    double yaw_=0.0;
    if(cmd.contains("yaw")){
        yaw_=std::clamp(cmd.at("yaw").get<double>(),-1.,1.);
    }
    double pCmd_=trimmed[5];
    if(cmd.contains("throttle")){//0〜1
        std::clamp(cmd.at("throttle").get<double>(),0.,1.);
    }else if(cmd.contains("accel")){//-1〜+1
        double clampedCmd=std::clamp(cmd.at("accel").get<double>(),-1.,1.);
        if(clampedCmd>=0){
            pCmd_=pCmd_+(1.0-pCmd_)*clampedCmd;
        }else{
            pCmd_=pCmd_*(1+clampedCmd);
        }
    }
    u<<-pitch_,-roll_,-yaw_,pCmd_;
    return {
        {"ue",std::clamp<double>(u(0),-1,1)},
        {"ua",std::clamp<double>(u(1),-1,1)},
        {"ur",std::clamp<double>(u(2),-1,1)},
        {"pCmd",std::clamp<double>(u(3),0,1)}
    };
}

nl::json SixDoFFighter::FlightController::calcFromManualInput(const nl::json &cmd){
    //スティック、ラダー、スロットルの入力で指定したものを操舵量に変換する場合
    auto p=getShared<SixDoFFighter>(parent);
    auto eng=getShared<SimpleFighterJetEngine>(p->engine);
    Eigen::VectorXd x=Eigen::VectorXd::Zero(17);
    x.block(0,0,3,1)=p->motion.pos;
    x.block(3,0,3,1)=p->motion.vel;
    x.block(6,0,4,1)=p->motion.q.toArray();
    x.block(10,0,3,1)=p->motion.omega;
    x(13)=p->de;
    x(14)=p->da;
    x(15)=p->dr;
    x(16)=eng->power;
    nl::json DIcmd;
    if(modelConfig.contains("defaultCmd")){
        DIcmd=modelConfig.at("defaultCmd");
    }else{
        DIcmd=nl::json::array();
    }
    double alt=-p->motion.pos(2);
    double V=p->motion.vel.norm();
    double tgtBank=0;
    Eigen::VectorXd u;
    nl::json commandParser=modelConfig.at("commandParser");
    nl::json elem;
    double roll_=0.0;
    if(cmd.contains("roll")){
        roll_=std::clamp(cmd.at("roll").get<double>(),-1.,1.)*maxPsCmd;
    }
    double pitch_=0.0;
    if(cmd.contains("pitch")){
        pitch_=std::clamp(cmd.at("pitch").get<double>(),-1.,1.);
    }
    if(pitch_<0){
        pitch_*=NegativeNzLimit;
    }else{
        pitch_*=PositiveNzLimit;
    }
    double yaw_=0.0;
    if(cmd.contains("yaw")){
        yaw_=std::clamp(cmd.at("yaw").get<double>(),-1.,1.)*maxNyrCmd;
    }
    if(cmd.contains("throttle")){//0〜1
        if(commandParser.contains("@dstThrottle")){
            elem=commandParser.at("@dstThrottle");
            elem["type"]="throttle(power)";
            elem["target"]=std::clamp(cmd.at("throttle").get<double>(),0.,1.);
            DIcmd.push_back(elem);
        }
    }else if(cmd.contains("accel")){//-1〜+1
        if(commandParser.contains("@dstAccel")){
            double accelLimit=2;
            elem=commandParser.at("@dstAccel");
            elem["type"]="acceleration(scalar)";
            elem["target"]=std::clamp(cmd.at("accel").get<double>(),-1.,1.*accelLimit);
            DIcmd.push_back(elem);
        }
    }else{
        throw std::runtime_error("Either 'throttle' or 'accel' must be in cmd keys.");
    }
    for(auto&& item:commandParser.items()){
        std::string key=item.key();
        nl::json elem=item.value();
        if(key=="@OmegaB"){
            elem["type"]="angular velocity(body)";
            elem["target"]=Eigen::Vector3d::Zero();
            DIcmd.push_back(elem);
        }else if(key=="@Ps"){
            elem["type"]="Ps";
            elem["target"]=roll_;
            elem["useIntegral"]=false;
            elem["integralKey"]="Ps";
            DIcmd.push_back(elem);
        }else if(key=="@Nz"){
            elem["type"]="Nz";
            elem["target"]=pitch_;
            elem["useIntegral"]=false;
            elem["integralKey"]="Nz";
            DIcmd.push_back(elem);
        }else if(key=="@Nyr"){
            elem["type"]="Ny+r";
            elem["target"]=yaw_;
            elem["useIntegral"]=false;
            elem["integralKey"]="Ny+r";
            DIcmd.push_back(elem);
        }else if(key=="@NegativeAlphaLimitter"){
            elem["type"]="AlphaLimitter";
            elem["oneSideOnly"]="negative";
            DIcmd.push_back(elem);
        }else if(key=="@PositiveAlphaLimitter"){
            elem["type"]="AlphaLimitter";
            elem["oneSideOnly"]="positive";
            DIcmd.push_back(elem);
        }else if(key=="@AlphaDotLimitter"){
            elem["type"]="AlphaDotLimitter";
            elem["target"]=0;
            DIcmd.push_back(elem);
        }
    }
    u=calcU(DIcmd,x);
    return {
        {"ue",std::clamp<double>(u(0),-1,1)},
        {"ua",std::clamp<double>(u(1),-1,1)},
        {"ur",std::clamp<double>(u(2),-1,1)},
        {"pCmd",std::clamp<double>(u(3),0,1)}
    };
}

nl::json SixDoFFighter::FlightController::calcFromDirAndVel(const nl::json &cmd){
    //目標とする方向や高度、速度等で指定したものを操舵量に変換する場合
    auto p=getShared<SixDoFFighter>(parent);
    double alt=-p->motion.pos(2);
    double V=p->motion.vel.norm();
    double tgtBank=0;
    auto eng=getShared<SimpleFighterJetEngine>(p->engine);
    Eigen::VectorXd x=Eigen::VectorXd::Zero(17);
    x.block(0,0,3,1)=p->motion.pos;
    x.block(3,0,3,1)=p->motion.vel;
    x.block(6,0,4,1)=p->motion.q.toArray();
    x.block(10,0,3,1)=p->motion.omega;
    x(13)=p->de;
    x(14)=p->da;
    x(15)=p->dr;
    x(16)=eng->power;
    nl::json DIcmd=nl::json::array();
    nl::json commandParser=modelConfig.at("commandParser");
    nl::json elem;
    if(cmd.contains("dstV")){
        if(commandParser.contains("@dstV")){
            elem=commandParser.at("@dstV");
            elem["type"]="velocity(scalar)";
            elem["target"]=cmd.at("dstV");
            DIcmd.push_back(elem);
        }
    }else if(cmd.contains("dstAccel")){
        if(commandParser.contains("@dstAccel")){
            elem=commandParser.at("@dstAccel");
            elem["type"]="acceleration(scalar)";
            elem["target"]=cmd.at("dstAccel");
            DIcmd.push_back(elem);
        }
    }else if(cmd.contains("dstThrust")){
        std::map<std::string,double> atm=atmosphere(alt);
        double Mach=V/atm["a"];
        double Tmin=eng->calcThrust(0.0,alt,Mach);
	    double Tmax=eng->calcThrust(1.0,alt,Mach);
        if(commandParser.contains("@dstThrust")){
            elem=commandParser.at("@dstThrust");
            elem["type"]="thrust";
            elem["target"]=std::clamp(cmd.at("dstThrust").get<double>(),Tmin,Tmax);
            DIcmd.push_back(elem);
        }
    }else if(cmd.contains("dstThrottle")){
        if(commandParser.contains("@dstThrottle")){
            elem=commandParser.at("@dstThrottle");
            elem["type"]="throttle(power)";
            elem["target"]=std::clamp(cmd.at("dstThrottle").get<double>(),0.,1.);
            DIcmd.push_back(elem);
        }
    }else{
		throw std::runtime_error("Only one of dstV, dstAccel, dstThrust or dstThrottle is acceptable.");
    }
    double eps=1e-4;
    Eigen::Vector3d b,n;
    double PsCmd,NyrCmd,NzCmd;
	if(cmd.contains("dstDir")){
		Eigen::Vector3d dstDir=cmd.at("dstDir");
        if(cmd.contains("dstAlt")){
            dstDir=altitudeKeeper(p->motion,dstDir,cmd.at("dstAlt").get<double>());
        }
		dstDir.normalize();
        Eigen::Vector3d omegaB=p->motion.relPtoB(p->motion.omega);
        Eigen::Vector3d ex=p->motion.relBtoP(Eigen::Vector3d(1,0,0));
        Eigen::Vector3d ey=p->motion.relBtoP(Eigen::Vector3d(0,1,0));
        Eigen::Vector3d eyh=Eigen::Vector3d(0,0,1).cross(ex).normalized();
        double dstAz=atan2(dstDir(1),dstDir(0));
        double vAz=atan2(p->motion.vel(1),p->motion.vel(0));
        double vEl=asin(p->motion.vel(2)/V);
        double dAz=dstAz-vAz;
        while(dAz>M_PI){dAz-=2*M_PI;}
        while(dAz<=-M_PI){dAz+=2*M_PI;}
        Eigen::Vector3d vb=p->motion.relPtoB(p->motion.vel);
        Eigen::Vector3d db=p->motion.relPtoB(dstDir);
        if(abs(dAz)>deg2rad(90.0) && vEl>=deg2rad(0.0) && alt<6000){
            double mAz = dAz>0 ? deg2rad(90.0) : -deg2rad(90.0);
            double dxy=sqrt(dstDir(0)*dstDir(0)+dstDir(1)*dstDir(1));
            Eigen::Vector3d modified(cos(vAz+mAz)*dxy,sin(vAz+mAz)*dxy,dstDir(2));
            db=p->motion.relPtoB(modified);
        }else if(abs(dAz)>deg2rad(45.0) && vEl>=deg2rad(0.0) && alt<3000){
            double mAz = dAz>0 ? deg2rad(45.0) : -deg2rad(45.0);
            double dxy=sqrt(dstDir(0)*dstDir(0)+dstDir(1)*dstDir(1));
            Eigen::Vector3d modified(cos(vAz+mAz)*dxy,sin(vAz+mAz)*dxy,dstDir(2));
            db=p->motion.relPtoB(modified);
        }else if(alt<500 && vEl>=deg2rad(0.0)){
            double dxy=sqrt(dstDir(0)*dstDir(0)+dstDir(1)*dstDir(1));
            Eigen::Vector3d modified(cos(vAz)*dxy,sin(vAz)*dxy,dstDir(2));
            db=p->motion.relPtoB(modified);
        }
        Eigen::Vector3d axb=vb.cross(db);
        double dRoll,dPitch,dYaw;
        if(axb.norm()<1e-6){
            //ほとんど並行
            if(vb.dot(db)>0){
                //今の向きのままでよい
                PsCmd=0.0;
                NyrCmd=0.0;
                NzCmd=0.0;
                axb<<0,1,0;
                dRoll=0.0;
            }else{
                //180度反転
                //回転軸は回転軸がなるべく水平になるようにとる
                axb=ex.cross(eyh).normalized();
                axb.normalize();
                dRoll=atan2(axb(2),axb(1));
            }
        }else{
            double snTheta=std::min(1.0,axb.norm());
            double csTheta=std::clamp(vb.dot(db),-1.0,1.0);
            double eTheta=atan2(snTheta,csTheta);
            axb.normalize();
            dRoll=atan2(axb(2),axb(1));
        }
        dPitch=atan2(-db(2),db(0))-atan2(-vb(2),vb(0));
        while(dPitch>M_PI){dPitch-=2*M_PI;}
        while(dPitch<=-M_PI){dPitch+=2*M_PI;}
        dYaw=atan2(db(1),db(0))-atan2(vb(1),vb(0));
        while(dYaw>M_PI){dYaw-=2*M_PI;}
        while(dYaw<=-M_PI){dYaw+=2*M_PI;}
        NyrCmd=kNyrCmdP*dYaw+kNyrCmdD*omegaB(2);
        NzCmd=kNzCmdP*dPitch+kNzCmdD*omegaB(1);
        if(NzCmd>=-NegativeNzLimit && 
            NyrCmd>-maxNyrCmd && NyrCmd<maxNyrCmd
        ){
                //ピッチとヨーでカバーし得る範囲ならばロールを抑制
                axb=p->motion.relPtoB(eyh);
                dRoll=atan2(axb(2),axb(1));
        }
        NzCmd=std::clamp<double>(NzCmd,-NegativeNzLimit,PositiveNzLimit);
        NyrCmd=std::clamp<double>(NyrCmd,-maxNyrCmd,maxNyrCmd);
        PsCmd=kPsCmdP*dRoll+kPsCmdD*omegaB(0);
        PsCmd=std::clamp<double>(PsCmd,-maxPsCmd,maxPsCmd);
    }else if(cmd.contains("dstTurnRate")){
		Eigen::Vector3d dstTurnRate=cmd.at("dstTurnRate");
        if(dstTurnRate.norm()<1e-4){
            //Almost no rotation
            NzCmd=0;
            PsCmd=0;
            NyrCmd=0;
        }else{
            Eigen::Vector3d omegaB=p->motion.relPtoB(p->motion.omega);
            Eigen::Vector3d dstAccelDir=(dstTurnRate.cross(p->motion.vel)).normalized();
            double dstAccel=V*dstTurnRate.dot(p->motion.vel.cross(dstAccelDir).normalized());
            Eigen::Vector3d dstAccelDirB=p->motion.relPtoB(dstAccelDirB);
            double dRoll=atan2(-dstAccelDirB(1),-dstAccelDirB(2));
            PsCmd=kPsCmdP*dRoll+kPsCmdD*omegaB(0);
            PsCmd=std::clamp<double>(PsCmd,-maxPsCmd,maxPsCmd);
            NzCmd=(dstAccel/gravity)*cos(dRoll);
            NzCmd=std::clamp<double>(NzCmd,-NegativeNzLimit,PositiveNzLimit);
            NyrCmd=-(dstAccel/gravity)*sin(dRoll);
            NyrCmd=std::clamp<double>(NyrCmd,-maxNyrCmd,maxNyrCmd);
        }
    }else{
        throw std::runtime_error("Only one of dstDir or dstTurnRate is acceptable.");
    }
    if(commandParser.contains("@Ps")){
        elem=commandParser.at("@Ps");
        elem["type"]="Ps";
        elem["target"]=PsCmd;
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@Nyr")){
        elem=commandParser.at("@Nyr");
        elem["type"]="Ny+r";
        elem["target"]=NyrCmd;
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@Nz")){
        elem=commandParser.at("@Nz");
        elem["type"]="Nz";
        elem["target"]=NzCmd;
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@OmegaB")){
        elem=commandParser.at("@OmegaB");
        elem["type"]="angular velocity(body)";
        elem["target"]=Eigen::Vector3d::Zero();
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@NegativeAlphaLimitter")){
        elem=commandParser.at("@NegativeAlphaLimitter");
        elem["type"]="AlphaLimitter";
        elem["oneSideOnly"]="negative";
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@PositiveAlphaLimitter")){
        elem=commandParser.at("@PositiveAlphaLimitter");
        elem["type"]="AlphaLimitter";
        elem["oneSideOnly"]="positive";
        DIcmd.push_back(elem);
    }
    if(commandParser.contains("@AlphaDotLimitter")){
        elem=commandParser.at("@AlphaDotLimitter");
        elem["type"]="AlphaDotLimitter";
        elem["target"]=0;
        DIcmd.push_back(elem);
    }
    Eigen::VectorXd u=calcU(DIcmd,x);
    return {
        {"ue",std::clamp<double>(u(0),-1,1)},
        {"ua",std::clamp<double>(u(1),-1,1)},
        {"ur",std::clamp<double>(u(2),-1,1)},
        {"pCmd",std::clamp<double>(u(3),0,1)}
    };
}
Eigen::VectorXd SixDoFFighter::FlightController::calcU(const nl::json &cmd,const Eigen::VectorXd& x){
    //Dynamic InversionとLinear Quadratic Regulatorで操舵量を計算
    auto p=getShared<SixDoFFighter>(parent);
    //xの読み込み
    MotionState motion(
        x.block(0,0,3,1),
        x.block(3,0,3,1),
        x.block(10,0,3,1),
        Quaternion(x(6),x(7),x(8),x(9)),
        0.0
    );
    double de=x(13);
    double da=x(14);
    double dr=x(15);
    double power=x(16);
    //基本の状態量に関する値
    Eigen::Vector3d ri=motion.pos;
    Eigen::Vector3d vi=motion.vel;
    Eigen::Vector3d vb=motion.relPtoB(vi);
    double V=motion.vel.norm();
    double Vxz=sqrt(vb(0)*vb(0)+vb(2)*vb(2));
    Eigen::Vector3d vni=vi/V;
    Eigen::Matrix<double,3,3> dvnidvi=(Eigen::Matrix3d::Identity()-vni*vni.transpose())/V;
    Eigen::Vector3d omegai=motion.omega;
    Quaternion q=motion.q;
    Eigen::Matrix<double,4,3> Jq=motion.q.dqdwi();
    Eigen::Matrix<double,3,3> Rbi=motion.q.toRotationMatrix();
    Eigen::Vector3d ex=Rbi.block(0,0,3,1);
    Eigen::Matrix<double,3,4> dexdq=motion.q.dC1dq(); 
    Eigen::Vector3d ey=Rbi.block(0,1,3,1);
    Eigen::Matrix<double,3,4> deydq=motion.q.dC2dq(); 
    Eigen::Vector3d ez=Rbi.block(0,2,3,1);
    Eigen::Matrix<double,3,4> dezdq=motion.q.dC3dq(); 
    Eigen::Vector3d exd=Rbi.block(0,0,1,3).transpose();
    Eigen::Matrix<double,3,4> dexddq=motion.q.dR1dq(); 
    Eigen::Vector3d eyd=Rbi.block(1,0,1,3).transpose();
    Eigen::Matrix<double,3,4> deyddq=motion.q.dR2dq(); 
    Eigen::Vector3d ezd=Rbi.block(2,0,1,3).transpose();
    Eigen::Matrix<double,3,4> dezddq=motion.q.dR3dq(); 
    Eigen::Matrix<double,3,3> domegabdomegai=Rbi.transpose();
    Eigen::Matrix<double,3,4> domegabdq=Eigen::Matrix<double,3,4>::Zero();
    domegabdq.block(0,0,1,4)=omegai.transpose()*dexdq;
    domegabdq.block(1,0,1,4)=omegai.transpose()*deydq;
    domegabdq.block(2,0,1,4)=omegai.transpose()*dezdq;
    Eigen::Matrix<double,3,17> domegabdx=Eigen::Matrix<double,3,17>::Zero();
    domegabdx.block(0,10,3,3)=domegabdomegai;
    domegabdx.block(0,6,3,4)=domegabdq;
    //大気,動圧に関する値
    double alt=-ri(2);
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    Eigen::Matrix<double,1,17> dMdx=Eigen::Matrix<double,1,17>::Zero();
    dMdx(0,2)=Mach/atm["a"]*atm["dadH"];
    dMdx.block(0,3,1,3)=vni.transpose()/atm["a"];
    double dp=atm["rho"]*V*V/2;
    double K=atm["rho"]*V*V*p->S*0.5;
    Eigen::Matrix<double,1,17> dKdx=Eigen::MatrixXd::Zero(1,17);
    dKdx(0,2)=-atm["dRhodH"]*V*V*p->S*0.5;
    dKdx.block(0,3,1,3)=atm["rho"]*vi.transpose()*p->S;
    //α,βに関する値
    double viex=vb(0);//vi.dot(ex);
    double viey=vb(1);//vi.dot(ey);
    double viez=vb(2);//vi.dot(ez);
    double alpha=atan2(viez,viex);
    double beta=asin(viey/V);
    Eigen::Matrix<double,1,17> dAdx=Eigen::MatrixXd::Zero(1,17);
    dAdx.block(0,3,1,3)=viex*ez.transpose()-viez*ex.transpose();
    dAdx.block(0,6,1,4)=vi.transpose()*(viex*dezdq-viez*dexdq);
    dAdx/=(Vxz*Vxz);//viex*viex+viez*viez;
    Eigen::Matrix<double,1,17> dBdx=Eigen::MatrixXd::Zero(1,17);
    dBdx.block(0,3,1,3)=ey.transpose()*dvnidvi;
    dBdx.block(0,6,1,4)=vni.transpose()*deydq;
    dBdx/=(Vxz/V);//sqrt(1-pow(viey/V,2));
    //力、モーメントに関する値
    //空気力
    double alphaClamp=std::clamp<double>(alpha,-deg2rad(10),deg2rad(45));
    double betaClamp=std::clamp<double>(beta,-deg2rad(30),deg2rad(30));
    std::map<std::string,double> aeroV=p->calcAeroV(alphaClamp,betaClamp,de,da,dr,Mach,true);
    Eigen::Vector3d omegaB=motion.relPtoB(omegai);
    double pd=omegaB(0)*p->b/(2*V);//omegai.dot(ex)
    double qd=omegaB(1)*p->mac/(2*V);//omegai.dot(ey)
    double rd=omegaB(2)*p->b/(2*V);//omegai.dot(ez)
    double CX=aeroV["Cx"]+aeroV["Cxq"]*qd;
    double dCXdA=aeroV["dCxdA"]+aeroV["dCxqdA"]*qd;
    double dCXdB=aeroV["dCxdB"]+aeroV["dCxqdB"]*qd;
    double CY=aeroV["Cy"]+aeroV["Cyp"]*pd+aeroV["Cyr"]*rd;
    double dCYdA=aeroV["dCydA"]+aeroV["dCypdA"]*pd+aeroV["dCyrdA"]*rd;
    double dCYdB=aeroV["dCydB"]+aeroV["dCypdB"]*pd+aeroV["dCyrdB"]*rd;
    double CZ=aeroV["Cz"]+aeroV["Czq"]*qd;
    double dCZdA=aeroV["dCzdA"]+aeroV["dCzqdA"]*qd;
    double dCZdB=aeroV["dCzdB"]+aeroV["dCzqdB"]*qd;
    double CL=aeroV["Cl"]+aeroV["Clp"]*pd+aeroV["Clr"]*rd+aeroV["Clda"]*da+aeroV["Cldr"]*dr;
    double dCLdA=aeroV["dCldA"]+aeroV["dClpdA"]*pd+aeroV["dClrdA"]*rd+aeroV["dCldadA"]*da+aeroV["dCldrdA"]*dr;
    double dCLdB=aeroV["dCldB"]+aeroV["dClpdB"]*pd+aeroV["dClrdB"]*rd+aeroV["dCldadB"]*da+aeroV["dCldrdA"]*dr;
    double CM=aeroV["Cm"]+aeroV["Cmq"]*qd+CZ*(p->XcgR-p->Xcg);
    double dCMdA=aeroV["dCmdA"]+aeroV["dCmqdA"]*qd+dCZdA*(p->XcgR-p->Xcg);
    double dCMdB=aeroV["dCmdB"]+aeroV["dCmqdB"]*qd+dCZdB*(p->XcgR-p->Xcg);
    double CN=aeroV["Cn"]+aeroV["Cnp"]*pd+aeroV["Cnr"]*rd+aeroV["Cnda"]*da+aeroV["Cndr"]*dr-CY*(p->XcgR-p->Xcg)*(p->mac/p->b);
    double dCNdA=aeroV["dCndA"]+aeroV["dCnpdA"]*pd+aeroV["dCnrdA"]*rd+aeroV["dCndadA"]*da+aeroV["dCndrdA"]*dr-dCYdA*(p->XcgR-p->Xcg)*(p->mac/p->b);
    double dCNdB=aeroV["dCndB"]+aeroV["dCnpdB"]*pd+aeroV["dCnrdB"]*rd+aeroV["dCndadB"]*da+aeroV["dCndrdB"]*dr-dCYdB*(p->XcgR-p->Xcg)*(p->mac/p->b);
    Eigen::Vector3d CFa(CX,CY,CZ);
    Eigen::Vector3d dCFadA(dCXdA,dCYdA,dCZdA);
    Eigen::Vector3d dCFadB(dCXdB,dCYdB,dCZdB);
    Eigen::Vector3d CNa(CL,CM,CN);
    Eigen::Vector3d dCNadA(dCLdA,dCMdA,dCNdA);
    Eigen::Vector3d dCNadB(dCLdB,dCMdB,dCNdB);
    Eigen::Matrix<double,3,17> dCFadx=Eigen::Matrix<double,3,17>::Zero();
    Eigen::Matrix<double,3,17> dCNadx=Eigen::Matrix<double,3,17>::Zero();
    Eigen::Matrix<double,3,3> bcbMat=Eigen::Vector3d(p->b,p->mac,p->b).asDiagonal().toDenseMatrix();
    Eigen::Matrix<double,1,3> vi_VV=vi.transpose()/pow(V,2);
    double b_2V=p->b/(2*V);
    double c_2V=p->mac/(2*V);
    dCFadx.block(0,3,1,3)=-aeroV["Cxq"]*qd*vi_VV;//dCFax/dv
    dCFadx.block(1,3,1,3)=-(aeroV["Cyp"]*pd+aeroV["Cyr"]*rd)*vi_VV;//dCFay/dv
    dCFadx.block(2,3,1,3)=-aeroV["Czq"]*qd*vi_VV;//dCFaz/dv
    dCFadx.block(0,6,1,4)=c_2V*aeroV["Cxq"]*omegai.transpose()*deydq;//dCFax/dq
    dCFadx.block(1,6,1,4)=b_2V*omegai.transpose()*(aeroV["Cyp"]*dexdq+aeroV["Cyr"]*dezdq);//dCFay/dq
    dCFadx.block(2,6,1,4)=c_2V*aeroV["Czq"]*omegai.transpose()*deydq;//dCFaz/dq
    dCFadx.block(0,10,1,3)=c_2V*aeroV["Cxq"]*ey.transpose();//dCFax/domega
    dCFadx.block(1,10,1,3)=b_2V*(aeroV["Cyp"]*ex.transpose()+aeroV["Cyr"]*ez.transpose());//dCFax/domega
    dCFadx.block(2,10,1,3)=c_2V*aeroV["Czq"]*ey.transpose();//dCFaz/domega
    dCFadx(0,13)=aeroV["dCxdde"];
    dCFadx(2,13)=aeroV["dCzdde"];
    dCFadx(1,14)=aeroV["dCydda"];
    dCFadx(1,15)=aeroV["dCyddr"];
    dCNadx.block(0,3,1,3)=-(aeroV["Clp"]*pd+aeroV["Clr"]*rd)*vi_VV;//dCNax/dv
    dCNadx.block(1,3,1,3)=-aeroV["Cmq"]*qd*vi_VV;//dCNay/dv
    dCNadx.block(2,3,1,3)=-(aeroV["Cnp"]*pd+aeroV["Cnr"]*rd)*vi_VV;//dCNaz/dv
    dCNadx.block(0,6,1,4)=b_2V*omegai.transpose()*(aeroV["Clp"]*dexdq+aeroV["Clr"]*dezdq);//dCNax/dq
    dCNadx.block(1,6,1,4)=c_2V*aeroV["Cmq"]*omegai.transpose()*deydq;//dCNay/dq
    dCNadx.block(2,6,1,4)=b_2V*omegai.transpose()*(aeroV["Cnp"]*dexdq+aeroV["Cnr"]*dezdq);//dCNaz/dq
    dCNadx.block(0,10,1,3)=b_2V*(aeroV["Clp"]*ex.transpose()+aeroV["Clr"]*ez.transpose());//dCNax/domega
    dCNadx.block(1,10,1,3)=c_2V*aeroV["Cmq"]*ey.transpose();//dCNax/domega
    dCNadx.block(2,10,1,3)=b_2V*(aeroV["Cnp"]*ex.transpose()+aeroV["Cnr"]*ez.transpose());//dCNaz/domega
    dCNadx(1,13)=aeroV["dCmdde"];
    dCNadx(0,14)=aeroV["Clda"];
    dCNadx(2,14)=aeroV["Cnda"];
    dCNadx(0,15)=aeroV["Cldr"];
    dCNadx(2,15)=aeroV["Cndr"];
    dCNadx.block(1,0,1,17)+=dCFadx.block(2,0,1,17)*(p->XcgR-p->Xcg);
    dCNadx.block(2,0,1,17)+=-dCFadx.block(1,0,1,17)*((p->XcgR-p->Xcg)*(p->mac/p->b));
    dCFadx+=dCFadA*dAdx+dCFadB*dBdx;
    dCNadx+=dCNadA*dAdx+dCNadB*dBdx;
    Eigen::Matrix<double,3,17> dFidx=Eigen::Matrix<double,3,17>::Zero();
    dFidx+=Rbi*(dCFadx*K+CFa*dKdx);
    dFidx.block(0,6,1,4)+=K*CFa.transpose()*dexddq;
    dFidx.block(1,6,1,4)+=K*CFa.transpose()*deyddq;
    dFidx.block(2,6,1,4)+=K*CFa.transpose()*dezddq;
    Eigen::Matrix<double,3,17> dNbdx=Eigen::Matrix<double,3,17>::Zero();
    dNbdx+=bcbMat*(dCNadx*K+CNa*dKdx);
    //造波抵抗
    dFidx+=-aeroV["Cdw"]*vni*dKdx-vni*K*aeroV["dCdwdM"]*dMdx;
    dFidx.block(0,3,3,3)+=-aeroV["Cdw"]*K*dvnidvi;
    //推力
    auto eng=getShared<SimpleFighterJetEngine>(p->engine);
    double thrust=eng->calcThrust(power,alt,Mach);
    double dTdP=eng->calcdTdP(power,alt,Mach);
    double dTdH=eng->calcdTdH(power,alt,Mach);
    double dTdM=eng->calcdTdM(power,alt,Mach);
    Eigen::Matrix<double,1,17> dTdx=dTdM*dMdx;
    dTdx(2)-=dTdH;
    dTdx(16)+=dTdP;
    dFidx+=ex*dTdx;
    dFidx.block(0,6,3,4)+=thrust*dexdq;
    Eigen::Matrix<double,3,17> dFabdx=Eigen::Matrix<double,3,17>::Zero();
    dFabdx+=dCFadx*K+CFa*dKdx;
    dFabdx+=-aeroV["Cdw"]*vb/V*dKdx-vb/V*K*aeroV["dCdwdM"]*dMdx;
    dFabdx.block(0,3,3,3)+=-aeroV["Cdw"]*K*Rbi.transpose()*dvnidvi;
    Eigen::Matrix<double,3,17> dFbdx=dFabdx;
    dFbdx.block(0,0,1,17)+=dTdx;
    dFbdx.block(0,6,1,4)+=p->m*gravity*dexdq.block(2,0,1,4);
    dFbdx.block(1,6,1,4)+=p->m*gravity*deydq.block(2,0,1,4);
    dFbdx.block(2,6,1,4)+=p->m*gravity*dezdq.block(2,0,1,4);

    //状態方程式
    Eigen::Vector3d Fab=(CFa-aeroV["Cdw"]*vb/V)*K;
    Eigen::Vector3d Fi=(Rbi*CFa-aeroV["Cdw"]*vni)*K+thrust*ex+Eigen::Vector3d(0,0,p->m*gravity);
    Eigen::Vector3d Nb=bcbMat*CNa*K;
    Eigen::Vector3d omegaDotB=p->Iinv*(Nb-omegaB.cross(p->I*omegaB+Eigen::Vector3d(p->Le,0.,0.)));
    Eigen::Vector3d omegaDotI=motion.relBtoP(omegaDotB);
    Eigen::Vector3d Lb=p->I*omegaB+Eigen::Vector3d(p->Le,0.,0.);
    Eigen::Matrix<double,3,17> domegaDotBdx=p->Iinv*(
        dNbdx+(
            skewMatrix(Lb)-skewMatrix(omegaB)*p->I
        )*domegabdx
    );
    Eigen::Matrix<double,3,17> domegaDotIdx=Rbi*domegaDotBdx;
    domegaDotIdx.block(0,6,1,4)+=omegaDotB.transpose()*dexddq;
    domegaDotIdx.block(1,6,1,4)+=omegaDotB.transpose()*deyddq;
    domegaDotIdx.block(2,6,1,4)+=omegaDotB.transpose()*dezddq;

    Eigen::VectorXd f=Eigen::VectorXd::Zero(17);
    f.block(0,0,3,1)=vi;
    f.block(3,0,3,1)=Fi/p->m;
    f.block(6,0,4,1)=Jq*omegai;
    f.block(10,0,3,1)=motion.relBtoP(omegaDotB);
    f(13,0)=-de/p->deTimeConstant;
    f(14,0)=-da/p->daTimeConstant;
    f(15,0)=-dr/p->drTimeConstant;
    Eigen::Matrix<double,17,4> g=Eigen::Matrix<double,17,4>::Zero();
    g(13,0)=p->deLimit/p->deTimeConstant;
    g(14,1)=p->daLimit/p->daTimeConstant;
    g(15,2)=p->drLimit/p->drTimeConstant;
    //推力コマンドのアフィンシステム化(線形化)
    double pMax,pMin,dPdtMax,dPdtMin;
    if(power<=17.0/72.0){
        //最大値が二次関数領域にある場合
        pMax=power+19.0/72.0;
        pMin=0;
    }else if(power<=4.0/15.0){
        //最大値がpCmd=0.5の場合
        pMax=0.5;
        pMin=0;
    }else if(power<0.5){
        //最大値がpCmd=0.6の場合
        pMax=0.6;
        pMin=0;
    }else{
        //AB領域の場合
        pMax=1.0;
        pMin=0.4;
    }
    dPdtMax=eng->pdot(power,pMax);
    dPdtMin=eng->pdot(power,pMin);
    f(16,0)=dPdtMin;
    g(16,3)=dPdtMax-dPdtMin;

    Eigen::Vector3d Fb=Fab+Eigen::Vector3d(thrust,0,0)+p->m*gravity*Eigen::Vector3d(ex(2),ey(2),ez(2));
    Eigen::Vector3d viDot=Fi/p->m;
    Eigen::Vector3d viDotDot_f=dFidx*f/p->m;
    Eigen::Matrix<double,3,4> viDotDot_g=dFidx*g/p->m;
    Eigen::Vector3d vbDot=Fb/p->m-omegaB.cross(vb);
    Eigen::Vector3d vbDotDot_f=-omegaDotB.cross(vb)-omegaB.cross(vbDot)+dFbdx*f/p->m;
    Eigen::Matrix<double,3,4> vbDotDot_g=dFbdx*g/p->m;
    double Vxz2=Vxz*Vxz;
    double Vxz2Dot_2=vb(0)*vbDot(0)+vb(2)*vbDot(2);
    double VDot=vni.dot(viDot);
    double VDotDot_f=(viDot.dot(viDot)-VDot*VDot)/V+vni.transpose()/p->m*dFidx*f;
    Eigen::Matrix<double,1,4> VDotDot_g=vni.transpose()/p->m*dFidx*g;
    double alphaDot=(vb(0)*vbDot(2)-vb(2)*vbDot(0))/(Vxz*Vxz);
    double betaDot=(V*vbDot(1)-vb(1)*VDot)/(V*Vxz);

    //DIとLQRの構築
    std::vector<Eigen::VectorXd> hs,zs;
    std::vector<Eigen::VectorXd> Lfs;
    std::vector<Eigen::MatrixXd> Lgs;
    std::vector<Eigen::MatrixXd> As,Bs;
    std::vector<Eigen::VectorXd> Qs,Rs;
    std::vector<std::string> cmdTypes;
    double hScale;
    int zdim=0;
    int vdim=0;
    for(int k=0;k<cmd.size();++k){
        int rk,hdim;
        Eigen::VectorXd zk,Lfk,Qk,Rk;
        Eigen::MatrixXd Ak,Bk,Lgk;
        nl::json elem=cmd[k];
        std::string cmdType=elem.at("type").get<std::string>();
        if(cmdType=="velocity(scalar)"){
            //速度(スカラー)(相対次数2)
            double targetVel=elem.at("target").get<double>();
            rk=2;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            zk(0)=V-targetVel;
            zk(1)=VDot;
            Lfk=Eigen::VectorXd(1);
            Lfk(0)=VDotDot_f;
            Lgk=VDotDot_g;
        }else if(cmdType=="acceleration(scalar)"){
            //加速度(スカラー)(相対次数1)
            double targetAccel=elem.at("target").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            zk(0)=VDot-targetAccel;
            Lfk=Eigen::VectorXd(1);
            Lfk(0)=VDotDot_f;
            Lgk=VDotDot_g;
        }else if(cmdType=="angular velocity(body)"){
            //機体座標系角速度ベクトル(相対次数2)
            Eigen::Vector3d targetOmega=elem.at("target").get<Eigen::Vector3d>();
            rk=2;hdim=3;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            zk.block(0*hdim,0,hdim,1)=omegaB-targetOmega;
            zk.block(1*hdim,0,hdim,1)=omegaDotB;
            Lfk=domegaDotBdx*f;
            Lgk=domegaDotBdx*g;
        }else if(cmdType=="thrust"){
            //推力(相対次数1)
            double targetThrust=elem.at("target").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            zk(0)=thrust-targetThrust;
            Lfk=dTdx*f;
            Lgk=dTdx*g;
        }else if(cmdType=="throttle(power)"){
            //スロットル(pCmd)(相対次数1)
            double targetPcmd=elem.at("target").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dPcmd=power-targetPcmd;
            zk(0)=dPcmd;
            Lfk=f.block(16,0,1,1);
            Lgk=g.block(16,0,1,4);
        }else if(cmdType=="Nz"){
            //Nz(相対次数1)…[5]で採用されているピッチ成分の制御量
            double targetNz=elem.at("target").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dNz=-(Fb(2)/p->m-ft2m(15.0)*omegaDotB(1))/gravity-targetNz;
            zk(0)=dNz;
            Eigen::Matrix<double,1,17> tmp=-1.0/(p->m*gravity)*dFbdx.block(2,0,1,17)+ft2m(15.0)/gravity*domegaDotBdx.block(1,0,1,17);
            Lfk=tmp*f;
            Lgk=tmp*g;
        }else if(cmdType=="Ps"){
            //ps(相対次数2)…[5]で採用されているロール成分の制御量
            double targetPs=elem.at("target").get<double>();
            rk=2;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dPs=vni.dot(omegai)-targetPs;
            zk(0)=dPs;
            zk.block(1,0,1,1)=omegai.transpose()*dvnidvi*Fi/p->m+vni.transpose()*omegaDotI;
            Eigen::Matrix<double,1,17> tmp=omegai.transpose()/(p->m)*dvnidvi*dFidx+vni.transpose()*domegaDotIdx;
            tmp.block(0,3,1,3)+=-omegai.transpose()/(p->m)*(
                vi.dot(Fi)*(3*dvnidvi-2*Eigen::Matrix3d::Identity()/V)
                +vni*Fi.transpose()+Fi*vni.transpose()
            )/(V*V)+omegaDotI.transpose()*dvnidvi;
            tmp.block(0,10,1,3)+=Fi.transpose()/(p->m)*dvnidvi;
            Lfk=tmp*f;
            Lgk=tmp*g;
        }else if(cmdType=="Ny+r"){
            //(18)Ny+r(相対次数1)…[5]で採用されているヨー成分の制御量
            double targetNyr=elem.at("target").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dNyr=((Fb(1)/p->m+ft2m(15.0)*omegaDotB(2))/gravity+omegaB(2))-targetNyr;
            zk(0)=dNyr;
            Eigen::Matrix<double,1,17> tmp=1.0/(p->m*gravity)*dFbdx.block(1,0,1,17)+ft2m(15.0)/gravity*domegaDotBdx.block(2,0,1,17);
            Lfk=tmp*f;Lfk(0)+=omegaDotB(2);
            Lgk=tmp*g;
        }else if(cmdType=="AlphaLimitter"){
            //αリミッタ(ソフト制約)
            double targetAlpha=elem.at("limit").get<double>();
            rk=2;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dAlpha=alpha-targetAlpha;
            zk(0)=dAlpha;
            zk(1)=alphaDot;
            Lfk=Eigen::VectorXd(1);
            Lfk(0)=(-alphaDot*2*Vxz2Dot_2+vb(0)*vbDotDot_f(2)-vb(2)*vbDotDot_f(0))/Vxz2;
            Lgk=(vb(0)*vbDotDot_g.block(2,0,1,4)-vb(2)*vbDotDot_g.block(0,0,1,4))/Vxz2;
        }else if(cmdType=="AlphaDotLimitter"){
            //dα/dtリミッタ(ソフト制約)
            //double targetAlphaDot=elem.at("limit").get<double>();
            rk=1;hdim=1;
            zk=Eigen::VectorXd::Zero(hdim*rk);
            double dAlphaDot=alphaDot;//-targetAlphaDot;
            zk(0)=dAlphaDot;
            Lfk=Eigen::VectorXd(1);
            Lfk(0)=(-alphaDot*2*Vxz2Dot_2+vb(0)*vbDotDot_f(2)-vb(2)*vbDotDot_f(0))/Vxz2;
            Lgk=(vb(0)*vbDotDot_g.block(2,0,1,4)-vb(2)*vbDotDot_g.block(0,0,1,4))/Vxz2;
        }else{
            throw std::runtime_error("Invalid control command. elem="+elem.dump());
            //continue;
        }
        if(hdim==3 && elem.contains("axis")){
            //特定の方向成分のみ指定する場合
            Eigen::Matrix<double,1,3> targetAxis=elem.at("axis").get<Eigen::Vector3d>().transpose();
            hdim=1;
            Eigen::VectorXd newZk=Eigen::VectorXd::Zero(hdim*rk);
            for(int r=0;r<rk;++r){
                newZk.block(r,0,1,1)=targetAxis*zk.block(r*3,0,3,1);
            }
            zk=newZk;
            Lfk=targetAxis*Lfk;
            Lgk=targetAxis*Lgk;
        }
        if(elem.contains("clamp")){
            double norm=zk.block(0,0,hdim,1).norm();
            double normLimit=elem.at("clamp").get<double>();
            if(norm>normLimit){
                zk.block(0,0,hdim,1)*=(normLimit/norm);
            }
        }
        if(elem.contains("useIntegral")){
            bool useIntegral=elem.at("useIntegral").get<bool>();
            if(useIntegral){
                std::string key=elem.at("integralKey").get<std::string>();
                if(integrals.count(key)==0){
                    integrals[key]=Eigen::VectorXd::Zero(hdim);
                }
                rk++;
                Eigen::VectorXd newZk=Eigen::VectorXd::Zero(hdim*rk);
                newZk.block(hdim,0,hdim*(rk-1),1)=zk.block(0,0,hdim*(rk-1),1);
                newZk.block(0,0,hdim,1)=integrals[key];
                zk=newZk;
                integrals[key]+=zk.block(hdim,0,hdim,1)*interval[SimPhase::CONTROL]*manager->getBaseTimeStep();
            }
        }
        if(elem.contains("scale")){
            hScale=elem.at("scale");
        }else{
            hScale=1.0;
        }
        zk*=hScale;
        Lfk*=hScale;
        Lgk*=hScale;
        if(elem.contains("oneSideOnly")){
            std::string oneSide=elem.at("oneSideOnly").get<std::string>();
            double sharpness;
            if(elem.contains("sharpness")){
                sharpness=elem.at("sharpness");
            }else{
                sharpness=1.0;
            }
            if(oneSide=="positive"){
                //nothing
            }else if(oneSide=="negative"){
                sharpness=-sharpness;
            }else{
                throw std::runtime_error("Only \"positive\" or \"negative\" can be designated for key \"oneSideOnly\".");
            }
            for(int i=0;i<hdim;++i){
                double x=sharpness*zk(i,0);
                double softplus=std::max(0.0,x)+log(1+exp(-abs(x)));//softplus
                double d1=exp(std::min(0.0,x))/(1+exp(-abs(x)));//sigmoid
                zk(i,0)=softplus/sharpness;
                assert(rk<=4);
                if(rk==1){
                    Lfk(i)=d1*Lfk(i);
                }else{
                    double z1=zk(i+1*hdim,0);
                    double d2=sharpness*d1*(1-d1);
                    zk(i+1*hdim,0)=d1*z1;
                    if(rk==2){
                        Lfk(i)=d1*Lfk(i)+d2*pow(z1,2);
                    }else{
                        double z2=zk(i+2*hdim,0);
                        double d3=sharpness*sharpness*d1*(1-d1)*(1-2*d1);
                        zk(i+2*hdim,0)=d1*z2+d2*pow(z1,2);
                        if(rk==3){
                            Lfk(i)=d1*Lfk(i)+d2*3*z1*z2+d3*pow(z1,3);
                        }else{
                            double z3=zk(i+3*hdim,0);
                            double d4=sharpness*sharpness*sharpness*d1*(1-d1)*(1-6*d1*(1-d1));
                            zk(i+3*hdim,0)=d1*z3+d2*3*z1*z2+d3*pow(z1,3);
                            Lfk(i)=d1*Lfk(i)+d2*(4*z1*z3+3*z2*z2)+d3*6*z1*z1*z2+d4*pow(z1,4);
                        }
                    }
                }
                Lgk.block(i,0,1,4)=d1*Lgk.block(i,0,1,4);
            }
        }
        if(elem.contains("Q")){
            Qk=elem.at("Q").get<Eigen::VectorXd>();
        }else{
            Qk=Eigen::VectorXd::Zero(rk*hdim);
            for(int i=0;i<hdim;++i){
                Qk(i)=1;
            }
        }
        if(elem.contains("R")){
            Rk=elem.at("R").get<Eigen::VectorXd>();
        }else{
            Rk=Eigen::VectorXd::Constant(hdim,1);
        }
        zdim+=rk*hdim;
        vdim+=hdim;
        Ak=Eigen::MatrixXd::Zero(rk*hdim,rk*hdim);
        Bk=Eigen::MatrixXd::Zero(rk*hdim,hdim);
        for(int i=0;i<hdim*(rk-1);++i){
            Ak(i,i+hdim)=1;
        }
        for(int i=0;i<hdim;++i){
            Bk(hdim*(rk-1)+i,i)=1;
        }
        zs.push_back(zk);
        As.push_back(Ak);
        Bs.push_back(Bk);
        Lfs.push_back(Lfk);
        Lgs.push_back(Lgk);
        Qs.push_back(Qk);
        Rs.push_back(Rk);
        cmdTypes.push_back(cmdType);
    }
    Eigen::VectorXd z=Eigen::VectorXd::Zero(zdim);
    Eigen::VectorXd Lf=Eigen::VectorXd::Zero(vdim);
    Eigen::MatrixXd Lg=Eigen::MatrixXd::Zero(vdim,4);
    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(zdim,zdim);
    Eigen::MatrixXd B=Eigen::MatrixXd::Zero(zdim,vdim);
    Eigen::VectorXd Q=Eigen::VectorXd::Zero(zdim);
    Eigen::VectorXd R=Eigen::VectorXd::Zero(vdim);
    int zIdx=0;
    int vIdx=0;
    for(int k=0;k<zs.size();++k){
        int zk=zs[k].rows();
        int vk=Bs[k].cols();
        z.block(zIdx,0,zk,1)=zs[k];
        Lf.block(vIdx,0,vk,1)=Lfs[k];
        Lg.block(vIdx,0,vk,4)=Lgs[k];
        A.block(zIdx,zIdx,zk,zk)=As[k];
        B.block(zIdx,vIdx,zk,vk)=Bs[k];
        Q.block(zIdx,0,zk,1)=Qs[k];
        R.block(vIdx,0,vk,1)=Rs[k];
        zIdx+=zk;
        vIdx+=vk;
    }
    P=ArimotoPotter(A,B,Q.asDiagonal().toDenseMatrix(),R.asDiagonal().toDenseMatrix());
    Eigen::VectorXd Rinv=R.cwiseInverse();
    Eigen::VectorXd v=-Rinv.asDiagonal().toDenseMatrix()*B.transpose()*P*z;
    Eigen::MatrixXd Ginv=Lg.transpose()*(Lg*Lg.transpose()).inverse();
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> LgDecomp(Lg);
    Eigen::VectorXd u=LgDecomp.solve(v-Lf);
    //アフィンシステム化した推力コマンドの復元
    u(3)=std::clamp<double>(u(3),0,1);
    if(power<0.25){
        //二次関数領域を含む場合
        double dPdt=dPdtMin+u(3)*(dPdtMax-dPdtMin);
        if(dPdt>0.25){
            u(3)=power+(19.0-sqrt(1.0-1440.0*(dPdt-0.25)))/72.0;
        }else{
            u(3)=power+dPdt;
        }
    }else if(power<=4.0/15.0){
        //線形領域の場合(pCmd=0.5が最大となる領域)
        u(3)=u(3)*0.5;
    }else if(power<0.5){
        //線形領域の場合(pCmd=0.6が最大となる領域)
        if(u(3)*0.6>=0.55){
            u(3)=0.6;
        }else if(u(3)*0.6>0.5){
            u(3)=0.5;
        }else{
            u(3)=u(3)*0.6;
        }
    }else{
        //AB領域の場合
        if(u(3)*0.6<=0.05){
            u(3)=0.4;
        }else if(u(3)*0.6<0.1){
            u(3)=0.5;
        }else{
            u(3)=0.4+u(3)*0.6;
        }
    }
    double total=u.sum();
    if(std::isnan(total)){
        DEBUG_PRINT_EXPRESSION(u);
        DEBUG_PRINT_EXPRESSION(P);
        DEBUG_PRINT_EXPRESSION(A);
        DEBUG_PRINT_EXPRESSION(B);
        DEBUG_PRINT_EXPRESSION(Q);
        DEBUG_PRINT_EXPRESSION(R);
        DEBUG_PRINT_EXPRESSION(Lf);
        DEBUG_PRINT_EXPRESSION(Lg);
        DEBUG_PRINT_EXPRESSION(v);
        DEBUG_PRINT_EXPRESSION(VDotDot_f);
        DEBUG_PRINT_EXPRESSION(VDotDot_g);
        Eigen::MatrixXd GGT=Lg*Lg.transpose();
        DEBUG_PRINT_EXPRESSION(GGT);
        DEBUG_PRINT_EXPRESSION(Ginv);
        throw std::runtime_error("NaN appeared.");
    }
    return u;
}

StevensFighter::StevensFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:SixDoFFighter(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    dampTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"damp",randomGen,Eigen::Tensor<double,2>(9,12).setValues({
        {-0.267,-0.110,+0.308,+1.340,+2.080,+2.910,+2.760,+2.050,+1.500,+1.490,+1.830,+1.210},
		{+0.882,+0.852,+0.876,+0.958,+0.962,+0.974,+0.819,+0.483,+0.590,+1.210,-0.493,-1.040},
		{-0.108,-0.108,-0.188,+0.110,+0.258,+0.226,+0.344,+0.362,+0.611,+0.529,+0.298,-2.270},
		{-8.800,-25.80,-28.90,-31.40,-31.20,-30.70,-27.70,-28.20,-29.00,-29.80,-38.30,-35.30},
		{-0.126,-0.026,+0.063,+0.113,+0.208,+0.230,+0.319,+0.437,+0.680,+0.100,+0.447,-0.330},
		{-0.360,-0.359,-0.443,-0.420,-0.383,-0.375,-0.329,-0.294,-0.230,-0.210,-0.120,-0.100},
		{-7.210,-5.400,-5.230,-5.260,-6.110,-6.640,-5.690,-6.000,-6.200,-6.400,-6.660,-6.000},//2列目は出典元では-0.540だが-5.400では？
		{-0.380,-0.363,-0.378,-0.386,-0.370,-0.453,-0.550,-0.582,-0.595,-0.637,-1.020,-0.840},
		{+0.061,+0.052,+0.052,-0.012,-0.013,-0.024,+0.050,+0.150,+0.130,+0.158,+0.240,+0.150}
    }));
    cxTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cx",randomGen,Eigen::Tensor<double,2>(5,12).setValues({
        {-0.099,-0.081,-0.081,-0.063,-0.025,+0.044,+0.097,+0.113,+0.145,+0.167,+0.174,+0.166},
		{-0.048,-0.038,-0.040,-0.021,+0.016,+0.083,+0.127,+0.137,+0.162,+0.177,+0.179,+0.167},
		{-0.022,-0.020,-0.021,-0.004,+0.032,+0.094,+0.128,+0.130,+0.154,+0.161,+0.155,+0.138},
		{-0.040,-0.038,-0.039,-0.025,+0.006,+0.062,+0.087,+0.085,+0.100,+0.110,+0.104,+0.091},
		{-0.083,-0.073,-0.076,-0.072,-0.046,+0.012,+0.024,+0.025,+0.043,+0.053,+0.047,+0.040}
    }));
    czTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cz",randomGen,Eigen::Tensor<double,1>(12).setValues({
		+0.770,+0.241,-0.100,-0.416,-0.731,-1.053,-1.366,-1.646,-1.917,-2.120,-2.248,-2.229
    }));
    clTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cl",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000},
        {-0.001,-0.004,-0.008,-0.012,-0.016,-0.019,-0.020,-0.020,-0.015,-0.008,-0.013,-0.015},
		{-0.003,-0.009,-0.017,-0.024,-0.030,-0.034,-0.040,-0.037,-0.016,-0.002,-0.010,-0.019},
		{-0.001,-0.010,-0.020,-0.030,-0.039,-0.044,-0.050,-0.049,-0.023,-0.006,-0.014,-0.027},
        {+0.000,-0.010,-0.022,-0.034,-0.047,-0.046,-0.059,-0.061,-0.033,-0.036,-0.035,-0.035},
		{+0.007,-0.010,-0.023,-0.034,-0.049,-0.046,-0.068,-0.071,-0.060,-0.058,-0.062,-0.059},
		{+0.009,-0.011,-0.023,-0.037,-0.050,-0.047,-0.074,-0.079,-0.091,-0.076,-0.077,-0.076}
    }));
    cmTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cm",randomGen,Eigen::Tensor<double,2>(5,12).setValues({
        {+0.205,+0.168,+0.186,+0.196,+0.213,+0.251,+0.245,+0.238,+0.252,+0.231,+0.198,+0.192},
		{+0.081,+0.077,+0.107,+0.110,+0.110,+0.141,+0.127,+0.119,+0.133,+0.108,+0.081,+0.093},
		{-0.046,-0.020,-0.009,-0.005,-0.006,+0.010,+0.006,-0.001,+0.014,+0.000,-0.013,+0.032},
        {-0.174,-0.145,-0.121,-0.127,-0.129,-0.102,-0.097,-0.113,-0.087,-0.084,-0.069,-0.006},
		{-0.259,-0.202,-0.184,-0.193,-0.199,-0.150,-0.160,-0.167,-0.104,-0.076,-0.041,-0.005}
    }));
    cnTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cn",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000,+0.000},
        {+0.018,+0.019,+0.018,+0.019,+0.019,+0.018,+0.013,+0.007,+0.004,-0.014,-0.017,-0.033},
        {+0.038,+0.042,+0.042,+0.042,+0.043,+0.039,+0.030,+0.017,+0.004,-0.035,-0.047,-0.057},
        {+0.056,+0.057,+0.059,+0.058,+0.058,+0.053,+0.032,+0.012,+0.002,-0.046,-0.071,-0.073},
        {+0.064,+0.077,+0.076,+0.074,+0.073,+0.057,+0.029,+0.007,+0.012,-0.034,-0.065,-0.041},
        {+0.074,+0.086,+0.093,+0.089,+0.080,+0.062,+0.049,+0.022,+0.028,-0.012,-0.002,-0.013},
        {+0.079,+0.090,+0.106,+0.106,+0.096,+0.080,+0.068,+0.030,+0.064,+0.015,+0.011,-0.001}
    }));
    dldaTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"dlda",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {-0.041,-0.052,-0.053,-0.056,-0.050,-0.056,-0.082,-0.059,-0.042,-0.038,-0.027,-0.017},
        {-0.041,-0.053,-0.053,-0.053,-0.050,-0.051,-0.066,-0.043,-0.038,-0.027,-0.023,-0.016},
        {-0.042,-0.053,-0.052,-0.051,-0.049,-0.049,-0.043,-0.035,-0.026,-0.016,-0.018,-0.014},
        {-0.040,-0.052,-0.051,-0.052,-0.048,-0.048,-0.042,-0.037,-0.031,-0.026,-0.017,-0.012},
        {-0.043,-0.049,-0.048,-0.049,-0.043,-0.042,-0.042,-0.036,-0.025,-0.021,-0.016,-0.011},
        {-0.044,-0.048,-0.048,-0.047,-0.042,-0.041,-0.020,-0.028,-0.013,-0.014,-0.011,-0.010},
        {-0.043,-0.049,-0.047,-0.045,-0.042,-0.037,-0.003,-0.013,-0.010,-0.003,-0.007,-0.008}
    }));
    dldrTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"dldr",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {+0.005,+0.017,+0.014,+0.010,-0.005,+0.009,+0.019,+0.005,-0.000,-0.005,-0.011,+0.008},
        {+0.007,+0.016,+0.014,+0.014,+0.013,+0.009,+0.012,+0.005,+0.000,+0.004,+0.009,+0.007},
        {+0.013,+0.013,+0.011,+0.012,+0.011,+0.009,+0.008,+0.005,-0.002,+0.005,+0.003,+0.005},
        {+0.018,+0.015,+0.015,+0.014,+0.014,+0.014,+0.014,+0.015,+0.013,+0.011,+0.006,+0.001},
        {+0.015,+0.014,+0.013,+0.013,+0.012,+0.011,+0.011,+0.010,+0.008,+0.008,+0.007,+0.003},
        {+0.021,+0.011,+0.010,+0.011,+0.010,+0.009,+0.008,+0.010,+0.006,+0.005,+0.000,+0.001},
        {+0.023,+0.010,+0.011,+0.011,+0.011,+0.010,+0.008,+0.010,+0.006,+0.014,+0.020,+0.000}
    }));
    dndaTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"dnda",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {+0.001,-0.027,-0.017,-0.013,-0.012,-0.016,+0.001,+0.017,+0.011,+0.017,+0.008,+0.016},
        {+0.002,-0.014,-0.016,-0.016,-0.014,-0.019,-0.021,+0.002,+0.012,+0.015,+0.015,+0.011},
        {-0.006,-0.008,-0.006,-0.006,-0.005,-0.008,-0.005,+0.007,+0.004,+0.007,+0.006,+0.006},
        {-0.011,-0.011,-0.010,-0.009,-0.008,-0.006,+0.000,+0.004,+0.007,+0.010,+0.004,+0.010},
        {-0.015,-0.015,-0.014,-0.012,-0.011,-0.008,-0.002,+0.002,+0.006,+0.012,+0.011,+0.011},
        {-0.024,-0.010,-0.004,-0.002,-0.001,+0.003,+0.014,+0.006,-0.001,+0.004,+0.004,+0.006},
        {-0.022,+0.002,-0.003,-0.005,-0.003,-0.001,-0.009,-0.009,-0.001,+0.003,-0.002,+0.001}
    }));
    dndrTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"dndr",randomGen,Eigen::Tensor<double,2>(7,12).setValues({
        {-0.018,-0.052,-0.052,-0.052,-0.054,-0.049,-0.059,-0.051,-0.030,-0.037,-0.026,-0.013},
        {-0.028,-0.051,-0.043,-0.046,-0.045,-0.049,-0.057,-0.052,-0.030,-0.033,-0.030,-0.008},
        {-0.037,-0.041,-0.038,-0.040,-0.040,-0.038,-0.037,-0.030,-0.027,-0.024,-0.019,-0.013},
        {-0.048,-0.045,-0.045,-0.045,-0.044,-0.045,-0.047,-0.048,-0.049,-0.045,-0.033,-0.016},
        {-0.043,-0.044,-0.041,-0.041,-0.040,-0.038,-0.034,-0.035,-0.035,-0.029,-0.022,-0.009},
        {-0.052,-0.034,-0.036,-0.036,-0.035,-0.028,-0.024,-0.023,-0.020,-0.016,-0.010,-0.014},
        {-0.062,-0.034,-0.027,-0.028,-0.027,-0.027,-0.023,-0.023,-0.019,-0.009,-0.025,-0.010}
    }));
    alphaTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"alpha",randomGen,
        (Eigen::Matrix<double,12,1>()<<-10.0,-5.0,0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0).finished()
    )*deg2rad(1);
    betaSymTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"betaSym",randomGen,
        (Eigen::Matrix<double,7,1>()<<0.0,5.0,10.0,15.0,20.0,25.0,30.0).finished()
    )*deg2rad(1);
    betaAsymTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"betaAsym",randomGen,
        (Eigen::Matrix<double,7,1>()<<-30.0,-20.0,-10.0,0.0,10.0,20.0,30.0).finished()
    )*deg2rad(1);
    deTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"de",randomGen,
        (Eigen::Matrix<double,5,1>()<<-24.0,-12.0,0.0,12.0,24.0).finished()
    )*deg2rad(1);
    dCydB=getValueFromJsonKRD(modelConfig.at("dynamics"),"dCydB",randomGen,rad2deg(-0.02));
    dCydda=getValueFromJsonKRD(modelConfig.at("dynamics"),"dCydda",randomGen,0.021/deg2rad(20.0));
    dCyddr=getValueFromJsonKRD(modelConfig.at("dynamics"),"dCyddr",randomGen,0.086/deg2rad(30.0));
    dCzdde=getValueFromJsonKRD(modelConfig.at("dynamics"),"dCzdde",randomGen,-0.019/deg2rad(25.0));
}
StevensFighter::~StevensFighter(){
}
std::map<std::string,double> StevensFighter::calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives){
    std::map<std::string,double> ret;
    std::vector<Eigen::VectorXd> points;
    Eigen::MatrixXd x(1,1);
    Eigen::MatrixXd grad;
    points.push_back(alphaTable);
    x<<alpha_;
    double cz_tmp=interpn(points,czTable,x)(0);
    ret["Cz"]=cz_tmp*(1-beta_*beta_)+de_*dCzdde;
    if(withDerivatives){
        ret["dCzdA"]=interpgradn(points,czTable,x)(0,0)*(1-beta_*beta_);
        ret["dCzdB"]=-2*beta_*cz_tmp;
        ret["dCzdde"]=dCzdde;
    }
    //αのみに依存
    Eigen::Tensor<double,1> table(12);
    Eigen::array<int,2> extents={1,12};
    Eigen::VectorXd damps(9),dampgrads(9);
    for(int i=0;i<9;++i){
        Eigen::array<int,2> offsets={i,0};
        table=dampTable.slice(offsets,extents).reshape(Eigen::array<std::int64_t,1>{{12}});
        damps(i)=interpn(points,table,x)(0);
        if(withDerivatives){
            dampgrads(i)=interpgradn(points,table,x)(0,0);
        }
    }
    ret["Cxq"]=damps(0);
    ret["Cyr"]=damps(1);
    ret["Cyp"]=damps(2);
    ret["Czq"]=damps(3);
    ret["Clr"]=damps(4);
    ret["Clp"]=damps(5);
    ret["Cmq"]=damps(6);
    ret["Cnr"]=damps(7);
    ret["Cnp"]=damps(8);
    if(withDerivatives){
	    ret["dCxqdA"]=dampgrads(0);
	    ret["dCyrdA"]=dampgrads(1);
	    ret["dCypdA"]=dampgrads(2);
	    ret["dCzqdA"]=dampgrads(3);
	    ret["dClrdA"]=dampgrads(4);
	    ret["dClpdA"]=dampgrads(5);
	    ret["dCmqdA"]=dampgrads(6);
	    ret["dCnrdA"]=dampgrads(7);
	    ret["dCnpdA"]=dampgrads(8);
    }
    //α,deに依存
    points[0]=deTable;
    points.push_back(alphaTable);
    x=Eigen::MatrixXd::Zero(1,2);
    x<<de_,alpha_;
    ret["Cx"]=interpn(points,cxTable,x)(0);
    ret["Cm"]=interpn(points,cmTable,x)(0);
    if(withDerivatives){
        grad=interpgradn(points,cxTable,x);
        ret["dCxdA"]=grad(1,0);
        ret["dCxdde"]=grad(0,0);
        grad=interpgradn(points,cmTable,x);
        ret["dCmdA"]=grad(1,0);
        ret["dCmdde"]=grad(0,0);
    }
    //α,βに依存(βについて偶関数)
    points[0]=betaSymTable;
    x<<abs(beta_),alpha_;
    int sgnBeta = (beta_>=0) ? 1 : -1;
    ret["Cl"]=sgnBeta*interpn(points,clTable,x)(0);
    ret["Cn"]=sgnBeta*interpn(points,cnTable,x)(0);
    if(withDerivatives){
        grad=interpgradn(points,clTable,x);
        ret["dCldA"]=sgnBeta*grad(1,0);
        ret["dCldB"]=sgnBeta*grad(0,0);
        grad=interpgradn(points,cnTable,x);
        ret["dCndA"]=sgnBeta*grad(1,0);
        ret["dCndB"]=sgnBeta*grad(0,0);
    }
    //α,βに依存(βについて奇関数)
    points[0]=betaAsymTable;
    x<<beta_,alpha_;
    ret["Clda"]=interpn(points,dldaTable,x)(0);
    ret["Cldr"]=interpn(points,dldrTable,x)(0);
    ret["Cnda"]=interpn(points,dndaTable,x)(0);
    ret["Cndr"]=interpn(points,dndrTable,x)(0);
    if(withDerivatives){
        grad=interpgradn(points,dldaTable,x);
        ret["dCldadA"]=grad(1,0);
        ret["dCldadB"]=grad(0,0);
        grad=interpgradn(points,dldrTable,x);
        ret["dCldrdA"]=grad(1,0);
        ret["dCldrdB"]=grad(0,0);
        grad=interpgradn(points,dndaTable,x);
        ret["dCndadA"]=grad(1,0);
        ret["dCndadB"]=grad(0,0);
        grad=interpgradn(points,dndrTable,x);
        ret["dCndrdA"]=grad(1,0);
        ret["dCndrdB"]=grad(0,0);
    }
    ret["Cy"]=dCydB*beta_+dCydda*da_+dCyddr*dr_;
    if(withDerivatives){
	    ret["dCydB"]=dCydB;
	    ret["dCydda"]=dCydda;
	    ret["dCyddr"]=dCyddr;
    }
	//造波抵抗
    double cdw0=cdwTable(0);
    double kdwm=cdwTable(1);
    double kdw=cdwTable(2);
    double dm=cdwTable(3);
    double d1=1.0+exp(-8*(mach_-(1.0-dm/2.0))/dm);
    double d2=pow(pow(mach_-kdwm,2)-1.0,2)+pow(kdw,4);
    ret["Cdw"]=cdw0*kdw/(d1*pow(d2,0.25));
    if(withDerivatives){
        //造波抵抗
        ret["dCdwdM"]=ret["Cdw"]*(8.0*(dm-1.0)/(dm*d1)-(mach_-kdwm)*(pow(mach_-kdwm,2)-1.0)/d2);
        //0のもの
        ret["dCydA"]=0;
	    ret["dCxdB"]=0;
        ret["dCxqdB"]=0;
        ret["dCypdB"]=0;
        ret["dCyrdB"]=0;
	    ret["dCzqdB"]=0;
	    ret["dClpdB"]=0;
	    ret["dClrdB"]=0;
	    ret["dCmdB"]=0;
	    ret["dCmqdB"]=0;
	    ret["dCnpdB"]=0;
	    ret["dCnrdB"]=0;
        //2階微分(αで微分した後α,β,deで微分) ※テーブルを参照する場合の2階微分は0とみなす
        ret["dCxadA"]=0;
        ret["dCxadB"]=0;
        ret["dCxqadA"]=0;
        ret["dCxqadB"]=0;
        ret["dCzadA"]=0;
        ret["dCzadB"]=-2*beta_*cz_tmp;
	    ret["dCzqadA"]=0;
	    ret["dCzqadB"]=0;
        ret["dCxadde"]=0;
        ret["dCzadde"]=0;
    }
    return std::move(ret);
}

MorelliFighter::MorelliFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:SixDoFFighter(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    aeroC["a"]=(Eigen::Matrix<double,7,1>()<<-1.943367e-2,2.136104e-1,-2.903457e-1,-3.348641e-3,-2.060504e-1,6.988016e-1,-9.035381e-1).finished();
    aeroC["b"]=(Eigen::Matrix<double,5,1>()<<4.833383e-1,8.644627e0,1.131098e1,-7.422961e1,6.075776e1).finished();
    aeroC["c"]=(Eigen::Matrix<double,3,1>()<<-1.145916e0,6.016057e-2,1.642479e-1).finished();
    aeroC["d"]=(Eigen::Matrix<double,4,1>()<<-1.006733e-1,8.679799e-1,4.260586e0,-6.923267e0).finished();
    aeroC["e"]=(Eigen::Matrix<double,4,1>()<<8.071648e-1,1.189633e-1,4.177702e0,-9.162236e0).finished();
    aeroC["f"]=(Eigen::Matrix<double,6,1>()<<-1.378278e-1,-4.211369e0,4.775187e0,-1.026225e1,8.399763e0,-4.354000e-1).finished();
    aeroC["g"]=(Eigen::Matrix<double,5,1>()<<-3.054956e1,-4.132305e1,3.292788e2,-6.848038e2,4.08024e2).finished();
    aeroC["h"]=(Eigen::Matrix<double,8,1>()<<-1.058583e-1,-5.776677e-1,-1.672435e-2,1.357256e-1,2.172952e-1,3.464156e0,-2.835451e0,-1.098104e0).finished();
    aeroC["i"]=(Eigen::Matrix<double,4,1>()<<-4.126806e-1,-1.189974e-1,1.247721e0,-7.391132e-1).finished();
    aeroC["j"]=(Eigen::Matrix<double,5,1>()<<6.250437e-2,6.067723e-1,-1.101964e0,9.100087e0,-1.192672e1).finished();
    aeroC["k"]=(Eigen::Matrix<double,7,1>()<<-1.463144e-1,-4.073901e-2,3.253159e-2,4.851209e-1,2.978850e-1,-3.746393e-1,-3.213068e-1).finished();
    aeroC["l"]=(Eigen::Matrix<double,7,1>()<<2.635729e-2,-2.192910e-2,-3.152901e-3,-5.817803e-2,4.516159e-1,-4.928702e-1,-1.579864e-2).finished();
    aeroC["m"]=(Eigen::Matrix<double,8,1>()<<-2.029370e-2,4.660702e-2,-6.012308e-1,-8.062977e-2,8.320429e-2,5.018538e-1,6.378864e-1,4.226356e-1).finished();
    aeroC["n"]=(Eigen::Matrix<double,6,1>()<<-5.159153e0,-3.554716e0,-3.598636e1,2.247355e2,-4.120991e2,2.411750e2).finished();
    aeroC["o"]=(Eigen::Matrix<double,7,1>()<<2.993363e-1,6.594004e-2,-2.003125e-1,-6.233977e-2,-2.107885e0,2.141420e0,8.476901e-1).finished();
    aeroC["p"]=(Eigen::Matrix<double,5,1>()<<2.677652e-2,-3.298246e-1,1.926178e-1,4.013325e0,-4.404302e0).finished();
    aeroC["q"]=(Eigen::Matrix<double,3,1>()<<-3.698756e-1,-1.167551e-1,-7.641297e-1).finished();
    aeroC["r"]=(Eigen::Matrix<double,10,1>()<<-3.348717e-2,4.276655e-2,6.573646e-3,3.535831e-1,-1.373308e0,1.237582e0,2.302543e-1,-2.512876e-1,1.588105e-1,-5.199526e-1).finished();
    aeroC["s"]=(Eigen::Matrix<double,6,1>()<<-8.115894e-2,-1.156580e-2,2.514167e-2,2.038748e-1,-3.337476e-1,1.004297e-1).finished();
    if(modelConfig.at("dynamics").contains("aero")){
        nl::json aeroConf=modelConfig.at("dynamics").at("aero");
        for(auto&& item:aeroC){
            aeroC[item.first]=getValueFromJsonKRD(aeroConf,item.first,randomGen,aeroC[item.first]);
        }
    }
}
MorelliFighter::~MorelliFighter(){
}
std::map<std::string,double> MorelliFighter::calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives){
    std::map<std::string,double> ret;
    ret["Cx"]=aeroC["a"][0]+aeroC["a"][1]*alpha_+aeroC["a"][2]*pow(de_,2)+aeroC["a"][3]*de_+aeroC["a"][4]*alpha_*de_+aeroC["a"][5]*pow(alpha_,2)+aeroC["a"][6]*pow(alpha_,3);
	ret["Cxq"]=aeroC["b"][0]+aeroC["b"][1]*alpha_+aeroC["b"][2]*pow(alpha_,2)+aeroC["b"][3]*pow(alpha_,3)+aeroC["b"][4]*pow(alpha_,4);
	ret["Cy"]=aeroC["c"][0]*beta_+aeroC["c"][1]*da_+aeroC["c"][2]*dr_;
	ret["Cyp"]=aeroC["d"][0]+aeroC["d"][1]*alpha_+aeroC["d"][2]*pow(alpha_,2)+aeroC["d"][3]*pow(alpha_,3);
	ret["Cyr"]=aeroC["e"][0]+aeroC["e"][1]*alpha_+aeroC["e"][2]*pow(alpha_,2)+aeroC["e"][3]*pow(alpha_,3);
	ret["Cz"]=(aeroC["f"][0]+aeroC["f"][1]*alpha_+aeroC["f"][2]*pow(alpha_,2)+aeroC["f"][3]*pow(alpha_,3)+aeroC["f"][4]*pow(alpha_,4))*(1.0-pow(beta_,2))+aeroC["f"][5]*de_;
	ret["Czq"]=aeroC["g"][0]+aeroC["g"][1]*alpha_+aeroC["g"][2]*pow(alpha_,2)+aeroC["g"][3]*pow(alpha_,3)+aeroC["g"][4]*pow(alpha_,4);
	ret["Cl"]=aeroC["h"][0]*beta_+aeroC["h"][1]*alpha_*beta_+aeroC["h"][2]*pow(alpha_,2)*beta_+aeroC["h"][3]*pow(beta_,2)+aeroC["h"][4]*alpha_*pow(beta_,2)+aeroC["h"][5]*pow(alpha_,3)*beta_+aeroC["h"][6]*pow(alpha_,4)*beta_+aeroC["h"][7]*pow(alpha_,2)*pow(beta_,2);
	ret["Clp"]=aeroC["i"][0]+aeroC["i"][1]*alpha_+aeroC["i"][2]*pow(alpha_,2)+aeroC["i"][3]*pow(alpha_,3);
	ret["Clr"]=aeroC["j"][0]+aeroC["j"][1]*alpha_+aeroC["j"][2]*pow(alpha_,2)+aeroC["j"][3]*pow(alpha_,3)+aeroC["j"][4]*pow(alpha_,4);
	ret["Clda"]=aeroC["k"][0]+aeroC["k"][1]*alpha_+aeroC["k"][2]*beta_+aeroC["k"][3]*pow(alpha_,2)+aeroC["k"][4]*alpha_*beta_+aeroC["k"][5]*pow(alpha_,2)*beta_+aeroC["k"][6]*pow(alpha_,3);
	ret["Cldr"]=aeroC["l"][0]+aeroC["l"][1]*alpha_+aeroC["l"][2]*beta_+aeroC["l"][3]*alpha_*beta_+aeroC["l"][4]*pow(alpha_,2)*beta_+aeroC["l"][5]*pow(alpha_,3)*beta_+aeroC["l"][6]*pow(beta_,2);
	ret["Cm"]=aeroC["m"][0]+aeroC["m"][1]*alpha_+aeroC["m"][2]*de_+aeroC["m"][3]*alpha_*de_+aeroC["m"][4]*pow(de_,2)+aeroC["m"][5]*pow(alpha_,2)*de_+aeroC["m"][6]*pow(de_,3)+aeroC["m"][7]*alpha_*pow(de_,2);
	ret["Cmq"]=aeroC["n"][0]+aeroC["n"][1]*alpha_+aeroC["n"][2]*pow(alpha_,2)+aeroC["n"][3]*pow(alpha_,3)+aeroC["n"][4]*pow(alpha_,4)+aeroC["n"][5]*pow(alpha_,5);
	ret["Cn"]=aeroC["o"][0]*beta_+aeroC["o"][1]*alpha_*beta_+aeroC["o"][2]*pow(beta_,2)+aeroC["o"][3]*alpha_*pow(beta_,2)+aeroC["o"][4]*pow(alpha_,2)*beta_+aeroC["o"][5]*pow(alpha_,2)*pow(beta_,2)+aeroC["o"][6]*pow(alpha_,3)*beta_;
	ret["Cnp"]=aeroC["p"][0]+aeroC["p"][1]*alpha_+aeroC["p"][2]*pow(alpha_,2)+aeroC["p"][3]*pow(alpha_,3)+aeroC["p"][4]*pow(alpha_,4);
	ret["Cnr"]=aeroC["q"][0]+aeroC["q"][1]*alpha_+aeroC["q"][2]*pow(alpha_,2);
	ret["Cnda"]=aeroC["r"][0]+aeroC["r"][1]*alpha_+aeroC["r"][2]*beta_+aeroC["r"][3]*alpha_*beta_+aeroC["r"][4]*pow(alpha_,2)*beta_+aeroC["r"][5]*pow(alpha_,3)*beta_+aeroC["r"][6]*pow(alpha_,2)+aeroC["r"][7]*pow(alpha_,3)+aeroC["r"][8]*pow(beta_,3)+aeroC["r"][9]*alpha_*pow(beta_,3);
	ret["Cndr"]=aeroC["s"][0]+aeroC["s"][1]*alpha_+aeroC["s"][2]*beta_+aeroC["s"][3]*alpha_*beta_+aeroC["s"][4]*pow(alpha_,2)*beta_+aeroC["s"][5]*pow(alpha_,2);
    //造波抵抗
    double cdw0=cdwTable(0);
    double kdwm=cdwTable(1);
    double kdw=cdwTable(2);
    double dm=cdwTable(3);
    double d1=1.0+exp(-8*(mach_-(1.0-dm/2.0))/dm);
    double d2=pow(pow(mach_-kdwm,2)-1.0,2)+pow(kdw,4);
    ret["Cdw"]=cdw0*kdw/(d1*pow(d2,0.25));
    if(withDerivatives){
        //α微分
	    ret["dCxdA"]=aeroC["a"][1]+aeroC["a"][4]*de_+2*aeroC["a"][5]*alpha_+3*aeroC["a"][6]*pow(alpha_,2);
	    ret["dCxqdA"]=aeroC["b"][1]+2*aeroC["b"][2]*alpha_+3*aeroC["b"][3]*pow(alpha_,2)+4*aeroC["b"][4]*pow(alpha_,3);
        ret["dCydA"]=0;
	    ret["dCypdA"]=aeroC["d"][1]+2*aeroC["d"][2]*alpha_+3*aeroC["d"][3]*pow(alpha_,2);
	    ret["dCyrdA"]=aeroC["e"][1]+2*aeroC["e"][2]*alpha_+3*aeroC["e"][3]*pow(alpha_,2);
	    ret["dCzdA"]=(aeroC["f"][1]+2*aeroC["f"][2]*alpha_+3*aeroC["f"][3]*pow(alpha_,2)+4*aeroC["f"][4]*pow(alpha_,3))*(1-pow(beta_,2));
	    ret["dCzqdA"]=aeroC["g"][1]+2*aeroC["g"][2]*alpha_+3*aeroC["g"][3]*pow(alpha_,2)+4*aeroC["g"][4]*pow(alpha_,3);
	    ret["dCldA"]=aeroC["h"][1]*beta_+2*aeroC["h"][2]*alpha_*beta_+aeroC["h"][4]*pow(beta_,2)+3*aeroC["h"][5]*pow(alpha_,2)*beta_+4*aeroC["h"][6]*pow(alpha_,3)*beta_+2*aeroC["h"][7]*alpha_*pow(beta_,2);
	    ret["dClpdA"]=aeroC["i"][1]+2*aeroC["i"][2]*alpha_+3*aeroC["i"][3]*pow(alpha_,2);
	    ret["dClrdA"]=aeroC["j"][1]+2*aeroC["j"][2]*alpha_+3*aeroC["j"][3]*pow(alpha_,2)+4*aeroC["j"][4]*pow(alpha_,3);
	    ret["dCldadA"]=aeroC["k"][1]+2*aeroC["k"][3]*alpha_+aeroC["k"][4]*beta_+2*aeroC["k"][5]*alpha_*beta_+3*aeroC["k"][6]*pow(alpha_,2);
	    ret["dCldrdA"]=aeroC["l"][1]+aeroC["l"][3]*beta_+2*aeroC["l"][4]*alpha_*beta_+3*aeroC["l"][5]*pow(alpha_,2)*beta_;
	    ret["dCmdA"]=aeroC["m"][1]+aeroC["m"][3]*de_+2*aeroC["m"][5]*alpha_*de_+aeroC["m"][7]*pow(de_,2);
	    ret["dCmqdA"]=aeroC["n"][1]+2*aeroC["n"][2]*alpha_+3*aeroC["n"][3]*pow(alpha_,2)+4*aeroC["n"][4]*pow(alpha_,3)+5*aeroC["n"][5]*pow(alpha_,4);
	    ret["dCndA"]=aeroC["o"][1]*beta_+aeroC["o"][3]*pow(beta_,2)+2*aeroC["o"][4]*alpha_*beta_+2*aeroC["o"][5]*alpha_*pow(beta_,2)+3*aeroC["o"][6]*pow(alpha_,2)*beta_;
	    ret["dCnpdA"]=aeroC["p"][1]+2*aeroC["p"][2]*alpha_+3*aeroC["p"][3]*pow(alpha_,2)+4*aeroC["p"][4]*pow(alpha_,3);
	    ret["dCnrdA"]=aeroC["q"][1]+2*aeroC["q"][2]*alpha_;
	    ret["dCndadA"]=aeroC["r"][1]+aeroC["r"][3]*beta_+2*aeroC["r"][4]*alpha_*beta_+3*aeroC["r"][5]*pow(alpha_,2)*beta_+2*aeroC["r"][6]*alpha_+3*aeroC["r"][7]*pow(alpha_,2)+aeroC["r"][9]*pow(beta_,3);
	    ret["dCndrdA"]=aeroC["s"][1]+aeroC["s"][3]*beta_+2*aeroC["s"][4]*alpha_*beta_+2*aeroC["s"][5]*alpha_;
        //β微分
	    ret["dCxdB"]=0;
        ret["dCxqdB"]=0;
	    ret["dCydB"]=aeroC["c"][0];
        ret["dCypdB"]=0;
        ret["dCyrdB"]=0;
	    ret["dCzdB"]=-2*beta_*(aeroC["f"][0]+aeroC["f"][1]*alpha_+aeroC["f"][2]*pow(alpha_,2)+aeroC["f"][3]*pow(alpha_,3)+aeroC["f"][4]*pow(alpha_,4));
	    ret["dCzqdB"]=0;
	    ret["dCldB"]=aeroC["h"][0]+aeroC["h"][1]*alpha_+aeroC["h"][2]*pow(alpha_,2)+2*aeroC["h"][3]*beta_+2*aeroC["h"][4]*alpha_*beta_+aeroC["h"][5]*pow(alpha_,3)+aeroC["h"][6]*pow(alpha_,4)+2*aeroC["h"][7]*pow(alpha_,2)*beta_;
	    ret["dClpdB"]=0;
	    ret["dClrdB"]=0;
	    ret["dCldadB"]=aeroC["k"][2]+aeroC["k"][4]*alpha_+aeroC["k"][5]*pow(alpha_,2);
	    ret["dCldrdB"]=aeroC["l"][2]+aeroC["l"][3]*alpha_+aeroC["l"][4]*pow(alpha_,2)+aeroC["l"][5]*pow(alpha_,3)+2*aeroC["l"][6]*beta_;
	    ret["dCmdB"]=0;
	    ret["dCmqdB"]=0;
	    ret["dCndB"]=aeroC["o"][0]+aeroC["o"][1]*alpha_+2*aeroC["o"][2]*beta_+2*aeroC["o"][3]*alpha_*beta_+aeroC["o"][4]*pow(alpha_,2)+2*aeroC["o"][5]*pow(alpha_,2)*beta_+aeroC["o"][6]*pow(alpha_,3);
	    ret["dCnpdB"]=0;
	    ret["dCnrdB"]=0;
	    ret["dCndadB"]=aeroC["r"][2]+aeroC["r"][3]*alpha_+aeroC["r"][4]*pow(alpha_,2)+aeroC["r"][5]*pow(alpha_,3)+3*aeroC["r"][8]*pow(beta_,2)+3*aeroC["r"][9]*alpha_*pow(beta_,2);
	    ret["dCndrdB"]=aeroC["s"][2]+aeroC["s"][3]*alpha_+aeroC["s"][4]*pow(alpha_,2);
        //舵角微分
        ret["dCxdde"]=2*aeroC["a"][2]*de_+aeroC["a"][3]+aeroC["a"][4]*alpha_;
	    ret["dCydda"]=aeroC["c"][1];
	    ret["dCyddr"]=aeroC["c"][2];
	    ret["dCzdde"]=aeroC["f"][5];
	    ret["dCmdde"]=aeroC["m"][2]+aeroC["m"][3]*alpha_+2*aeroC["m"][4]*de_+aeroC["m"][5]*pow(alpha_,2)+3*aeroC["m"][6]*pow(de_,2)+2*aeroC["m"][7]*alpha_*de_;
        //造波抵抗
        ret["dCdwdM"]=ret["Cdw"]*(8.0*(dm-1.0)/(dm*d1)-(mach_-kdwm)*(pow(mach_-kdwm,2)-1.0)/d2);
        //2階微分(α+何か)
        ret["dCxadA"]=2*aeroC["a"][5]+6*aeroC["a"][6]*alpha_;
        ret["dCxadB"]=0;
        ret["dCxqadA"]=2*aeroC["b"][2]+6*aeroC["b"][3]*alpha_+12*aeroC["b"][4]*pow(alpha_,2);
        ret["dCxqadB"]=0;
        ret["dCzadA"]=(2*aeroC["f"][2]+6*aeroC["f"][3]*alpha_+12*aeroC["f"][4]*pow(alpha_,2))*(1-pow(beta_,2));
        ret["dCzadB"]=-2*beta_*(aeroC["f"][1]+2*aeroC["f"][2]*alpha_+3*aeroC["f"][3]*pow(alpha_,2)+4*aeroC["f"][4]*pow(alpha_,3));
	    ret["dCzqadA"]=2*aeroC["g"][2]+6*aeroC["g"][3]*alpha_+12*aeroC["g"][4]*pow(alpha_,2);
	    ret["dCzqadB"]=0;
        ret["dCxadde"]=aeroC["a"][4];
        ret["dCzadde"]=0;
    }
    return std::move(ret);
}

void exportSixDoFFighter(py::module& m)
{
    using namespace pybind11::literals;
    auto clsObj=EXPOSE_CLASS(SixDoFFighter)
    DEF_FUNC(SixDoFFighter,validate)
    DEF_FUNC(SixDoFFighter,calcOptimumCruiseFuelFlowRatePerDistance)
    DEF_FUNC(SixDoFFighter,trim6)
    DEF_FUNC(SixDoFFighter,calcAeroV)
    DEF_FUNC(SixDoFFighter,calcDerivative)
    DEF_FUNC(SixDoFFighter,calcMotion)
    DEF_READWRITE(SixDoFFighter,S)
    DEF_READWRITE(SixDoFFighter,Le)
    DEF_READWRITE(SixDoFFighter,b)
    DEF_READWRITE(SixDoFFighter,mac)
    DEF_READWRITE(SixDoFFighter,XcgR)
    DEF_READWRITE(SixDoFFighter,Xcg)
    DEF_READWRITE(SixDoFFighter,I)
    DEF_READWRITE(SixDoFFighter,Iinv)
    DEF_READWRITE(SixDoFFighter,deLimit)
    DEF_READWRITE(SixDoFFighter,daLimit)
    DEF_READWRITE(SixDoFFighter,drLimit)
    DEF_READWRITE(SixDoFFighter,deMaxRate)
    DEF_READWRITE(SixDoFFighter,daMaxRate)
    DEF_READWRITE(SixDoFFighter,drMaxRate)
    DEF_READWRITE(SixDoFFighter,deTimeConstant)
    DEF_READWRITE(SixDoFFighter,daTimeConstant)
    DEF_READWRITE(SixDoFFighter,drTimeConstant)
    DEF_READWRITE(SixDoFFighter,cdwTable)
    DEF_READWRITE(SixDoFFighter,de)
    DEF_READWRITE(SixDoFFighter,da)
    DEF_READWRITE(SixDoFFighter,dr)
    DEF_READWRITE(SixDoFFighter,T)
    ;
    EXPOSE_INNER_CLASS(clsObj,SixDoFFighter,FlightController)
    DEF_FUNC(SixDoFFighter::FlightController,getDefaultCommand)
    DEF_FUNC(SixDoFFighter::FlightController,calc)
    .def("calc",[](SixDoFFighter::FlightController& v,const py::object &cmd){
        return v.calc(cmd);
    })
    DEF_FUNC(SixDoFFighter::FlightController,calcDirect)
    .def("calcDirect",[](SixDoFFighter::FlightController& v,const py::object &cmd){
        return v.calcDirect(cmd);
    })
    DEF_FUNC(SixDoFFighter::FlightController,calcFromDirAndVel)
    .def("calcFromManualInput",[](SixDoFFighter::FlightController& v,const py::object &cmd){
        return v.calcFromManualInput(cmd);
    })
    DEF_FUNC(SixDoFFighter::FlightController,calcFromDirAndVel)
    .def("calcFromDirAndVel",[](SixDoFFighter::FlightController& v,const py::object &cmd){
        return v.calcFromDirAndVel(cmd);
    })
    DEF_FUNC(SixDoFFighter::FlightController,calcU)
    .def("calcU",[](SixDoFFighter::FlightController& v,const py::object &cmd,const Eigen::VectorXd& x){
        return v.calcU(cmd,x);
    })
    DEF_READWRITE(SixDoFFighter::FlightController,PositiveNzLimit)
    DEF_READWRITE(SixDoFFighter::FlightController,NegativeNzLimit)
    DEF_READWRITE(SixDoFFighter::FlightController,maxNyrCmd)
    DEF_READWRITE(SixDoFFighter::FlightController,maxPsCmd)
    DEF_READWRITE(SixDoFFighter::FlightController,kPsCmdP)
    DEF_READWRITE(SixDoFFighter::FlightController,kPsCmdD)
    DEF_READWRITE(SixDoFFighter::FlightController,kNzCmdP)
    DEF_READWRITE(SixDoFFighter::FlightController,kNzCmdD)
    DEF_READWRITE(SixDoFFighter::FlightController,kNyrCmdP)
    DEF_READWRITE(SixDoFFighter::FlightController,kNyrCmdD)
    DEF_READWRITE(SixDoFFighter::FlightController,altitudeKeeper)
    DEF_READWRITE(SixDoFFighter::FlightController,Ak)
    DEF_READWRITE(SixDoFFighter::FlightController,Bk)
    DEF_READWRITE(SixDoFFighter::FlightController,Qk)
    DEF_READWRITE(SixDoFFighter::FlightController,Rk)
    DEF_READWRITE(SixDoFFighter::FlightController,P)
    DEF_READWRITE(SixDoFFighter::FlightController,K)
    DEF_READWRITE(SixDoFFighter::FlightController,buffer)
    DEF_READWRITE(SixDoFFighter::FlightController,integrals)
    ;
    EXPOSE_CLASS(MorelliFighter)
    DEF_FUNC(MorelliFighter,calcAeroV)
    DEF_READWRITE(MorelliFighter,aeroC)
    ;
    EXPOSE_CLASS(StevensFighter)
    DEF_FUNC(StevensFighter,calcAeroV)
    DEF_READWRITE(StevensFighter,dampTable)
    DEF_READWRITE(StevensFighter,cxTable)
    DEF_READWRITE(StevensFighter,clTable)
    DEF_READWRITE(StevensFighter,cmTable)
    DEF_READWRITE(StevensFighter,cnTable)
    DEF_READWRITE(StevensFighter,dldaTable)
    DEF_READWRITE(StevensFighter,dldrTable)
    DEF_READWRITE(StevensFighter,dndaTable)
    DEF_READWRITE(StevensFighter,dndrTable)
    DEF_READWRITE(StevensFighter,czTable)
    DEF_READWRITE(StevensFighter,alphaTable)
    DEF_READWRITE(StevensFighter,betaSymTable)
    DEF_READWRITE(StevensFighter,betaAsymTable)
    DEF_READWRITE(StevensFighter,deTable)
    DEF_READWRITE(StevensFighter,dCydB)
    DEF_READWRITE(StevensFighter,dCydda)
    DEF_READWRITE(StevensFighter,dCyddr)
    DEF_READWRITE(StevensFighter,dCzdde)
    ;
}
