// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "CoordinatedFighter.h"
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

SimpleFighterJetEngine::SimpleFighterJetEngine(const nl::json& modelConfig_,const nl::json& instanceConfig_)
    :Propulsion(modelConfig_,instanceConfig_),
    tminTable(6,6),tmilTable(6,6),tmaxTable(11,6),
    alts(6),machs(6),machsEx(11){
    if(isDummy){return;}
    enableInstantResponse=getValueFromJsonKRD(modelConfig,"enableInstantResponse",randomGen,false);
    considerFuel=getValueFromJsonKRD(modelConfig,"considerFuel",randomGen,true);
    tHalf=getValueFromJsonKRD(modelConfig,"tHalf",randomGen,0.77);
    tminTable=getValueFromJsonKRD(modelConfig,"tmin",randomGen,Eigen::Tensor<double,2>(6,6).setValues({
        {1060.0,670.0,880.0,1140.0,1500.0,1860.0},
		{635.0,425.0,690.0,1010.0,1330.0,1700.0},
		{60.0,25.0,345.0,755.0,1130.0,1525.0},
		{-1020.0,-710.0,-300.0,350.0,910.0,1360.0},
		{-2700.0,-1900.0,-1300.0,-247.0,600.0,1100.0},
		{-3600.0,-1400.0,-595.0,-342.0,-200.0,700.0}
    }));
    tmilTable=getValueFromJsonKRD(modelConfig,"tmil",randomGen,Eigen::Tensor<double,2>(6,6).setValues({
        {12680.0,9150.0,6200.0,3950.0,2450.0,1400.0},
		{12680.0,9150.0,6313.0,4040.0,2470.0,1400.0},
		{12610.0,9312.0,6610.0,4290.0,2600.0,1560.0},
		{12640.0,9839.0,7090.0,4660.0,2840.0,1660.0},
		{12390.0,10176.0,7750.0,5320.0,3250.0,1930.0},
		{11680.0,9848.0,8050.0,6100.0,3800.0,2310.0}
    }));
    tmaxTable=getValueFromJsonKRD(modelConfig,"tmax",randomGen,Eigen::Tensor<double,2>(11,6).setValues({
        {20000.0,15000.0,10800.0,7000.0,4000.0,2500.0},
		{21420.0,15700.0,11225.0,7323.0,4435.0,2600.0},
		{22700.0,16860.0,12250.0,8154.0,5000.0,2835.0},
		{24240.0,18910.0,13760.0,9285.0,5700.0,3215.0},
		{26070.0,21075.0,15975.0,11115.0,6860.0,3950.0},
		{28886.0,23319.0,18300.0,13484.0,8642.0,5057.0},
		{32000.0,26000.0,20500.0,15500.0,10400.0,6000.0},
		{34957.0,28800.0,22800.0,17500.0,11800.0,6800.0},
		{37557.0,30700.0,24200.0,18600.0,12800.0,7400.0},
		{39378.5,32000.0,25400.0,19400.0,13600.0,7800.0},
		{40400.0,32950.0,26300.0,20000.0,14000.0,8200.0}
    }));
    fuelFlowRateMin=getValueFromJsonKRD(modelConfig,"fuelFlowRateMin",randomGen,1006.0);
    fuelFlowRateMil=getValueFromJsonKRD(modelConfig,"fuelFlowRateMil",randomGen,8888.0);
    fuelFlowRateMax=getValueFromJsonKRD(modelConfig,"fuelFlowRateMax",randomGen,40123.0);
    alts=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"alt",randomGen,
        (Eigen::Matrix<double,6,1>()<<0.0,10000.0,20000.0,30000.0,40000.0,50000.0).finished()
    );
    machs=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"mach",randomGen,
        (Eigen::Matrix<double,6,1>()<<0.0,0.2,0.4,0.6,0.8,1.0).finished()
    );
    machsEx=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"machEx",randomGen,
        (Eigen::Matrix<double,11,1>()<<0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0).finished()
    );
    points.push_back(machs);
    points.push_back(alts);
    pointsEx.push_back(machsEx);
    pointsEx.push_back(alts);
    power=0.0;
    pCmd=0.0;
    thrust=0.0;
}
SimpleFighterJetEngine::~SimpleFighterJetEngine(){}
double SimpleFighterJetEngine::getFuelFlowRate(){
    if(considerFuel){
        if(power<0.5){
            return lb2kg(fuelFlowRateMin+(fuelFlowRateMil-fuelFlowRateMin)*(power/0.5))/3600.0;
        }else{
            return lb2kg(fuelFlowRateMil+(fuelFlowRateMax-fuelFlowRateMil)*(power/0.5-1.0))/3600.0;
        }
    }else{
        return 0.0;
    }
}
double SimpleFighterJetEngine::getThrust(){
    return thrust;
}
double SimpleFighterJetEngine::calcFuelFlowRate(const nl::json& args){
    double arg_power=args.at("power");
    if(considerFuel){
        if(arg_power<0.5){
            return lb2kg(fuelFlowRateMin+(fuelFlowRateMil-fuelFlowRateMin)*(arg_power/0.5))/3600.0;
        }else{
            return lb2kg(fuelFlowRateMil+(fuelFlowRateMax-fuelFlowRateMil)*(arg_power/0.5-1.0))/3600.0;
        }
    }else{
        return 0.0;
    }
}
double SimpleFighterJetEngine::calcThrust(const nl::json& args){
    double arg_power=args.at("power");
    double arg_alt=args.at("alt");
    double arg_rmach=args.at("rmach");
    return calcThrust(arg_power,arg_alt,arg_rmach);
}
void SimpleFighterJetEngine::setPowerCommand(const double& pCmd_){
    pCmd=pCmd_;
    auto p=getShared<Fighter>(parent);
    if(enableInstantResponse && p->isAlive()){
        double alt=-p->motion.pos(2);
        std::map<std::string,double> atm=atmosphere(alt);
        double V=p->motion.vel.norm();
        double Mach=V/atm["a"];
        power=pCmd;
        thrust=calcThrust(power,alt,Mach);
    }
}
double SimpleFighterJetEngine::calcThrust(double power_,double alt,double rmach){
    Eigen::MatrixXd x(1,2);
    x<<rmach,m2ft(alt);
    double tmin=interpn(points,tminTable,x)(0);
    double tmil=interpn(points,tmilTable,x)(0);
    double tmax=interpn(pointsEx,tmaxTable,x)(0);
    if(power_<0.5){
        return lbf2N(tmin+(tmil-tmin)*(power_/0.5));
    }else{
        return lbf2N(tmil+(tmax-tmil)*(power_/0.5-1.0));
    }
}
double SimpleFighterJetEngine::trimPower(double tgtThrust,double alt,double Mach){
	double thrustMin=calcThrust(0.0,alt,Mach);
	double thrustMax=calcThrust(1.0,alt,Mach);
	auto sub=[&](double pCmd){
		return calcThrust(pCmd,alt,Mach)-tgtThrust;
    };
    double trimmed;
	if(thrustMin>=tgtThrust){
		trimmed=0.0;
    }else if(thrustMax<=tgtThrust){
		trimmed=1.0;
    }else{
        boost::math::tools::eps_tolerance<double> tol(16);
        boost::uintmax_t max_iter=10;
        try{
            auto result=boost::math::tools::toms748_solve(sub,0.0,1.0,tol,max_iter);
            trimmed=(result.first+result.second)/2;
        }catch(std::exception& e){
            std::cout<<"at trimPower"<<std::endl;
            DEBUG_PRINT_EXPRESSION(sub(0.0))
            DEBUG_PRINT_EXPRESSION(sub(1.0))
            DEBUG_PRINT_EXPRESSION(tgtThrust)
            DEBUG_PRINT_EXPRESSION(alt)
            DEBUG_PRINT_EXPRESSION(Mach)
            throw e;
        }
    }
	return trimmed;
}
double SimpleFighterJetEngine::calcPCmdFromThrottle(double throttle){
    if(throttle>=tHalf){
        return 0.5*(1+(throttle-tHalf)/(1-tHalf));
    }else{
        return throttle/tHalf;
    }
}
double SimpleFighterJetEngine::calcdTdP(double power_,double alt,double rmach){
    Eigen::MatrixXd x(1,2);
    x<<rmach,m2ft(alt);
    double tmin=interpn(points,tminTable,x)(0);
    double tmil=interpn(points,tmilTable,x)(0);
    double tmax=interpn(pointsEx,tmaxTable,x)(0);
    if(power_<0.5){
        return lbf2N(2*(tmil-tmin));
    }else{
        return lbf2N(2*(tmax-tmil));
    }
}
double SimpleFighterJetEngine::calcdTdH(double power_,double alt,double rmach){
    Eigen::MatrixXd x(1,2);
    std::pair<Eigen::Index,double> tmp=_search1d(points[1],m2ft(alt));
    double alt_h=points[1](tmp.first+1);
    double alt_l=points[1](tmp.first);
    x<<rmach,alt_l;
    double tmin_l=interpn(points,tminTable,x)(0);
    double tmil_l=interpn(points,tmilTable,x)(0);
    double tmax_l=interpn(pointsEx,tmaxTable,x)(0);
    x<<rmach,alt_h;
    double tmin_h=interpn(points,tminTable,x)(0);
    double tmil_h=interpn(points,tmilTable,x)(0);
    double tmax_h=interpn(pointsEx,tmaxTable,x)(0);
    if(power_<0.5){
        return lbf2N(2*power_*(tmil_h-tmin_h-tmil_l+tmil_l))/ft2m(alt_h-alt_l);
    }else{
        return lbf2N((2*power_-1)*(tmax_h-tmil_h-tmax_l+tmil_l))/ft2m(alt_h-alt_l);
    }
}
double SimpleFighterJetEngine::calcdTdM(double power_,double alt,double rmach){
    Eigen::MatrixXd x(1,2);
    std::pair<Eigen::Index,double> tmp=_search1d(points[0],rmach);
    double rmach_h=points[0](tmp.first+1);
    double rmach_l=points[0](tmp.first);
    tmp=_search1d(pointsEx[0],rmach);
    double rmachEx_h=pointsEx[0](tmp.first+1);
    double rmachEx_l=pointsEx[0](tmp.first);
    x<<rmach_l,m2ft(alt);
    double tmin_l=interpn(points,tminTable,x)(0);
    double tmil_l=interpn(points,tmilTable,x)(0);
    x<<rmachEx_l,m2ft(alt);
    double tmax_l=interpn(pointsEx,tmaxTable,x)(0);
    x<<rmach_h,m2ft(alt);
    double tmin_h=interpn(points,tminTable,x)(0);
    double tmil_h=interpn(points,tmilTable,x)(0);
    x<<rmachEx_h,m2ft(alt);
    double tmax_h=interpn(pointsEx,tmaxTable,x)(0);
    if(power_<0.5){
        return lbf2N(2*power_*((tmil_h-tmil_l)-(tmin_h-tmin_l))/(rmach_h-rmach_l));
    }else{
        return lbf2N((2*power_-1)*((tmax_h-tmax_l)/(rmachEx_h-rmachEx_l)-(tmil_h-tmil_l)/(rmach_h-rmach_l)));
    }
}
double SimpleFighterJetEngine::rtau(double dp){
    if(dp<=0.25){
        return 1.0;
    }else if(dp>=0.50){
        return 0.1;
    }else{
        return 1.9-3.6*dp;
    }
}
double SimpleFighterJetEngine::pdot(double pCurr_,double pCmd_){
    double p2,T;
    if(pCmd_>=0.50){
        if(pCurr_>=0.50){
            T=5.0;
            p2=pCmd_;
        }else{
            p2=0.60;
            T=rtau(p2-pCurr_);
        }
    }else{
        if(pCurr_>=0.50){
            T=5.0;
            p2=0.40;
        }else{
            p2=pCmd_;
            T=rtau(p2-pCurr_);
        }
    }
    return T*(p2-pCurr_);
}
void SimpleFighterJetEngine::behave(){
    auto p=getShared<Fighter>(parent);
    if(!enableInstantResponse && p->isAlive()){
        double alt=-p->motion.pos(2);
        std::map<std::string,double> atm=atmosphere(alt);
        double V=p->motion.vel.norm();
        double Mach=V/atm["a"];
        double dt=interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();

        std::function<Eigen::VectorXd (const Eigen::VectorXd& x)> deriv=[&](const Eigen::VectorXd& x){
            Eigen::VectorXd f(1);
            f(0)=pdot(std::clamp<double>(x(0),0,1),pCmd);
            return f;
        };
        Eigen::VectorXd x0(1);
        x0(0)=power;
        Eigen::VectorXd nextX=RK4(x0,dt,deriv);
        power=std::clamp<double>(nextX(0),0,1);
        thrust=calcThrust(power,alt,Mach);
    }
}

CoordinatedFighter::CoordinatedFighter(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter(modelConfig_,instanceConfig_),
cxTable(5,12),cmTable(5,12),czTable(12),trimmedDe(12),
aoaTable(12),deTable(5),cdwTable(4){
    if(isDummy){return;}
    m=getValueFromJsonKRD(modelConfig.at("dynamics"),"m",randomGen,lb2kg(20500.0));
    S=getValueFromJsonKRD(modelConfig.at("dynamics"),"S",randomGen,ft2m(ft2m(300.0)));
    rollMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"rollMax",randomGen,180.0));
    sideGLimit=getValueFromJsonKRD(modelConfig.at("dynamics"),"sideGLimit",randomGen,10.0);
    minAoa=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"minAoa",randomGen,-10.00));
    maxAoa=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"maxAoa",randomGen,36.35));
    cxTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cx",randomGen,Eigen::Tensor<double,2>(5,12).setValues({
        {-0.099,-0.081,-0.081,-0.063,-0.025,+0.044,+0.097,+0.113,+0.145,+0.167,+0.174,+0.166},
		{-0.048,-0.038,-0.040,-0.021,+0.016,+0.083,+0.127,+0.137,+0.162,+0.177,+0.179,+0.167},
		{-0.022,-0.020,-0.021,-0.004,+0.032,+0.094,+0.128,+0.130,+0.154,+0.161,+0.155,+0.138},
		{-0.040,-0.038,-0.039,-0.025,+0.006,+0.062,+0.087,+0.085,+0.100,+0.110,+0.104,+0.091},
		{-0.083,-0.073,-0.076,-0.072,-0.046,+0.012,+0.024,+0.025,+0.043,+0.053,+0.047,+0.040}
    }));
    cmTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cm",randomGen,Eigen::Tensor<double,2>(5,12).setValues({
        {+0.205,+0.168,+0.186,+0.196,+0.213,+0.251,+0.245,+0.238,+0.252,+0.231,+0.198,+0.192},
		{+0.081,+0.077,+0.107,+0.110,+0.110,+0.141,+0.127,+0.119,+0.133,+0.108,+0.081,+0.093},
		{-0.046,-0.020,-0.009,-0.005,-0.006,+0.010,+0.006,-0.001,+0.014,+0.000,-0.013,+0.032},
        {-0.174,-0.145,-0.121,-0.127,-0.129,-0.102,-0.097,-0.113,-0.087,-0.084,-0.069,-0.006},
		{-0.259,-0.202,-0.184,-0.193,-0.199,-0.150,-0.160,-0.167,-0.104,-0.076,-0.041,-0.005}
    }));
    czTable=getValueFromJsonKRD(modelConfig.at("dynamics"),"cz",randomGen,Eigen::Tensor<double,1>(12).setValues({
		+0.770,+0.241,-0.100,-0.416,-0.731,-1.053,-1.366,-1.646,-1.917,-2.120,-2.248,-2.229
    }));
    cdwTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"cdw",randomGen,
        (Eigen::Matrix<double,4,1>()<<2.76e-2,1.95e-3,1.49e0,1.31e-1).finished()
    );
    aoaTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"aoa",randomGen,
        (Eigen::Matrix<double,12,1>()<<-10.0,-5.0,0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0).finished()
    );
    deTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig.at("dynamics"),"de",randomGen,
        (Eigen::Matrix<double,5,1>()<<-24.0,-12.0,0.0,12.0,24.0).finished()
    );
    for(int i=0;i<12;++i){
        trimmedDe(i)=cmtrim(deg2rad(aoaTable(i)));
    }
	//その他の位置、姿勢等の運動状態に関する変数の初期化
    maxCz=calcAero(minAoa,0.0)(2);
    minCz=calcAero(maxAoa,0.0)(2);
    observables["spec"].merge_patch({
        {"dynamics",{
            {"rollMax",rollMax}
        }}
    });
}
CoordinatedFighter::~CoordinatedFighter(){
}
double CoordinatedFighter::calcOptimumCruiseFuelFlowRatePerDistance(){
    //high computational cost, so once you obtained the value, should write it in json modelConfig.
    auto eng=getShared<SimpleFighterJetEngine>(engine);
    double ffr=eng->getFuelFlowRate();
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
            //trim α for level flight
            double dp=atm["rho"]*V*V/2;
            double tgtCz=-relItoB(Eigen::Vector3d(0,0,m*gravity))(2)/(dp*S);
            double aoa_=trimAoa(tgtCz);
	        Eigen::Vector3d CFb=calcAero(aoa_,Mach);
	        double tgtThrust=-CFb(0)*dp*S;
	        double pCmd_=eng->trimPower(tgtThrust,alt,Mach);
            double fuelFlowRate=eng->calcFuelFlowRate({{"power",pCmd_}});
            if(!(
                abs(tgtThrust-eng->calcThrust(pCmd_,alt,Mach))<1.0 &&
                minCz<=tgtCz &&
                maxCz>=tgtCz
            )){//infeasible to keep steady level flight
                return eng->calcFuelFlowRate({{"power",1.0}});
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
double CoordinatedFighter::trimAoa(double tgtCz){
    double trimmed;
	if(minCz>=tgtCz){
		trimmed=maxAoa;
    }else if(maxCz<=tgtCz){
		trimmed=minAoa;
    }else{
	    auto sub=[&](double aoa_){
    		return calcAero(aoa_,0.0)(2)-tgtCz;
        };
        boost::math::tools::eps_tolerance<double> tol(16);
        boost::uintmax_t max_iter=10;
        try{
            auto result=boost::math::tools::toms748_solve(sub,minAoa,maxAoa,tol,max_iter);
            trimmed=(result.first+result.second)/2;
        }catch(std::exception& e){
            std::cout<<"at trimAoa"<<std::endl;
            DEBUG_PRINT_EXPRESSION(sub(minAoa))
            DEBUG_PRINT_EXPRESSION(sub(maxAoa))
            DEBUG_PRINT_EXPRESSION(tgtCz)
            throw e;
        }
    }
	return trimmed;
}
double CoordinatedFighter::cm(double aoa_,double de_){
    std::vector<Eigen::VectorXd> points;
    points.push_back(deTable);
    points.push_back(aoaTable);
    Eigen::MatrixXd x(1,2);
    x<<rad2deg(de_),rad2deg(aoa_);
    return interpn(points,cmTable,x)(0);
}
double CoordinatedFighter::cmtrim(double aoa_){
    double deMin=deg2rad(-15.0);
    double deMax=deg2rad(15.0);
	if(cm(aoa_,deMin)<=0.0){
		return deMin;
    }else if(cm(aoa_,deMax)>=0.0){
		return deMax;
    }else{
	    auto sub=[&](double de_){
    		return cm(aoa_,de_);
        };
        boost::math::tools::eps_tolerance<double> tol(16);
        boost::uintmax_t max_iter=10;
        try{
            auto result=boost::math::tools::toms748_solve(sub,deMin,deMax,tol,max_iter);
            return (result.first+result.second)/2;
        }catch(std::exception& e){
            std::cout<<"cmtrim"<<std::endl;
            DEBUG_PRINT_EXPRESSION(sub(deMin))
            DEBUG_PRINT_EXPRESSION(sub(deMax))
            DEBUG_PRINT_EXPRESSION(aoa_)
            throw e;
        }
    }
}
double CoordinatedFighter::cx(double aoa_){
    std::vector<Eigen::VectorXd> points;
    Eigen::MatrixXd x(1,1);
    x<<rad2deg(aoa_);
    points.push_back(aoaTable);
    double de_=interpn(points,trimmedDe,x)(0);
    points.insert(points.begin(),deTable);
    x=Eigen::MatrixXd::Zero(1,2);
    x<<rad2deg(de_),rad2deg(aoa_);
    return interpn(points,cxTable,x)(0);
}
double CoordinatedFighter::cz(double aoa_){
    std::vector<Eigen::VectorXd> points;
    Eigen::MatrixXd x(1,1);
    x<<rad2deg(aoa_);
    points.push_back(aoaTable);
    double de_=interpn(points,trimmedDe,x)(0);
    return interpn(points,czTable,x)(0)-de_*0.019/deg2rad(25.0);
}
double CoordinatedFighter::cxa(double aoa_){
    std::vector<Eigen::VectorXd> points;
    Eigen::MatrixXd x(1,1);
    x<<rad2deg(aoa_);
    points.push_back(aoaTable);
    double de_=interpn(points,trimmedDe,x)(0);
    double ddeda_=interpgradn(points,trimmedDe,x)(0,0);
    points.insert(points.begin(),deTable);
    x=Eigen::MatrixXd::Zero(1,2);
    x<<rad2deg(de_),rad2deg(aoa_);
    Eigen::MatrixXd grad=interpgradn(points,cxTable,x);
    double ret=(grad(1,0)+grad(0,0)*ddeda_)*(180/M_PI);
    return ret;
}
double CoordinatedFighter::cza(double aoa_){
    std::vector<Eigen::VectorXd> points;
    Eigen::MatrixXd x(1,1);
    x<<rad2deg(aoa_);
    points.push_back(aoaTable);
    double ddeda_=interpgradn(points,trimmedDe,x)(0,0);
    double grad=interpgradn(points,czTable,x)(0,0);
     double ret=(grad-ddeda_*0.019/deg2rad(25.0))*(180/M_PI);
    return ret;
}
double CoordinatedFighter::cl(double aoa_){
    return cx(aoa_)*sin(aoa_)-cz(aoa_)*cos(aoa_);
}
double CoordinatedFighter::cd(double aoa_,double mach_){
    return -cx(aoa_)*cos(aoa_)-cz(aoa_)*sin(aoa_)+cdw(mach_);
}
double CoordinatedFighter::cdw(double mach_){
    //wave drag is based on [3] and [4].
    //coefficients are obtained by curve fitting with some points.
    double cdw0=cdwTable(0);
    double kdwm=cdwTable(1);
    double kdw=cdwTable(2);
    double dm=cdwTable(3);
    double fm=1.0/(1.0+exp(-8*(mach_-(1.0-dm/2.0))/dm));
    return fm*cdw0*kdw/pow(pow(pow(mach_-kdwm,2)-1.0,2)+pow(kdw,4),0.25);
}
Eigen::Vector3d CoordinatedFighter::calcAero(double aoa_,double mach_){
	/*
	"Coordinated" flight assumes de=da=dr=β=0.
	Angle of attack is only considered for aerodynamic force and the aircraft attitude neglects it.
	CX=-CD=Cx(α)*cos(α)+Cz(α)*sin(α)
	CZ=-CL=Cz(α)*cos(α)-Cx(α)*sin(α)
	Polynomial coefficients of Cx and Cz is based on [2] and the lift is maximized when α=0.66449rad
	*/
    double Cx=cx(aoa_);
    double Cz=cz(aoa_);
    return Eigen::Vector3d(Cx*cos(aoa_)+Cz*sin(aoa_)-cdw(mach_),0.0,Cz*cos(aoa_)-Cx*sin(aoa_));
}
void CoordinatedFighter::calcMotion(double dt){
    vel_prev=motion.vel;
    pos_prev=motion.pos;
    Quaternion qPrev=motion.q;
    double alt=-motion.pos(2);
    double V=motion.vel.norm();
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    double dp=atm["rho"]*V*V/2;
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    double thrust=engine.lock()->getThrust();
    double aoa=ctrl.at("aoa");
    double dRoll=ctrl.at("dRoll");
    Eigen::Vector3d rollAx=motion.vel/V;
    //Roll rotation(Immidiate)
    if(abs(dRoll)>1e-8){
        Quaternion dq1=Quaternion::fromAngle(rollAx,dRoll);
        motion.q=dq1*motion.q;
    }else{
        Eigen::VectorXd dq1=motion.q.dqdwi()*rollAx*dRoll;
        motion.q=(motion.q+Quaternion(dq1)).normalized();
    }
    //position and velocity update
    Eigen::Vector3d CFb=calcAero(aoa,Mach);
    Eigen::Vector3d F=relBtoI(Eigen::Vector3d(thrust*cos(aoa),0.,-thrust*sin(aoa))+CFb*(dp*S))+Eigen::Vector3d(0.,0.,m*gravity);
	Eigen::Vector3d accel=F/m;
    double accelScalar=accel.dot(rollAx);
    Eigen::Vector3d omegaV=rollAx.cross(accel/V);
    double rotAngle=omegaV.norm()*dt;
    if(rotAngle<1e-8){
        motion.pos+=motion.vel*dt+accel*dt*dt/2.;
        motion.vel+=accel*dt;
        Eigen::VectorXd dq2=motion.q.dqdwi()*omegaV*dt;
        motion.q=(motion.q+Quaternion(dq2)).normalized();
    }else{
        Eigen::Vector3d ex0=motion.vel/V;
        Eigen::Vector3d ey0=omegaV.cross(ex0).normalized();
        double wt=rotAngle;
        double acc=accelScalar;
        Eigen::Vector3d vn=(ex0*cos(wt)+ey0*sin(wt)).normalized();
        motion.vel=(V+acc*dt)*vn;
        Eigen::Vector3d dr=V*dt*(ex0*sinc(wt)+ey0*oneMinusCos_x2(wt)*wt)+acc*dt*dt*(ex0*(sinc(wt)-oneMinusCos_x2(wt))+ey0*wt*(sincMinusOne_x2(wt)+oneMinusCos_x2(wt)));
        motion.pos+=dr;
        accel=(motion.vel-vel_prev)/dt;
        Quaternion dq2=Quaternion::fromAngle(omegaV/(rotAngle/dt),rotAngle);
        motion.q=dq2*motion.q;
    }
    Quaternion dq=motion.q*qPrev.conjugate();
    Eigen::Vector3d ax=motion.q.vec();
    double sn=ax.norm();
    if(sn>0){ax/=sn;}
    motion.omega=ax*atan2(sn,dq.w())+rollAx*dRoll/dt;
    motion.time=manager->getTime()+dt;
    motion.calcQh();
}

CoordinatedFighter::FlightController::FlightController(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Fighter::FlightController(modelConfig_,instanceConfig_){
    if(isDummy){return;}
    mode="fromDirAndVel";
    lambdaVel=getValueFromJsonKRD(modelConfig,"lambdaVel",randomGen,1.0);
    lambdaTheta=getValueFromJsonKRD(modelConfig,"lambdaTheta",randomGen,1.0);
    clampTheta=deg2rad(getValueFromJsonKRD(modelConfig,"clampTheta",randomGen,45.0));
    kVel=getValueFromJsonKRD(modelConfig,"kVel",randomGen,10.0);
    kAccel=getValueFromJsonKRD(modelConfig,"kAccel",randomGen,10.0);
    kPower=getValueFromJsonKRD(modelConfig,"kPower",randomGen,20.0);
    kTheta=getValueFromJsonKRD(modelConfig,"kTheta",randomGen,10000.0);
    kOmega=getValueFromJsonKRD(modelConfig,"kOmega",randomGen,10000.0);
    kAoa=getValueFromJsonKRD(modelConfig,"kAoa",randomGen,1000.0);
    kEps=getValueFromJsonKRD(modelConfig,"kEps",randomGen,0.1);
    pitchLimit=deg2rad(getValueFromJsonKRD(modelConfig,"pitchLimit",randomGen,0.0));
    pitchLimitThreshold=deg2rad(getValueFromJsonKRD(modelConfig,"pitchLimitThreshold",randomGen,0.0));
    if(modelConfig.contains("altitudeKeeper")){
        nl::json sub=modelConfig.at("altitudeKeeper");
        altitudeKeeper=AltitudeKeeper(sub);
    }
}
nl::json CoordinatedFighter::FlightController::getDefaultCommand(){
    return {
        {"roll",0.0},
        {"pitch",0.0},
        {"accel",0.0}
    };
}
nl::json CoordinatedFighter::FlightController::calc(const nl::json &cmd){
    if(mode=="fromDirAndVel"){
        return calcFromDirAndVel(cmd);
    }else{
        return calcDirect(cmd);
    }
}
nl::json CoordinatedFighter::FlightController::calcDirect(const nl::json &cmd){
    auto p=getShared<CoordinatedFighter>(parent);
    auto roll=std::clamp(cmd.at("roll").get<double>(),-1.,1.);
    auto pitch=std::clamp(cmd.at("pitch").get<double>(),-1.,1.);
    double alt=-p->motion.pos(2);
    double V=p->motion.vel.norm();
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    double dp=atm["rho"]*V*V/2;
    double tgtCz=-p->relItoB(Eigen::Vector3d(0,0,p->m*gravity))(2)/(dp*p->S);
    double aoa_=std::clamp(p->trimAoa(tgtCz)+(p->maxAoa-p->minAoa)*pitch/2,p->minAoa,p->maxAoa);
	Eigen::Vector3d CFb=p->calcAero(aoa_,Mach);
	double tgtThrust=-CFb(0)*dp*p->S;
    auto eng=getShared<SimpleFighterJetEngine>(p->engine);
	double pCmd_=eng->trimPower(tgtThrust,alt,Mach);
    if(cmd.contains("throttle")){//0〜1
        pCmd_=std::clamp(cmd.at("throttle").get<double>(),0.,1.);
    }else if(cmd.contains("accel")){//-1〜+1
        double clampedCmd=std::clamp(cmd.at("accel").get<double>(),-1.,1.);
        if(clampedCmd>=0){
            pCmd_=pCmd_+(1.0-pCmd_)*clampedCmd;
        }else{
            pCmd_=pCmd_*(1+clampedCmd);
        }
    }else{
        throw std::runtime_error("Either 'throttle' or 'accel' must be in cmd keys.");
    }
	double dRoll_=p->rollMax*roll*interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
    //side G Limitter
    Eigen::Vector3d ey=p->relBtoI(Eigen::Vector3d(0,cos(dRoll_),sin(dRoll_)));
    Eigen::Vector3d ez=p->relBtoI(Eigen::Vector3d(0,-sin(dRoll_),cos(dRoll_)));
    double zz=p->calcAero(aoa_,Mach)(2)/(p->m*gravity)+ez(2);
    if(Eigen::Vector3d(0,ey(2),zz).norm() >= p->sideGLimit){
        if(zz>=0){
            tgtCz=p->m*util::gravity*(sqrt(p->sideGLimit-ey(2)*ey(2))-ez(2));
        }else{
            tgtCz=-p->m*util::gravity*(sqrt(p->sideGLimit-ey(2)*ey(2))-ez(2));
        }
        aoa_=p->trimAoa(tgtCz);
    }
    return {{"pCmd",pCmd_},{"aoa",aoa_},{"dRoll",dRoll_}};
}
nl::json CoordinatedFighter::FlightController::calcFromDirAndVel(const nl::json &cmd){
    auto p=getShared<CoordinatedFighter>(parent);
    double eps=1e-4;
    double alt=-p->motion.pos(2);
    double V=p->motion.vel.norm();
    Eigen::Vector3d d=p->motion.vel/V;
    std::map<std::string,double> atm=atmosphere(alt);
    double Mach=V/atm["a"];
    double dp=atm["rho"]*V*V/2;
    Eigen::Vector3d mg(0,0,p->m*gravity);
    Eigen::Vector3d mgVert=mg-(mg.dot(d))*d;//==g.dot(n)+g.dot(b) for any φ
    Eigen::Vector3d b,n;
    auto eng=getShared<SimpleFighterJetEngine>(p->engine);
    double Tmin=eng->calcThrust(0.0,alt,Mach);
	double Tmax=eng->calcThrust(1.0,alt,Mach);
    double dstVd,dstP;//velocity command
    double dstTd,dstAoa;//turn command
    int velCmdType,turnCmdType;
    if(cmd.contains("dstV")){
        velCmdType=0;
	    double dstV=cmd.at("dstV");
		dstVd=-lambdaVel*(V-dstV);
    }else if(cmd.contains("dstAccel")){
        velCmdType=1;
		double dstAccel=cmd.at("dstAccel");
        dstVd=dstAccel;
    }else if(cmd.contains("dstThrust")){
        velCmdType=2;
		double dstT=cmd.at("dstThrust");
		dstP=std::clamp((dstT-Tmin)/(Tmax-Tmin),0.0,1.0);
    }else if(cmd.contains("dstThrottle")){
        velCmdType=2;
        double dstT=eng->calcThrust(cmd.at("dstThrottle"),alt,Mach);
		dstP=std::clamp((dstT-Tmin)/(Tmax-Tmin),0.0,1.0);
    }else{
		throw std::runtime_error("Only one of dstV, dstAccel, dstThrust or dstThrottle is acceptable.");
    }
	if(cmd.contains("dstDir")){
        turnCmdType=0;
		Eigen::Vector3d dstDir=cmd.at("dstDir");
        if(cmd.contains("dstAlt")){
            dstDir=altitudeKeeper(p->motion,dstDir,cmd.at("dstAlt").get<double>());
        }
		dstDir.normalize();
		b=d.cross(dstDir);
		double snTheta=std::min(1.0,b.norm());
        double csTheta=std::clamp(d.dot(dstDir),-1.0,1.0);
        double eTheta=atan2(snTheta,csTheta);
		if(snTheta<eps){
    		//Almost no rotation or 180deg rotation
	    	b=mgVert.normalized();
            dstTd=lambdaTheta*std::min(eTheta,clampTheta);
        }else{
            b/=snTheta;
            //ピッチ制限の判定
            if(eTheta>=pitchLimitThreshold){//一定以上の回転指示
                b=pitchLimitter(p->motion,dstDir,pitchLimit);
            }
            dstTd=lambdaTheta*std::min(eTheta,clampTheta);
        }
    }else if(cmd.contains("dstTurnRate")){
        turnCmdType=1;
		Eigen::Vector3d dstTurnRate=cmd.at("dstTurnRate");
		if(dstTurnRate.norm()<eps){
			//Almost no rotation
			b=mgVert.normalized();
            dstTd=0;
        }else{
			b=d.cross(dstTurnRate.cross(d)).normalized();
            dstTd=b.dot(dstTurnRate);
        }
    }else if(cmd.contains("dstAlpha") && cmd.contains("ey")){
        turnCmdType=2;
		dstAoa=std::clamp<double>(cmd.at("dstAlpha"),p->minAoa,p->maxAoa);
        b=d.cross((cmd.at("ey").get<Eigen::Vector3d>()).cross(d)).normalized();
    }else{
		throw std::runtime_error("Only one of dstDir, dstTurnRate, or (dstAlpha & ey) is acceptable.");
    }
    n=b.cross(d);
    double gt=mg.dot(d)/p->m;
    double gb_v=mg.dot(b)/(p->m*V);
    double gn_v=mg.dot(n)/(p->m*V);
	std::function<double(unsigned int,const double*,double*,void*)> obj=[&](unsigned int n,const double* x,double* grad,void* data){//x=[T,alpha]
            //x[0]:power [0,1)
            //x[1]:aoa [minAoa,maxAoa]
            //x[2]:phi [-pi,pi)
			double ct=(Tmin+x[0]*(Tmax-Tmin));
			double aoa=x[1];
            double phi=x[2];
            double dps=dp*p->S;
			double cx_=p->cx(aoa)*dps;
			double cz_=p->cz(aoa)*dps;
            double ctp=Tmax-Tmin;
			double cxa_=p->cxa(aoa)*dps;
			double cza_=p->cza(aoa)*dps;
            double ft=((ct+cx_)*cos(aoa)+cz_*sin(aoa))/p->m;
            double ft_a=(-cx_+cza_)*sin(aoa)+(cz_+cxa_)*cos(aoa)/p->m;
            double fnb=((ct+cx_)*sin(aoa)-cz_*cos(aoa))/(p->m*V);
            double fnb_a=((cz_+cxa_)*sin(aoa)+(cx_-cza_)*cos(aoa))/(p->m*V);
            double vd=ft-(p->cdw(Mach)*dps/p->m)+gt;
            double td=fnb*cos(phi)+gn_v;
            double ed=fnb*sin(phi)+gb_v;
            double f=0;
            Eigen::Vector3d g=Eigen::Vector3d::Zero();
            if(velCmdType==0){//dstV
                double fv1=vd-dstVd;
                f+=kVel*fv1*fv1;
                if(grad){
                    g+=2*kVel*fv1*Eigen::Vector3d(ctp*cos(aoa)/p->m,ft_a,0);
                }
            }else if(velCmdType==1){//dstAccel
                double fv1=vd-dstVd;
                f+=kAccel*fv1*fv1;
                if(grad){
                    g+=2*kAccel*fv1*Eigen::Vector3d(ctp*cos(aoa)/p->m,ft_a,0);
                }
            }else{//dstThrust or dstThrottle
                double fv1=x[0]-dstP;
                f+=kPower*fv1*fv1;
                if(grad){
                    g+=Eigen::Vector3d(2*kPower*fv1,0,0);
                }
            }
            if(turnCmdType==0){//dstDir
                double ft1=td-dstTd;
                f+=kTheta*ft1*ft1;
                if(grad){
                    g+=2*kTheta*ft1*Eigen::Vector3d(ctp*sin(aoa)*cos(phi)/(p->m*V),fnb_a*cos(phi),-fnb*sin(phi));
                }
            }else if(turnCmdType==1){//dstOmega
                double ft1=td-dstTd;
                f+=kOmega*ft1*ft1;
                if(grad){
                    g+=2*kOmega*ft1*Eigen::Vector3d(ctp*sin(aoa)*cos(phi)/(p->m*V),fnb_a*cos(phi),-fnb*sin(phi));
                }
            }else{//dstAlpha
                double ft1=x[1]-dstAoa;
                f+=kAoa*ft1*ft1;
                if(grad){
                    g+=Eigen::Vector3d(0,2*kAoa*ft1,0);
                }
            }
            f+=kEps*ed*ed;
            if(grad){
                g+=kEps*Eigen::Vector3d(ctp*sin(aoa)*sin(phi)/(p->m*V),fnb_a*sin(phi),fnb*cos(phi));
                grad[0]=g(0);
                grad[1]=g(1);
                grad[2]=g(2);
            }
            return f;
        };
	std::function<double(unsigned int,const double*,double*,void*)> cnstr=[&](unsigned int n,const double* x,double* grad,void* data){//x=[T,alpha]
        //x[0]:power [0,1)
        //x[1]:aoa [minAoa,maxAoa]
        //x[2]:phi [-pi,pi)
		double ct=(Tmin+x[0]*(Tmax-Tmin));
		double aoa=x[1];
        double phi=x[2];
        double dps=dp*p->S;
		double cx_=p->cx(aoa)*dps;
		double cz_=p->cz(aoa)*dps;
        double ctp=Tmax-Tmin;
		double cxa_=p->cxa(aoa)*dps;
		double cza_=p->cza(aoa)*dps;
        double fnb=((ct+cx_)*sin(aoa)-cz_*cos(aoa))/(p->m*V);
        double fnb_a=((cz_+cxa_)*sin(aoa)+(cx_-cza_)*cos(aoa))/(p->m*V);
        double td=fnb*cos(phi)+gn_v;
        double ed=fnb*sin(phi)+gb_v;
        double f=(td*td+ed*ed)*V*V-pow(util::gravity*p->sideGLimit,2);
        if(grad){
            Eigen::Vector3d g=Eigen::Vector3d(
                ctp*sin(aoa)/(p->m*V)*(td*cos(phi)+ed*sin(phi)),
                fnb_a*(td*cos(phi)+ed*sin(phi)),
                fnb*(-td*sin(phi)+ed*cos(phi))
            )*(2*V*V);
            grad[0]=g(0);
            grad[1]=g(1);
            grad[2]=g(2);
        }
        return f;
    };
    nlopt_delegator obj_d(&obj,nullptr);
    nlopt_delegator cnstr_d(&cnstr,nullptr);
	std::vector<double> x_p={0.5,(p->minAoa+p->maxAoa)/2,0.0};
	std::vector<double> x_n={0.5,(p->minAoa+p->maxAoa)/2,M_PI};
    std::vector<double> lb_p={0.0,p->minAoa,-M_PI_2};
    std::vector<double> ub_p={1.0,p->maxAoa,M_PI_2};
    std::vector<double> lb_n={0.0,p->minAoa,M_PI_2};
    std::vector<double> ub_n={1.0,p->maxAoa,3*M_PI_2};
	nlopt::opt opt=nlopt::opt(nlopt::algorithm::LD_MMA,3);
    opt.set_min_objective(nlopt_delegatee,&obj_d);
    opt.add_inequality_constraint(nlopt_delegatee,&cnstr_d);
    opt.set_xtol_rel(1e-4);
    opt.set_ftol_rel(1e-4);
    opt.set_xtol_abs(1e-4);
    opt.set_ftol_abs(1e-4);
    opt.set_maxeval(200);
    double minf_p,minf_n;
    try{
        opt.set_lower_bounds(lb_p);
        opt.set_upper_bounds(ub_p);
        auto result=opt.optimize(x_p,minf_p);
    }catch(...){
        if(std::isnan(x_p[0])){x_p[0]=1.0;}
        if(std::isnan(x_p[1])){x_p[1]=p->trimAoa(-p->m*gravity);}
        if(std::isnan(x_p[2])){x_p[2]=atan2(-gn_v,-gb_v);}
        minf_p=std::numeric_limits<double>::infinity();
    }
    try{
        opt.set_lower_bounds(lb_n);
        opt.set_upper_bounds(ub_n);
        auto result=opt.optimize(x_n,minf_n);
    }catch(...){
        if(std::isnan(x_n[0])){x_n[0]=1.0;}
        if(std::isnan(x_n[1])){x_n[1]=p->trimAoa(-p->m*gravity);}
        if(std::isnan(x_n[2])){x_n[2]=atan2(-gn_v,-gb_v);}
        minf_n=std::numeric_limits<double>::infinity();
    }
    double thrust,aoa,phi;
	Eigen::Vector3d eyPrev=p->relBtoI(Eigen::Vector3d(0.,1.,0.));
    if(abs(x_n[2]-x_p[2]-M_PI)<0.05 && abs(x_n[1]-x_p[1])<0.05){
        Eigen::Vector3d ey_p=(-n*sin(x_p[2])+b*cos(x_p[2])).normalized();
        Eigen::Vector3d ey_n=(-n*sin(x_n[2])+b*cos(x_n[2])).normalized();
        if(eyPrev.dot(ey_p)>=eyPrev.dot(ey_n)){
            thrust=Tmin+x_p[0]*(Tmax-Tmin);
            aoa=x_p[1];
            phi=x_p[2];
        }else{
            thrust=Tmin+x_n[0]*(Tmax-Tmin);
            aoa=x_n[1];
            phi=x_n[2];
        }
    }else{
        if(minf_p<minf_n){
            thrust=Tmin+x_p[0]*(Tmax-Tmin);
            aoa=x_p[1];
            phi=x_p[2];
        }else{
            thrust=Tmin+x_n[0]*(Tmax-Tmin);
            aoa=x_n[1];
            phi=x_n[2];
        }
    }
    Eigen::Vector3d ey=(-n*sin(phi)+b*cos(phi)).normalized();
    double pCmd=eng->trimPower(thrust,alt,Mach);
	double csDRoll=eyPrev.dot(ey);
	double snDRoll=d.dot(eyPrev.cross(ey));
	double dRoll=atan2(snDRoll,csDRoll);
    return {{"pCmd",pCmd},{"aoa",aoa},{"dRoll",dRoll}};
}

void exportCoordinatedFighter(py::module& m)
{
    using namespace pybind11::literals;
    EXPOSE_CLASS(SimpleFighterJetEngine)
    DEF_READWRITE(SimpleFighterJetEngine,enableInstantResponse)
    DEF_READWRITE(SimpleFighterJetEngine,considerFuel)
    DEF_READWRITE(SimpleFighterJetEngine,tHalf)
    DEF_READWRITE(SimpleFighterJetEngine,tminTable)
    DEF_READWRITE(SimpleFighterJetEngine,tmilTable)
    DEF_READWRITE(SimpleFighterJetEngine,tmaxTable)
    DEF_READWRITE(SimpleFighterJetEngine,fuelFlowRateMin)
    DEF_READWRITE(SimpleFighterJetEngine,fuelFlowRateMil)
    DEF_READWRITE(SimpleFighterJetEngine,fuelFlowRateMax)
    DEF_READWRITE(SimpleFighterJetEngine,alts)
    DEF_READWRITE(SimpleFighterJetEngine,machs)
    DEF_READWRITE(SimpleFighterJetEngine,machsEx)
    DEF_READWRITE(SimpleFighterJetEngine,points)
    DEF_READWRITE(SimpleFighterJetEngine,pointsEx)
    DEF_READWRITE(SimpleFighterJetEngine,thrust)
    DEF_READWRITE(SimpleFighterJetEngine,power)
    DEF_READWRITE(SimpleFighterJetEngine,pCmd)
    DEF_FUNC(SimpleFighterJetEngine,getFuelFlowRate)
    DEF_FUNC(SimpleFighterJetEngine,getThrust)
    .def("calcFuelFlowRate",[](SimpleFighterJetEngine& v,const py::object &args){
        return v.calcFuelFlowRate(args);
    })
    .def("calcThrust",[](SimpleFighterJetEngine& v,const py::object &args){
        return v.calcThrust(args);
    })
    DEF_FUNC(SimpleFighterJetEngine,setPowerCommand)
    .def("calcThrust",py::overload_cast<double,double,double>(&SimpleFighterJetEngine::calcThrust))
    DEF_FUNC(SimpleFighterJetEngine,trimPower)
    DEF_FUNC(SimpleFighterJetEngine,calcPCmdFromThrottle)
    DEF_FUNC(SimpleFighterJetEngine,calcdTdP)
    DEF_FUNC(SimpleFighterJetEngine,calcdTdH)
    DEF_FUNC(SimpleFighterJetEngine,calcdTdM)
    DEF_FUNC(SimpleFighterJetEngine,rtau)
    DEF_FUNC(SimpleFighterJetEngine,pdot)
    DEF_FUNC(SimpleFighterJetEngine,behave)
    ;

    auto clsObj=EXPOSE_CLASS(CoordinatedFighter)
    DEF_FUNC(CoordinatedFighter,calcOptimumCruiseFuelFlowRatePerDistance)
    DEF_FUNC(CoordinatedFighter,trimAoa)
    DEF_FUNC(CoordinatedFighter,cm)
    DEF_FUNC(CoordinatedFighter,cmtrim)
    DEF_FUNC(CoordinatedFighter,cx)
    DEF_FUNC(CoordinatedFighter,cz)
    DEF_FUNC(CoordinatedFighter,cxa)
    DEF_FUNC(CoordinatedFighter,cza)
    DEF_FUNC(CoordinatedFighter,cl)
    DEF_FUNC(CoordinatedFighter,cd)
    DEF_FUNC(CoordinatedFighter,cdw)
    DEF_FUNC(CoordinatedFighter,calcAero)
    DEF_FUNC(CoordinatedFighter,calcMotion)
    DEF_READWRITE(CoordinatedFighter,S)
    DEF_READWRITE(CoordinatedFighter,rollMax)
    DEF_READWRITE(CoordinatedFighter,sideGLimit)
    DEF_READWRITE(CoordinatedFighter,cxTable)
    DEF_READWRITE(CoordinatedFighter,cmTable)
    DEF_READWRITE(CoordinatedFighter,czTable)
    DEF_READWRITE(CoordinatedFighter,trimmedDe)
    DEF_READWRITE(CoordinatedFighter,aoaTable)
    DEF_READWRITE(CoordinatedFighter,deTable)
    DEF_READWRITE(CoordinatedFighter,cdwTable)
    DEF_READWRITE(CoordinatedFighter,minAoa)
    DEF_READWRITE(CoordinatedFighter,maxAoa)
    DEF_READWRITE(CoordinatedFighter,maxCz)
    DEF_READWRITE(CoordinatedFighter,minCz)
    ;
    EXPOSE_INNER_CLASS(clsObj,CoordinatedFighter,FlightController)
    DEF_FUNC(CoordinatedFighter::FlightController,getDefaultCommand)
    DEF_FUNC(CoordinatedFighter::FlightController,calc)
    .def("calc",[](CoordinatedFighter::FlightController& v,const py::object &cmd){
        return v.calc(cmd);
    })
    DEF_FUNC(CoordinatedFighter::FlightController,calcDirect)
    .def("calcDirect",[](CoordinatedFighter::FlightController& v,const py::object &cmd){
        return v.calcDirect(cmd);
    })
    DEF_FUNC(CoordinatedFighter::FlightController,calcFromDirAndVel)
    .def("calcFromDirAndVel",[](CoordinatedFighter::FlightController& v,const py::object &cmd){
        return v.calcFromDirAndVel(cmd);
    })
    DEF_READWRITE(CoordinatedFighter::FlightController,lambdaVel)
    DEF_READWRITE(CoordinatedFighter::FlightController,lambdaTheta)
    DEF_READWRITE(CoordinatedFighter::FlightController,clampTheta)
    DEF_READWRITE(CoordinatedFighter::FlightController,kVel)
    DEF_READWRITE(CoordinatedFighter::FlightController,kAccel)
    DEF_READWRITE(CoordinatedFighter::FlightController,kPower)
    DEF_READWRITE(CoordinatedFighter::FlightController,kTheta)
    DEF_READWRITE(CoordinatedFighter::FlightController,kOmega)
    DEF_READWRITE(CoordinatedFighter::FlightController,kAoa)
    DEF_READWRITE(CoordinatedFighter::FlightController,kEps)
    DEF_READWRITE(CoordinatedFighter::FlightController,pitchLimit)
    DEF_READWRITE(CoordinatedFighter::FlightController,pitchLimitThreshold)
    DEF_READWRITE(CoordinatedFighter::FlightController,altitudeKeeper)
    ;
}
