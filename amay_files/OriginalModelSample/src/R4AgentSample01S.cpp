// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R4AgentSample01S.h"
#include <utility>
#include <algorithm>
#include <iomanip>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/FlightControllerUtility.h>
#include <ASRCAISim1/CoordinatedFighter.h>
#include <ASRCAISim1/SixDoFFighter.h>
#include <ASRCAISim1/Missile.h>
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Ruler.h>
using namespace util;
R4AgentSample01S::ActionInfo::ActionInfo(){
	dstDir<<1,0,0;
	dstAlt=10000.0;
	velRecovery=false;
	asThrottle=false;
	keepVel=false;
	dstThrottle=1.0;
	dstV=300.0;
	launchFlag=false;
	target=Track3D();
}

R4AgentSample01S::TeamOrigin::TeamOrigin(){
	isEastSider=true;
	pos<<0,0,0;
}
R4AgentSample01S::TeamOrigin::TeamOrigin(bool isEastSider_,double dLine){
	isEastSider=isEastSider_;
	if(isEastSider){
		pos=Eigen::Vector3d(0,dLine,0);
	}else{
		pos=Eigen::Vector3d(0,-dLine,0);
	}
}
R4AgentSample01S::TeamOrigin::~TeamOrigin(){}
Eigen::Vector3d R4AgentSample01S::TeamOrigin::relBtoP(const Eigen::Vector3d& v) const{//陣営座標系⇛慣性座標系
	if(isEastSider){
		return Eigen::Vector3d(v(1),-v(0),v(2));
	}else{
		return Eigen::Vector3d(-v(1),v(0),v(2));
	}
}
Eigen::Vector3d R4AgentSample01S::TeamOrigin::relPtoB(const Eigen::Vector3d& v) const{//慣性座標系⇛陣営座標系
	if(isEastSider){
		return Eigen::Vector3d(-v(1),v(0),v(2));
	}else{
		return Eigen::Vector3d(v(1),-v(0),v(2));
	}
}

R4AgentSample01S::R4AgentSample01S(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:SingleAssetAgent(modelConfig_,instanceConfig_){
	if(isDummy){return;}
	//modelConfigの読み込み
	//observation spaceの設定
	maxTrackNum=getValueFromJsonKRD<std::map<std::string,int>>(modelConfig,"maxTrackNum",randomGen,{{"Friend",4},{"Enemy",4}});
	maxMissileNum=getValueFromJsonKRD<std::map<std::string,int>>(modelConfig,"maxMissileNum",randomGen,{{"Friend",8},{"Enemy",1}});
	horizontalNormalizer=getValueFromJsonKRD(modelConfig,"horizontalNormalizer",randomGen,100000.0);
	verticalNormalizer=getValueFromJsonKRD(modelConfig,"verticalNormalizer",randomGen,20000.0);
	fgtrVelNormalizer=getValueFromJsonKRD(modelConfig,"fgtrVelNormalizer",randomGen,300.0);
	mslVelNormalizer=getValueFromJsonKRD(modelConfig,"mslVelNormalizer",randomGen,1200.0);
	use_image_observation=getValueFromJsonKRD(modelConfig,"use_image_observation",randomGen,true);
	use_vector_observation=getValueFromJsonKRD(modelConfig,"use_vector_observation",randomGen,true);
	assert(use_image_observation||use_vector_observation);
	//  2次元画像として
	if(use_image_observation){
		image_longitudinal_resolution=getValueFromJsonKRD(modelConfig,"image_longitudinal_resolution",randomGen,32);
		image_lateral_resolution=getValueFromJsonKRD(modelConfig,"image_lateral_resolution",randomGen,32);
		image_front_range=getValueFromJsonKRD(modelConfig,"image_front_range",randomGen,200000.0);
		image_back_range=getValueFromJsonKRD(modelConfig,"image_back_range",randomGen,0.0);
		image_side_range=getValueFromJsonKRD(modelConfig,"image_side_range",randomGen,100000.0);
		image_horizon=getValueFromJsonKRD(modelConfig,"image_horizon",randomGen,120);
		image_interval=getValueFromJsonKRD(modelConfig,"image_interval",randomGen,1);
		image_rotate=getValueFromJsonKRD(modelConfig,"image_rotate",randomGen,false);
		image_relative_position=getValueFromJsonKRD(modelConfig,"image_relative_position",randomGen,false);
	}
	//  実数値ベクトルとして
	if(use_vector_observation){
		include_last_action=getValueFromJsonKRD(modelConfig,"include_last_action",randomGen,true);
		vector_past_points=getValueFromJsonKRD<std::vector<int>>(modelConfig,"vector_past_points",randomGen,std::vector<int>());
		std::sort(vector_past_points.begin(),vector_past_points.end());
	}
	//action spaceの設定
	flatten_action_space=getValueFromJsonKRD(modelConfig,"flatten_action_space",randomGen,false);
	//  左右旋回に関する設定
	dstAz_relative=getValueFromJsonKRD(modelConfig,"dstAz_relative",randomGen,true);
	turnTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"turnTable",randomGen,
		(Eigen::Matrix<double,9,1>()<<-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0).finished());
	turnTable*=deg2rad(1.0);
	std::sort(turnTable.begin(),turnTable.end());
	use_override_evasion=getValueFromJsonKRD(modelConfig,"use_override_evasion",randomGen,true);
	if(use_override_evasion){
		evasion_turnTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"evasion_turnTable",randomGen,
			(Eigen::Matrix<double,9,1>()<<-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0).finished());
		evasion_turnTable*=deg2rad(1.0);
		std::sort(evasion_turnTable.begin(),evasion_turnTable.end());
		assert(turnTable.size()==evasion_turnTable.size());
	}else{
		evasion_turnTable=turnTable;
	}
	//  上昇・下降に関する設定
	use_altitude_command=getValueFromJsonKRD(modelConfig,"use_altitude_command",randomGen,false);
	if(use_altitude_command){
		altTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"altTable",randomGen,
			(Eigen::Matrix<double,9,1>()<<-8000.0,-4000.0,-2000.0,-1000.0,0.0,1000.0,2000.0,4000.0,8000.0).finished());
		std::sort(altTable.begin(),altTable.end());
		refAltInterval=getValueFromJsonKRD(modelConfig,"refAltInterval",randomGen,1000.0);
	}else{
		pitchTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"pitchTable",randomGen,
			(Eigen::Matrix<double,9,1>()<<-45.0,-20.0,-10.0,-5.0,0.0,5.0,10.0,20.0,45.0).finished());
		std::sort(pitchTable.begin(),pitchTable.end());
		pitchTable*=deg2rad(1.0);
	}
	//  加減速に関する設定
	accelTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"accelTable",randomGen,
		(Eigen::Matrix<double,3,1>()<<-20.0,0.0,20.0).finished());
	std::sort(accelTable.begin(),accelTable.end());
	always_maxAB=getValueFromJsonKRD(modelConfig,"always_maxAB",randomGen,false);
	//  射撃に関する設定
	use_Rmax_fire=getValueFromJsonKRD(modelConfig,"use_Rmax_fire",randomGen,true);
	if(use_Rmax_fire){
		shotIntervalTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"shotIntervalTable",randomGen,
			(Eigen::Matrix<double,5,1>()<<5.0,10.0,20.0,40.0,80.0).finished());
		std::sort(shotIntervalTable.begin(),shotIntervalTable.end());
		shotThresholdTable=getValueFromJsonKRD<Eigen::VectorXd>(modelConfig,"shotThresholdTable",randomGen,
			(Eigen::Matrix<double,5,1>()<<0.0,0.25,0.5,0.75,1.0).finished());
		std::sort(shotThresholdTable.begin(),shotThresholdTable.end());
	}
	//行動制限に関する設定
	//  高度制限に関する設定
	altMin=getValueFromJsonKRD(modelConfig,"altMin",randomGen,2000.0);
	altMax=getValueFromJsonKRD(modelConfig,"altMax",randomGen,15000.0);
	if(modelConfig.contains("altitudeKeeper")){
		nl::json sub=modelConfig.at("altitudeKeeper");
		altitudeKeeper=AltitudeKeeper(sub);
	}
	//  場外制限に関する設定
	dOutLimitRatio=getValueFromJsonKRD(modelConfig,"dOutLimitRatio",randomGen,0.9);
	dOutLimitScale=getValueFromJsonKRD(modelConfig,"dOutLimitScale",randomGen,50.0);
	//  同時射撃数の制限に関する設定
	maxSimulShot=getValueFromJsonKRD(modelConfig,"maxSimulShot",randomGen,4);
	//  下限速度の制限に関する設定
	minimumV=getValueFromJsonKRD(modelConfig,"minimumV",randomGen,150.0);
	minimumRecoveryV=getValueFromJsonKRD(modelConfig,"minimumRecoveryV",randomGen,180.0);
	minimumRecoveryDstV=getValueFromJsonKRD(modelConfig,"minimumRecoveryDstV",randomGen,200.0);

	//内部変数の初期化
	//  observationに関するもの
	py::module_ spaces=py::module_::import("gym.spaces");
	float floatLow=std::numeric_limits<float>::lowest();
	float floatHigh=std::numeric_limits<float>::max();
	//	2次元画像として
	py::object image_space;
	if(use_image_observation){
		numChannels=8;
		//画像データバッファの生成
		image_buffer=Eigen::Tensor<float,3>(numChannels,image_longitudinal_resolution,image_lateral_resolution);
		image_buffer.setZero();
		image_buffer_coords=Eigen::Tensor<double,3>(2,image_longitudinal_resolution,image_lateral_resolution);
		image_buffer_coords.setZero();
		for(int lon=0;lon<image_longitudinal_resolution;++lon){
			for(int lat=0;lat<image_lateral_resolution;++lat){
				image_buffer_coords(0,lon,lat)=(image_front_range+image_back_range)*(0.5+lat)/image_longitudinal_resolution-image_back_range;
				image_buffer_coords(1,lon,lat)=2.0*image_side_range*(0.5+lon)/image_lateral_resolution-image_side_range;
			}
		}
		//軌跡描画用の過去データバッファの生成
		for(int i=0;i<image_horizon;++i){
			image_past_data.push_back(InstantInfo());
		}
		//spaceの生成
		Eigen::VectorXi shape(3);
		shape<<numChannels,image_longitudinal_resolution,image_lateral_resolution;
		image_space=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
	}
	//	実数値ベクトルとして
	py::object vector_space;
	last_action_dim=3+(1+maxTrackNum["Enemy"]);
	if(use_vector_observation){
		//次元の計算
		friend_dim=3+4+2+3+1+3*maxMissileNum["Enemy"];
		enemy_dim=3+4+4+3;
		friend_msl_dim=3+4+1+3+1;
		//データバッファの生成
		friend_temporal=Eigen::MatrixXf::Zero(maxTrackNum["Friend"],friend_dim);
		enemy_temporal=Eigen::MatrixXf::Zero(maxTrackNum["Enemy"],enemy_dim);
		friend_msl_temporal=Eigen::MatrixXf::Zero(maxMissileNum["Friend"],friend_msl_dim);
		last_action_obs=Eigen::VectorXf::Zero(last_action_dim);
		vector_single_dim=(include_last_action ? last_action_dim : 0)+
			maxTrackNum["Friend"]*friend_dim+
			maxTrackNum["Enemy"]*enemy_dim+
			maxMissileNum["Friend"]*friend_msl_dim;
		//過去データバッファ及びspaceの生成
		if(vector_past_points.size()>0){
			for(int i=0;i<vector_past_points[vector_past_points.size()-1];++i){
				vector_past_data.push_back(Eigen::VectorXf::Zero(vector_single_dim));
			}
			Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,vector_single_dim*(1+vector_past_points.size()));
			vector_space=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}else{
			Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,vector_single_dim);
			vector_space=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}
	//spaceの確定
	if(use_image_observation && use_vector_observation){
		py::dict dict_space;
		dict_space["image"]=image_space;
		dict_space["vector"]=vector_space;
		_observation_space=spaces.attr("Dict")(dict_space);
	}else if(use_image_observation){
		_observation_space=image_space;
	}else if(use_vector_observation){
		_observation_space=vector_space;
	}else{
		throw std::runtime_error("Either use_image_observation or use_vector_observation must be true.");
	}
	//  actionに関するもの
	std::vector<int> nvec;
	nvec.push_back(turnTable.size());
	nvec.push_back(use_altitude_command ? altTable.size() : pitchTable.size());
	if(!always_maxAB){
		nvec.push_back(accelTable.size());
	};
	nvec.push_back(maxTrackNum["Enemy"]+1);//射撃対象
	if(use_Rmax_fire){
		if(shotIntervalTable.size()>1){
			nvec.push_back(shotIntervalTable.size());
		}
		if(shotThresholdTable.size()>1){
			nvec.push_back(shotThresholdTable.size());
		}
	}
	totalActionDim=1;
	actionDims=Eigen::VectorXi::Zero(nvec.size());
	for(int i=0;i<nvec.size();++i){
		totalActionDim*=nvec[i];
		actionDims(i)=nvec[i];
	}
	if(flatten_action_space){
		_action_space=spaces.attr("Discrete")(totalActionDim);
	}else{
		_action_space=spaces.attr("MultiDiscrete")(nvec);
	}
}
R4AgentSample01S::~R4AgentSample01S(){}

void R4AgentSample01S::validate(){
	//Rulerに関する情報の取得
	auto rulerObs=manager->getRuler().lock()->observables;
	dOut=rulerObs.at("dOut");
	dLine=rulerObs.at("dLine");
	std::string eastSider=rulerObs.at("eastSider");
	teamOrigin=TeamOrigin(getTeam()==eastSider,dLine);
	//機体制御方法の設定
	if(parent->isinstance<CoordinatedFighter>()||parent->isinstance<SixDoFFighter>()){
		std::dynamic_pointer_cast<FighterAccessor>(parent)->setFlightControllerMode("fromDirAndVel");
	}else{
		throw std::runtime_error("R4AgentSample01S accepts only SixDoFFighter or CoordinatedFighter for its parent.");
	}
	actionInfo.dstDir<<0,(getTeam()==eastSider ? -1 : +1),0;
	last_action_obs(0)=atan2(actionInfo.dstDir(1),actionInfo.dstDir(0));
	actionInfo.lastShotTimes.clear();
}
py::object R4AgentSample01S::observation_space(){
	return _observation_space;
}
py::object R4AgentSample01S::makeObs(){
	//observablesの収集
	extractObservables();

	//observationの生成
	Eigen::VectorXf vector_obs;
	if(use_image_observation){
		makeImageObservation();
	}
	if(use_vector_observation){
		if(vector_past_points.size()>0){
			//過去のobsをまとめる
			vector_obs=Eigen::VectorXf::Zero(vector_single_dim*(1+vector_past_points.size()));
			//現在のobs
			Eigen::VectorXf current=makeVectorObservation();
			//過去のobsの読み込み(リングバッファ)
			int stepCount=round(manager->getTickCount()/manager->getAgentInterval());
			int bufferSize=vector_past_data.size();
			int bufferIdx=bufferSize-1-(stepCount%bufferSize);
			int ofs=0;
			vector_obs.block(ofs+0,0,vector_single_dim,1)=current;
			ofs+=vector_single_dim;
			for(int frame=0;frame<vector_past_points.size();++frame){
				vector_obs.block(ofs+0,0,vector_single_dim,1)=vector_past_data[(bufferIdx+vector_past_points[frame])%bufferSize];
				ofs+=vector_single_dim;
			}
			//最も古いものを現在のobsに置換
			vector_past_data[bufferIdx]=current;
		}else{
			//現在のobsのみ
			vector_obs=makeVectorObservation();
		}
	}
	if(use_image_observation && use_vector_observation){
		py::dict ret;
		ret["image"]=image_buffer;
		ret["vector"]=vector_obs;
		return ret;
	}else if(use_image_observation){
		return py::cast(image_buffer);
	}else if(use_vector_observation){
		return py::cast(vector_obs);
	}else{
		throw std::runtime_error("Either use_image_observation or use_vector_observation must be true.");
	}
}

void R4AgentSample01S::extractObservables(){
	//味方機(自機含む)
	MotionState myMotion(parent->observables.at("motion"));
	ourMotion.clear();
	ourObservables.clear();
	ourMotion.push_back(myMotion);
	ourObservables.push_back(parent->observables);
	for(auto&& f:parent->observables.at("/shared/fighter"_json_pointer).items()){
		std::string n=f.key();
		if(n==parent->getFullName()){
			continue;
		}
		if(f.value().at("isAlive")){
			ourMotion.push_back(MotionState(f.value().at("motion")));
		}else{
			ourMotion.push_back(MotionState());
		}
		ourObservables.push_back(f.value());
	}
	//彼機(味方の誰かが探知しているもののみ)
	//観測されている航跡を、自機に近いものから順にソートしてlastTrackInfoに格納する。
	//lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
	lastTrackInfo.clear();
	for(auto&& t:ourObservables[0].at("/sensor/track"_json_pointer)){
		lastTrackInfo.push_back(t);
	}
	std::sort(lastTrackInfo.begin(),lastTrackInfo.end(),
	[myMotion](Track3D& lhs,Track3D& rhs)->bool{
		return (lhs.posI()-myMotion.pos).norm()<(rhs.posI()-myMotion.pos).norm();
	});
	//味方誘導弾(射撃時刻が古いものから最大N発分)
	//味方の誘導弾を射撃時刻の古い順にソート
	msls.clear();
	for(int fIdx=0;fIdx<ourObservables.size();++fIdx){
		for(auto&& msl:ourObservables[fIdx].at("/weapon/missiles"_json_pointer)){
			msls.push_back(msl);
		}
	}
	std::sort(msls.begin(),msls.end(),
	[](const nl::json& lhs,const nl::json& rhs){
		double lhsT,rhsT;
		if(lhs.at("isAlive").get<bool>() && lhs.at("hasLaunched").get<bool>()){
			lhsT=lhs.at("launchedT").get<double>();
		}else{
			lhsT=std::numeric_limits<double>::infinity();
		}
		if(rhs.at("isAlive").get<bool>() && rhs.at("hasLaunched").get<bool>()){
			rhsT=rhs.at("launchedT").get<double>();
		}else{
			rhsT=std::numeric_limits<double>::infinity();
		}
		return lhsT<rhsT;
	});
}
double R4AgentSample01S::calcDistanceMargin(const MotionState& myMotion, const nl::json& myObservables){
    //正規化した余剰燃料(距離換算)の計算
	double optCruiseFuelFlowRatePerDistance=myObservables.at("/spec/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer);
	double fuelRemaining=myObservables.at("/propulsion/fuelRemaining"_json_pointer);
	double maxReachableRange=std::numeric_limits<double>::infinity();
    if(optCruiseFuelFlowRatePerDistance>0.0){
        maxReachableRange=fuelRemaining/optCruiseFuelFlowRatePerDistance;
    }
    auto rulerObs=manager->getRuler().lock()->observables;
	Eigen::Vector2d forwardAx=rulerObs.at("forwardAx").at(getTeam());
    double distanceFromLine=forwardAx.dot(myMotion.pos.block<2,1>(0,0,2,1))+dLine;
    double distanceFromBase=rulerObs.at("distanceFromBase").at(getTeam());
    double fuelMargin=rulerObs.at("fuelMargin");
	return std::clamp<double>((maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase))/(2*dLine),-1.0,1.0);
}
void R4AgentSample01S::makeImageObservation(){
	Eigen::Vector3d pos,vel;
	double V,R;
	image_buffer.setZero();
	//現在の情報を収集
	InstantInfo current;
	//味方機(自機含む)の諸元
	for(int fIdx=0;fIdx<ourMotion.size();++fIdx){
		if(ourObservables[fIdx].at("isAlive")){
			current.friend_pos.push_back(ourMotion[fIdx].pos);
		}
	}
	//彼機(味方の誰かが探知しているもののみ)
	for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
		const auto& t=lastTrackInfo[tIdx];
		current.enemy_pos.push_back(t.posI());
	}
	//味方誘導弾(射撃時刻が古いものから最大N発分)
	for(int mIdx=0;mIdx<msls.size();++mIdx){
		const auto& m=msls[mIdx];
		if(!(m.at("isAlive").get<bool>()&&m.at("hasLaunched").get<bool>())){
			break;
		}
		MotionState mm(m.at("motion"));
		current.friend_msl_pos.push_back(mm.pos);
	}
	//画像の生成
	//ch0:自機軌跡
	//ch1:味方軌跡
	//ch2:彼機軌跡
	//ch3:味方誘導弾軌跡
	//ch4:自機覆域
	//ch5:味方覆域
	//ch6:彼我防衛ライン
	//ch7:場外ライン
	int stepCount=round(manager->getTickCount()/manager->getAgentInterval());
	int bufferIdx=image_horizon-1-(stepCount%image_horizon);
	image_past_data[bufferIdx]=current;
	int tMax=std::min<int>(
        floor(stepCount/image_interval)*image_interval,
        floor((image_horizon-1)/image_interval)*image_interval
    );
	for(int t=tMax;t>=0;t-=image_interval){
		InstantInfo frame=image_past_data[(bufferIdx+t)%image_horizon];
		int ch=0;
		//ch0:自機
		plotPoint(
			frame.friend_pos[0],
			ch,
			1.0-1.0*t/image_horizon
		);
		++ch;
		//ch1:味方
		for(int i=1;i<frame.friend_pos.size();++i){
			plotPoint(
				frame.friend_pos[i],
				ch,
				1.0-1.0*t/image_horizon
			);
		}
		++ch;
		//ch2:彼機
		for(int i=0;i<frame.enemy_pos.size();++i){
			plotPoint(
				frame.enemy_pos[i],
				ch,
				1.0-1.0*t/image_horizon
			);
		}
		++ch;
		//ch3:味方誘導弾
		for(int i=0;i<frame.friend_msl_pos.size();++i){
			plotPoint(
				frame.friend_msl_pos[i],
				ch,
				1.0-1.0*t/image_horizon
			);
		}
	}
	
	//ch4:自機レーダ覆域, ch5:味方レーダ覆域
	for(int i=0;i<ourObservables.size();++i){
		if(ourObservables[i].at("isAlive")){
			Eigen::Vector3d ex=ourMotion[i].relBtoP(Eigen::Vector3d(1,0,0));
			double Lref=ourObservables[i].at("/spec/sensor/radar/Lref"_json_pointer);
			double thetaFOR=ourObservables[i].at("/spec/sensor/radar/thetaFOR"_json_pointer);
			for(int lon=0;lon<image_longitudinal_resolution;++lon){
				for(int lat=0;lat<image_lateral_resolution;++lat){
					Eigen::Vector3d pos=gridToPos(lon,lat,10000.0);
					Eigen::Vector3d rPos=pos-ourMotion[i].pos;
					double L=rPos.norm();
					if(L<=Lref && rPos.dot(ex)>=L*cos(thetaFOR)){
						plotPoint(
							pos,
							i==0 ? 4 : 5,
							1.0
						);
					}
				}
			}
		}
	}
	
	//ch6:彼我防衛ライン
	//彼側防衛ライン
	double eps=1e-10;//戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
	plotLine(
		teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,-dOut+eps,0)),
		teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,dOut-eps,0)),
		6,
		1.0
	);
	//我側防衛ライン
	plotLine(
		teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,-dOut+eps,0)),
		teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,dOut-eps,0)),
		6,
		-1.0
	);
	//ch7:場外ライン
	//進行方向右側
	plotLine(
		teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,dOut-eps,0)),
		teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,dOut-eps,0)),
		7,
		1.0
	);
	//進行方向左側
	plotLine(
		teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,-dOut+eps,0)),
		teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,-dOut+eps,0)),
		7,
		1.0
	);
}
Eigen::VectorXf R4AgentSample01S::makeVectorObservation(){
	MotionState myMotion(parent->observables.at("motion"));
	Eigen::Vector3d pos,vel;
	double V,R;
	//前回の行動は前回のdeployで計算済だが、dstAzのみ現在のazからの差に置き換える
	double deltaAz=last_action_obs(0)-myMotion.az;
	last_action_obs(0)=atan2(sin(deltaAz),cos(deltaAz));

	//味方機(自機含む)の諸元
	Eigen::MatrixXf friend_temporal=Eigen::MatrixXf::Zero(maxTrackNum["Friend"],friend_dim);
	for(int fIdx=0;fIdx<ourMotion.size();++fIdx){
		if(fIdx>=maxTrackNum["Friend"]){
			break;
		}
		if(ourObservables[fIdx].at("isAlive")){
			MotionState fMotion=ourMotion[fIdx];
			pos=teamOrigin.relPtoB(fMotion.pos);//慣性座標系→陣営座標系に変換
			vel=teamOrigin.relPtoB(fMotion.vel);//慣性座標系→陣営座標系に変換
			V=vel.norm();
			//位置
			Eigen::Vector3f p=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
			friend_temporal.block(fIdx,0,1,3)=p.transpose();
			//速度
			friend_temporal(fIdx,3)=V/fgtrVelNormalizer;
			friend_temporal.block(fIdx,4,1,3)=(vel.transpose()/V).cast<float>();
			//初期弾数
			friend_temporal(fIdx,7)=ourObservables[fIdx].at("/spec/weapon/numMsls"_json_pointer);
			//残弾数
			friend_temporal(fIdx,8)=ourObservables[fIdx].at("/weapon/remMsls"_json_pointer);
			//姿勢(バンク、α、β)
			Eigen::Vector3d vb=fMotion.relPtoB(fMotion.vel);
			Eigen::Vector3d ex=fMotion.relBtoP(Eigen::Vector3d(1,0,0));
			Eigen::Vector3d ey=fMotion.relBtoP(Eigen::Vector3d(0,1,0));
			Eigen::Vector3d horizontalY=Eigen::Vector3d(0,0,1).cross(ex);
			double roll=0;
			if(horizontalY.norm()>0){
				horizontalY.normalize();
				double sinRoll=horizontalY.cross(ey).dot(ex);
				double cosRoll=horizontalY.dot(ey);
				roll=atan2(sinRoll,cosRoll);
			}
			double alpha=atan2(vb(2),vb(0));
			double beta=asin(vb(1)/V);
			friend_temporal(fIdx,9)=roll;
			friend_temporal(fIdx,10)=alpha;
			friend_temporal(fIdx,11)=beta;
			//余剰燃料
			friend_temporal(fIdx,12)=calcDistanceMargin(fMotion,ourObservables[fIdx]);
			//MWS検出情報
			std::vector<Track2D> mws;
			for(auto&& m:ourObservables[fIdx].at("/sensor/mws/track"_json_pointer)){
				mws.push_back(m);
			}
			std::sort(mws.begin(),mws.end(),
				[fMotion](const Track2D& lhs,const Track2D& rhs)->bool{
					return -lhs.dirI().dot(fMotion.relBtoP(Eigen::Vector3d(1,0,0)))
						<-rhs.dirI().dot(fMotion.relBtoP(Eigen::Vector3d(1,0,0)));
			});
			for(int mIdx=0;mIdx<mws.size();++mIdx){
				if(mIdx>=maxMissileNum["Enemy"]){
					break;
				}
				const auto& m=mws[mIdx];
				Eigen::Vector3d dir=teamOrigin.relPtoB(m.dirI());//慣性座標系→陣営座標系に変換
				friend_temporal.block(fIdx,13+mIdx*3,1,3)=dir.transpose().cast<float>();
			}
		}
	}
	//彼機(味方の誰かが探知しているもののみ)
	enemy_temporal=Eigen::MatrixXf::Zero(maxTrackNum["Enemy"],enemy_dim);
	for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
		const auto& t=lastTrackInfo[tIdx];
		if(tIdx>=maxTrackNum["Enemy"]){
			break;
		}
		pos=teamOrigin.relPtoB(t.posI());//慣性座標系→陣営座標系に変換
		vel=teamOrigin.relPtoB(t.velI());//慣性座標系→陣営座標系に変換
		V=vel.norm();
		//位置
		Eigen::Vector3f p=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		enemy_temporal.block(tIdx,0,1,3)=p.transpose();
		//速度
		enemy_temporal(tIdx,3)=V/fgtrVelNormalizer;
		enemy_temporal.block(tIdx,4,1,3)=(vel.transpose()/V).cast<float>();
		//この航跡に対し飛翔中で最も近い味方誘導弾の情報(距離、誘導状態)
		//味方誘導弾の諸元格納時に格納
		enemy_temporal(tIdx,7)=0;//距離
		enemy_temporal.block(tIdx,8,1,3)<<0,0,0;//誘導状態
		//自機からこの航跡への射程情報
		enemy_temporal(tIdx,11)=calcRHead(myMotion,t)/horizontalNormalizer;//RHead
		enemy_temporal(tIdx,12)=calcRTail(myMotion,t)/horizontalNormalizer;//RTail
		enemy_temporal(tIdx,13)=calcRNorm(myMotion,t);//現在の距離をRTail〜RHeadで正規化したもの
	}

	//味方誘導弾(射撃時刻が古いものから最大N発分)
	Eigen::MatrixXf friend_msl_temporal=Eigen::MatrixXf::Zero(maxMissileNum["Friend"],friend_msl_dim);
	for(int mIdx=0;mIdx<msls.size();++mIdx){
		const auto& m=msls[mIdx];
		if(mIdx>=maxMissileNum["Friend"] || !(m.at("isAlive").get<bool>()&&m.at("hasLaunched").get<bool>())){
			break;
		}
		MotionState mm(m.at("motion"));
		pos=teamOrigin.relPtoB(mm.pos);//慣性座標系→陣営座標系に変換
		vel=teamOrigin.relPtoB(mm.vel);//慣性座標系→陣営座標系に変換
		V=vel.norm();
		//位置
		Eigen::Vector3f p=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		friend_msl_temporal.block(mIdx,0,1,3)=p.transpose();
		//速度
		friend_msl_temporal(mIdx,3)=V/mslVelNormalizer;
		friend_msl_temporal.block(mIdx,4,1,3)=(vel.transpose()/V).cast<float>();
		//目標情報(距離、誘導状態、対応する彼機航跡ID)
		Track3D mTgt=Track3D(m.at("target")).extrapolateTo(manager->getTime());
		friend_msl_temporal(mIdx,7)=1-std::min<double>(1.0,(mTgt.posI()-mm.pos).norm()/horizontalNormalizer);
		Missile::Mode mode=jsonToEnum<Missile::Mode>(m.at("mode"));
		if(mode==Missile::Mode::GUIDED){
			friend_msl_temporal.block(mIdx,8,1,3)<<1,0,0;
		}else if(mode==Missile::Mode::SELF){
			friend_msl_temporal.block(mIdx,8,1,3)<<0,1,0;
		}else{
			friend_msl_temporal.block(mIdx,8,1,3)<<0,0,1;
		}
		friend_msl_temporal(mIdx,11)=-1;
		for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
			const auto& t=lastTrackInfo[tIdx];
			if(tIdx>=maxTrackNum["Enemy"]){
				break;
			}
			if(t.isSame(mTgt)){
				double R=(t.posI()-mm.pos).norm();
				if(R<horizontalNormalizer && enemy_temporal(tIdx,7)<1-R/horizontalNormalizer){
					enemy_temporal(tIdx,7)=1-R/horizontalNormalizer;
					if(mode==Missile::Mode::GUIDED){
						enemy_temporal.block(tIdx,8,1,3)<<1,0,0;
					}else if(mode==Missile::Mode::SELF){
						enemy_temporal.block(tIdx,8,1,3)<<0,1,0;
					}else{
						enemy_temporal.block(tIdx,8,1,3)<<0,0,1;
					}
				}
				friend_msl_temporal(mIdx,11)=tIdx;
			}
		}
	}

	//observationの生成
	Eigen::VectorXf ret(
		(include_last_action ? last_action_dim : 0)+
		maxTrackNum["Friend"]*friend_dim+
		maxTrackNum["Enemy"]*enemy_dim+
		maxMissileNum["Friend"]*friend_msl_dim
	);
	int ofs=0;
	int d;
	if(include_last_action){
		d=last_action_dim;
		ret.block(ofs,0,d,1)=Eigen::Map<Eigen::VectorXf>(last_action_obs.data(),d);
		ofs+=d;
	}
	d=maxTrackNum["Friend"]*friend_dim;
	ret.block(ofs,0,d,1)=Eigen::Map<Eigen::VectorXf>(friend_temporal.data(),d);
	ofs+=d;
	d=maxTrackNum["Enemy"]*enemy_dim;
	ret.block(ofs,0,d,1)=Eigen::Map<Eigen::VectorXf>(enemy_temporal.data(),d);
	ofs+=d;
	d=maxMissileNum["Friend"]*friend_msl_dim;
	ret.block(ofs,0,d,1)=Eigen::Map<Eigen::VectorXf>(friend_msl_temporal.data(),d);
	return ret;
}

py::object R4AgentSample01S::action_space(){
	return _action_space;
}

void R4AgentSample01S::deploy(py::object action_){
	MotionState myMotion(parent->observables.at("motion"));
	last_action_obs=Eigen::VectorXf::Zero(last_action_dim);
	Eigen::VectorXi unflattened_action;
	if(flatten_action_space){
		int flattened_idx=py::cast<int>(action_);
		unflattened_action=unflatten_action(flattened_idx,actionDims);
	}else{
		unflattened_action=py::cast<Eigen::VectorXi>(action_);
	}
	int action_idx=0;
	//左右旋回
	double deltaAz=turnTable(unflattened_action(action_idx));
	std::vector<Track2D> mws;
	for(auto&& m:parent->observables.at("/sensor/mws/track"_json_pointer)){
		mws.push_back(m);
	}
	std::sort(mws.begin(),mws.end(),
		[myMotion](const Track2D& lhs,const Track2D& rhs)->bool{
			return -lhs.dirI().dot(myMotion.relBtoP(Eigen::Vector3d(1,0,0)))
				<-rhs.dirI().dot(myMotion.relBtoP(Eigen::Vector3d(1,0,0)));
	});
	double dstAz;
	if(mws.size()>0 && use_override_evasion){
		deltaAz=evasion_turnTable(unflattened_action(action_idx));
		Eigen::Vector3d dr=Eigen::Vector3d::Zero();
		for(auto& m:mws){
			dr+=m.dirI();
		}
		dr.normalize();
		double dstAz=atan2(-dr(1),-dr(0))+deltaAz;
		actionInfo.dstDir<<cos(dstAz),sin(dstAz),0;
	}else if(dstAz_relative){
		actionInfo.dstDir=myMotion.relHtoP(Eigen::Vector3d(cos(deltaAz),sin(deltaAz),0));
	}else{
		actionInfo.dstDir=teamOrigin.relBtoP(Eigen::Vector3d(cos(deltaAz),sin(deltaAz),0));
	}
	action_idx++;
	dstAz=atan2(actionInfo.dstDir(1),actionInfo.dstDir(0));
	last_action_obs(0)=dstAz;

	//上昇・下降
	double dstPitch;
	if(use_altitude_command){
		double refAlt=std::round<int>(-myMotion.pos(2)/refAltInterval)*refAltInterval;
		actionInfo.dstAlt=std::clamp<double>(refAlt+altTable(unflattened_action(action_idx)),altMin,altMax);
		dstPitch=0;//dstAltをcommandsに与えればSixDoFFighter::FlightControllerのaltitudeKeeperで別途計算されるので0でよい。
	}else{
		dstPitch=pitchTable(unflattened_action(action_idx));
	}
	action_idx++;
	actionInfo.dstDir<<actionInfo.dstDir(0)*cos(dstPitch),actionInfo.dstDir(1)*cos(dstPitch),-sin(dstPitch);
	last_action_obs(1)=use_altitude_command ? actionInfo.dstAlt : dstPitch;
	
	//加減速
	double V=myMotion.vel.norm();
	if(always_maxAB){
		actionInfo.asThrottle=true;
		actionInfo.keepVel=false;
		actionInfo.dstThrottle=1.0;
		last_action_obs(2)=1.0;
	}else{
		actionInfo.asThrottle=false;
		double accel=accelTable(unflattened_action(action_idx));
		action_idx++;
		actionInfo.dstV=V+accel;
		actionInfo.keepVel = accel==0.0;
		last_action_obs(2)=accel/std::max(accelTable(accelTable.size()-1),accelTable(0));
	}
	//下限速度の制限
	if(V<minimumV){
		actionInfo.velRecovery=true;
	}
	if(V>=minimumRecoveryV){
		actionInfo.velRecovery=false;
	}
	if(actionInfo.velRecovery){
		actionInfo.dstV=minimumRecoveryDstV;
		actionInfo.asThrottle=false;
	}

	//射撃
	//actionのパース
	int shotTarget=unflattened_action(action_idx)-1;
	action_idx++;
	double shotInterval,shotThreshold;
	if(use_Rmax_fire){
		if(shotIntervalTable.size()>1){
			shotInterval=shotIntervalTable(unflattened_action(action_idx));
			action_idx++;
		}else{
			shotInterval=shotIntervalTable(0);
		}
		if(shotThresholdTable.size()>1){
			shotThreshold=shotThresholdTable(unflattened_action(action_idx));
			action_idx++;
		}else{
			shotThreshold=shotThresholdTable(0);
		}
	}
	//射撃可否の判断、射撃コマンドの生成
	int flyingMsls=0;
	for(auto&& msl:parent->observables.at("/weapon/missiles"_json_pointer)){
		if(msl.at("isAlive").get<bool>() && msl.at("hasLaunched").get<bool>()){
			flyingMsls++;
		}
	}
	if(
		parent->observables.at("/weapon/remMsls"_json_pointer)>0 && 
		parent->observables.at("/weapon/launchable"_json_pointer) && 
		shotTarget>=0 && 
		shotTarget<lastTrackInfo.size() && 
		flyingMsls<maxSimulShot
	){
		if(use_Rmax_fire){
			double rMin=std::numeric_limits<double>::infinity();
			Track3D t=lastTrackInfo[shotTarget];
			double r=calcRNorm(myMotion,t);
			if(r<=shotThreshold){
				//射程の条件を満たしている
				if(actionInfo.lastShotTimes.count(t.truth)==0){
					actionInfo.lastShotTimes[t.truth]=0.0;
				}
				if(manager->getTime()-actionInfo.lastShotTimes[t.truth]>=shotInterval){
					//射撃間隔の条件を満たしている
					actionInfo.lastShotTimes[t.truth]=manager->getTime();
				}else{
					//射撃間隔の条件を満たさない
					shotTarget=-1;
				}
			}else{
				//射程の条件を満たさない
				shotTarget=-1;
			}
		}
	}else{
		shotTarget=-1;
	}
	last_action_obs(3+(shotTarget+1))=1;
	if(shotTarget>=0){
		actionInfo.launchFlag=true;
		actionInfo.target=lastTrackInfo[shotTarget];
	}else{
		actionInfo.launchFlag=false;
		actionInfo.target=Track3D();
	}
	
	observables[parent->getFullName()]["decision"]={
		{"Roll",nl::json::array({"Don't care"})},
		{"Fire",nl::json::array({actionInfo.launchFlag,actionInfo.target})}
	};
	if(mws.size()>0 && use_override_evasion){
		observables[parent->getFullName()]["decision"]["Horizontal"]=nl::json::array({"Az_NED",dstAz});
	}else{
		if(dstAz_relative){
			observables[parent->getFullName()]["decision"]["Horizontal"]=nl::json::array({"Az_BODY",deltaAz});
		}else{
			observables[parent->getFullName()]["decision"]["Horizontal"]=nl::json::array({"Az_NED",dstAz});
		}
	}
	if(use_altitude_command){
		observables[parent->getFullName()]["decision"]["Vertical"]=nl::json::array({"Pos",-actionInfo.dstAlt});
	}else{
		observables[parent->getFullName()]["decision"]["Vertical"]=nl::json::array({"El",-dstPitch});
	}
	if(actionInfo.asThrottle){
		observables[parent->getFullName()]["decision"]["Throttle"]=nl::json::array({"Throttle",actionInfo.dstThrottle});
	}else{
		observables[parent->getFullName()]["decision"]["Throttle"]=nl::json::array({"Vel",actionInfo.dstV});
	}
}
void R4AgentSample01S::control(){
	MotionState myMotion(parent->observables.at("motion"));
	Eigen::Vector3d pos=myMotion.pos;
	Eigen::Vector3d vel=myMotion.vel;
	//目標方位の補正
	if(abs(pos(0))>=(1-2*(1-dOutLimitRatio))*dOut){
		//戦域逸脱を避けるための方位補正
		//判定ラインの超過具合に応じて制限角度を変化させる。(無限遠でラインに直交、ライン上でラインと平行)
		double over=dOutLimitScale*(abs(pos(0))/dOut-dOutLimitRatio);
		double n=sqrt(actionInfo.dstDir(0)*actionInfo.dstDir(0)+actionInfo.dstDir(1)*actionInfo.dstDir(1));
		double theta=atan(over);
		double cs=cos(theta);
		double sn=sin(theta);
		if(pos(0)>0){//北側
			if(vel(1)>0){//現在東向き
				//南東側のみ許容
				if(actionInfo.dstDir(1)<0){
					//西を向きたい場合、dstDirは真南を限度とする
					actionInfo.dstDir=Eigen::Vector3d(-n,0,actionInfo.dstDir(2));
				}else{
					//東を向きたい場合、角度を制限する
					if(atan2(-actionInfo.dstDir(0),actionInfo.dstDir(1))<theta){
						actionInfo.dstDir=Eigen::Vector3d(-n*sn,n*cs,actionInfo.dstDir(2));
					}
				}
			}else{//現在西向き
				//南西側のみ許容
				if(actionInfo.dstDir(1)>0){
					//東を向きたい場合、dstDirは真南を限度とする
					actionInfo.dstDir=Eigen::Vector3d(-n,0,actionInfo.dstDir(2));
				}else{
					//西を向きたい場合、角度を制限する
					if(atan2(-actionInfo.dstDir(0),-actionInfo.dstDir(1))<theta){
						actionInfo.dstDir=Eigen::Vector3d(-n*sn,-n*cs,actionInfo.dstDir(2));
					}
				}
			}
		}else{//南側
			if(vel(1)>0){//現在東向き
				//北東側のみ許容
				if(actionInfo.dstDir(1)<0){
					//西を向きたい場合、dstDirは真北を限度とする
					actionInfo.dstDir=Eigen::Vector3d(n,0,actionInfo.dstDir(2));
				}else{
					//東を向きたい場合、角度を制限する
					if(atan2(actionInfo.dstDir(0),actionInfo.dstDir(1))<theta){
						actionInfo.dstDir=Eigen::Vector3d(n*sn,n*cs,actionInfo.dstDir(2));
					}
				}
			}else{//現在西向き
				//北西側のみ許容
				if(actionInfo.dstDir(1)>0){
					//東を向きたい場合、dstDirは真北を限度とする
					actionInfo.dstDir=Eigen::Vector3d(n,0,actionInfo.dstDir(2));
				}else{
					//西を向きたい場合、角度を制限する
					if(atan2(actionInfo.dstDir(0),-actionInfo.dstDir(1))<theta){
						actionInfo.dstDir=Eigen::Vector3d(n*sn,-n*cs,actionInfo.dstDir(2));
					}
				}
			}
		}
	}
	//高度方向の補正(actionがピッチ指定の場合)
	if(!use_altitude_command){
		double n=sqrt(actionInfo.dstDir(0)*actionInfo.dstDir(0)+actionInfo.dstDir(1)*actionInfo.dstDir(1));
		double dstPitch=atan2(-actionInfo.dstDir(2),n);
		//高度下限側
		Eigen::Vector3d bottom=altitudeKeeper(myMotion,actionInfo.dstDir,altMin);
		double minPitch=atan2(-bottom(2),sqrt(bottom(0)*bottom(0)+bottom(1)*bottom(1)));
		//高度上限側
		Eigen::Vector3d top=altitudeKeeper(myMotion,actionInfo.dstDir,altMax);
		double maxPitch=atan2(-top(2),sqrt(top(0)*top(0)+top(1)*top(1)));
		dstPitch=std::clamp<double>(dstPitch,minPitch,maxPitch);
		double cs=cos(dstPitch);
		double sn=sin(dstPitch);
		actionInfo.dstDir=Eigen::Vector3d(actionInfo.dstDir(0)/n*cs,actionInfo.dstDir(1)/n*cs,-sn);
	}
	commands[parent->getFullName()]={
		{"motion",{
			{"dstDir",actionInfo.dstDir}
		}},
		{"weapon",{
			{"launch",actionInfo.launchFlag},
			{"target",actionInfo.target}
		}}
	};
	if(use_altitude_command){
		commands[parent->getFullName()]["motion"]["dstAlt"]=actionInfo.dstAlt;
	}
	if(actionInfo.asThrottle){
		commands[parent->getFullName()]["motion"]["dstThrottle"]=actionInfo.dstThrottle;
	}else if(actionInfo.keepVel){
		commands[parent->getFullName()]["motion"]["dstAccel"]=0.0;
	}else{
		commands[parent->getFullName()]["motion"]["dstV"]=actionInfo.dstV;
	}
	actionInfo.launchFlag=false;
}
py::object R4AgentSample01S::convertActionFromAnother(const nl::json& decision,const nl::json& command){
	double interval=manager->getAgentInterval()*manager->getBaseTimeStep();
	MotionState myMotion(parent->observables.at("motion"));
	Eigen::VectorXi unflattened_action=Eigen::VectorXi::Zero(actionDims.size());
	int action_idx=0;
	Eigen::VectorXd tmp_d;
	int tmp_i;

	//左右旋回
	std::string decisionType=decision[parent->getFullName()]["Horizontal"][0];
	double value=decision[parent->getFullName()]["Horizontal"][1];
	nl::json motionCmd=command[parent->getFullName()].at("motion");
	double dstAz_ex=0.0;
	if(decisionType=="Az_NED"){
		dstAz_ex=value;
	}else if(decisionType=="Az_BODY"){
		dstAz_ex=value+myMotion.az;
	}else if(motionCmd.contains("dstDir")){
		Eigen::Vector3d dd=motionCmd.at("dstDir");
		dstAz_ex=atan2(dd(1),dd(0));
	}else{
		throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
	}
	double deltaAz_ex=0.0;
	std::vector<Track2D> mws;
	for(auto&& m:parent->observables.at("/sensor/mws/track"_json_pointer)){
		mws.push_back(m);
	}
	if(use_override_evasion && mws.size()>0){
		std::sort(mws.begin(),mws.end(),
			[myMotion](const Track2D& lhs,const Track2D& rhs)->bool{
				return -lhs.dirI().dot(myMotion.relBtoP(Eigen::Vector3d(1,0,0)))
					<-rhs.dirI().dot(myMotion.relBtoP(Eigen::Vector3d(1,0,0)));
		});
		Eigen::Vector3d dr=Eigen::Vector3d::Zero();
		for(auto& m:mws){
			dr+=m.dirI();
		}
		dr.normalize();
		deltaAz_ex=dstAz_ex-atan2(-dr(1),-dr(0));
		deltaAz_ex=atan2(sin(deltaAz_ex),cos(deltaAz_ex));
		tmp_d=evasion_turnTable-Eigen::VectorXd::Constant(evasion_turnTable.rows(),deltaAz_ex);
	}else{
		Eigen::Vector3d dstDir_ex;
		if(dstAz_relative){
			dstDir_ex = myMotion.relPtoH(Eigen::Vector3d(cos(dstAz_ex),sin(dstAz_ex),0));
		}else{
			dstDir_ex = teamOrigin.relPtoB(Eigen::Vector3d(cos(dstAz_ex),sin(dstAz_ex),0));
		}
		deltaAz_ex = atan2(dstDir_ex(1),dstDir_ex(0));
		tmp_d=turnTable-Eigen::VectorXd::Constant(turnTable.rows(),deltaAz_ex);
	}
	tmp_d.cwiseAbs().minCoeff(&tmp_i);
	unflattened_action(action_idx)=tmp_i;
	action_idx++;

	//上昇・下降
	decisionType=decision[parent->getFullName()]["Vertical"][0];
	value=decision[parent->getFullName()]["Vertical"][1];
	double dstPitch_ex=0.0;
	double deltaAlt_ex=0.0;
	double refAlt=std::round<int>(-myMotion.pos(2)/refAltInterval)*refAltInterval;
	if(decisionType=="El"){
		dstPitch_ex=-value;
		deltaAlt_ex=altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt;
	}else if(decisionType=="Pos"){
		dstPitch_ex=altitudeKeeper.getDstPitch(myMotion,-value);
		deltaAlt_ex=-value-refAlt;
	}else if(motionCmd.contains("dstDir")){
		Eigen::Vector3d dd=motionCmd.at("dstDir");
		dstPitch_ex=-atan2(dd(2),sqrt(dd(0)*dd(0)+dd(1)*dd(1)));
		deltaAlt_ex=altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt;
	}else{
		throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
	}
	dstPitch_ex=atan2(sin(dstPitch_ex),cos(dstPitch_ex));
	if(use_altitude_command){
		tmp_d=altTable-Eigen::VectorXd::Constant(altTable.rows(),deltaAlt_ex);
	}else{
		tmp_d=pitchTable-Eigen::VectorXd::Constant(pitchTable.rows(),dstPitch_ex);
	}
	tmp_d.cwiseAbs().minCoeff(&tmp_i);
	unflattened_action(action_idx)=tmp_i;
	action_idx++;

	//加減速
	if(!always_maxAB){
		//本サンプルでは「現在の速度と目標速度の差」をactionとしてdstVを機体へのコマンドとしているため、
		//教師役の行動が目標速度以外(スロットルや加速度)の場合、正確な変換は困難である。
		//そのため、最も簡易的な変換の例として、最大減速、等速、最大加速の3値への置換を実装している。
		decisionType=decision[parent->getFullName()]["Throttle"][0];
		value=decision[parent->getFullName()]["Throttle"][1];
		double V=myMotion.vel.norm();
		int accelIdx=-1;
		double dstV_ex;
		if(motionCmd.contains("dstV")){
			dstV_ex=motionCmd.at("dstV");
		}else if(decisionType=="Vel"){
			dstV_ex=value;
		}else if(decisionType=="Throttle"){
			//0〜1のスロットルで指定していた場合
			double th=0.3;
			if(value>1-th){
				accelIdx=accelTable.size()-1;
			}else if(value<th){
				accelIdx=0;
			}else{
				dstV_ex=V;
			}
		}else if(decisionType=="Accel"){
			//加速度ベースの指定だった場合
			double eps=0.5;
			if(abs(value)<eps){
				dstV_ex=V;
			}else if(value>0){
				accelIdx=accelTable.size()-1;
			}else{
				accelIdx=0;
			}
		}else{
			throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
		}
		if(accelIdx<0){
			double deltaV_ex=dstV_ex-V;
			tmp_d=accelTable-Eigen::VectorXd::Constant(accelTable.rows(),deltaV_ex);
			tmp_d.cwiseAbs().minCoeff(&tmp_i);
			unflattened_action(action_idx)=tmp_i;
		}else{
			unflattened_action(action_idx)=accelIdx;
		}
		action_idx++;
	}

	//射撃
	int shotTarget_ex=-1;
	Track3D expertTarget=decision[parent->getFullName()]["Fire"][1];
	if(decision[parent->getFullName()]["Fire"][0].get<bool>()){
		for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
			const auto& t=lastTrackInfo[tIdx];
			if(t.isSame(expertTarget)){
				shotTarget_ex=tIdx;
			}
		}
		if(shotTarget_ex>=maxTrackNum["Enemy"]){
			shotTarget_ex=-1;
		}
	}
	unflattened_action(action_idx)=shotTarget_ex+1;
	action_idx++;
	if(use_Rmax_fire){
		if(shotTarget_ex<0){
			//射撃なしの場合、間隔と射程の条件は最も緩いものとする
			if(shotIntervalTable.size()>1){
				unflattened_action(action_idx)=0;
				action_idx++;
			}
			if(shotThresholdTable.size()>1){
				unflattened_action(action_idx)=shotThresholdTable.size()-1;
				action_idx++;
			}
		}else{
			//射撃ありの場合、間隔と射程の条件は最も厳しいものとする
			if(actionInfo.lastShotTimes.count(expertTarget.truth)==0){
				actionInfo.lastShotTimes[expertTarget.truth]=0.0;
			}
			if(shotIntervalTable.size()>1){
				double shotInterval_ex=manager->getTime()-actionInfo.lastShotTimes[expertTarget.truth];
				unflattened_action(action_idx)=0;
				for(int i=shotIntervalTable.size()-1;i>=0;i--){
					if(shotIntervalTable(i)<shotInterval_ex){
						unflattened_action(action_idx)=i;
						break;
					}
				}
				action_idx++;
			}
			if(shotThresholdTable.size()>1){
				double r=calcRNorm(myMotion,expertTarget);
				unflattened_action(action_idx)=shotThresholdTable.size()-1;
				for(int i=0;i<shotThresholdTable.size();i++){
					if(r<shotThresholdTable(i)){
						unflattened_action(action_idx)=i;
						break;
					}
				}
				action_idx++;
			}
		}
	}
	if(flatten_action_space){
		return py::cast(flatten_action(unflattened_action,actionDims));
	}else{
		return py::cast(unflattened_action);
	}
}
void R4AgentSample01S::controlWithAnotherAgent(const nl::json& decision,const nl::json& command){
	//基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
	control();
	//例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
	//commands[parent->getFullName()]["weapon"]=command.at(parent->getFullName()).at("weapon");
}

int R4AgentSample01S::flatten_action(const Eigen::VectorXi& original_action,const Eigen::VectorXi& original_shape){
    //MultiDiscretionなactionをDiscreteなactionに変換する。
	int ret=0;
	for(int i=0;i<original_shape.size();++i){
		ret*=original_shape(i);
		ret+=original_action(i);
	}
	return ret;
}
Eigen::VectorXi R4AgentSample01S::unflatten_action(const int& flattened_action,const Eigen::VectorXi& original_shape){
    //Discrete化されたactionを元のMultiDiscreteなactionに変換する。
	Eigen::VectorXi ret=Eigen::VectorXi::Zero(original_shape.size());
	int idx=flattened_action;
	for(int i=original_shape.size()-1;i>=0;--i){
		ret(i)=idx%original_shape(i);
		idx-=ret(i);
		idx/=original_shape(i);
	}
	return ret;
}

bool R4AgentSample01S::isInside(const int& lon, const int& lat){
    //ピクセル座標(lon,lat)が画像の範囲内かどうかを返す。
	return 0<=lon && lon<image_longitudinal_resolution && 0<=lat && lat<image_lateral_resolution;
}
Eigen::Vector2i R4AgentSample01S::rposToGrid(const Eigen::Vector3d& dr){
	//基準点からの相対位置drに対応するピクセル座標(lon,lat)を返す。
	int lon=floor((dr(1)+image_side_range)*image_lateral_resolution/(2*image_side_range));
	int lat=floor((dr(0)+image_back_range)*image_longitudinal_resolution/(image_front_range+image_back_range));
	return Eigen::Vector2i(lon,lat);
}
Eigen::Vector3d R4AgentSample01S::gridToRpos(const int& lon, const int& lat, const double& alt){
	//指定した高度においてピクセル座標(lon,lat)に対応する基準点からの相対位置drを返す。
	return Eigen::Vector3d(
		image_buffer_coords(0,lon,lat),
		image_buffer_coords(1,lon,lat),
		-alt
	);
}
Eigen::Vector2i R4AgentSample01S::posToGrid(const Eigen::Vector3d& pos){
	//慣性座標系での絶対位置posに対応するピクセル座標(lon,lat)を返す。
	MotionState myMotion=ourMotion[0];
	Eigen::Vector3d rpos;
	if(image_relative_position){
		//自機位置が基準点
		rpos=pos-myMotion.pos;
	}else{
		//自陣営防衛ライン中央が基準点
		rpos=pos-teamOrigin.pos;
	}
	if(image_rotate){
		//慣性座標系からH座標系に回転
		rpos=myMotion.relPtoH(rpos);
	}else{
		//慣性座標系から陣営座標系に回転
		rpos=teamOrigin.relPtoB(rpos);
	}
	return rposToGrid(rpos);
}
Eigen::Vector3d R4AgentSample01S::gridToPos(const int& lon, const int& lat, const double& alt){
	//指定した高度においてピクセル座標(lon,lat)に対応する慣性座標系での絶対位置posを返す。
	MotionState myMotion=ourMotion[0];
	Eigen::Vector3d rpos=gridToRpos(lon,lat,alt);
	if(image_rotate){
		//H座標系から慣性座標系に回転
		rpos=myMotion.relHtoP(rpos);
	}else{
		//陣営座標系から慣性座標系に回転
		rpos=teamOrigin.relBtoP(rpos);
	}
	if(image_relative_position){
		//自機位置が基準点
		rpos=rpos+myMotion.pos;
	}else{
		//自陣営防衛ライン中央が基準点
		rpos=rpos+teamOrigin.pos;
	}
	return rpos;
}
void R4AgentSample01S::plotPoint(const Eigen::Vector3d& pos,const int& ch, const float& value){
	//画像バッファのチャネルchで慣性座標系での絶対位置posに対応する点の値をvalueにする。
	Eigen::Vector2i g=posToGrid(pos);
	int lon=g(0);
	int lat=g(1);
	if(isInside(lon,lat)){
		image_buffer(ch,lon,lat)=value;
	}
}
void R4AgentSample01S::plotLine(const Eigen::Vector3d& pBegin,const Eigen::Vector3d& pEnd,const int& ch, const float& value){
    //画像バッファのチャネルchで慣性座標系での絶対位置pBeginからpEndまでの線分に対応する各点の値をvalueにする。
	//線分の描画にはブレゼンハムのアルゴリズムを使用している。
	Eigen::Vector2i gBegin=posToGrid(pBegin);
	Eigen::Vector2i gEnd=posToGrid(pEnd);
	int x0=gBegin(0);
	int y0=gBegin(1);
	int x1=gEnd(0);
	int y1=gEnd(1);
	bool steep=abs(y1-y0)>abs(x1-x0);
	if(steep){
		std::swap(x0,y0);
		std::swap(x1,y1);
	}
	if(x0>x1){
		std::swap(x0,x1);
		std::swap(y0,y1);
	}
	int deltax=x1-x0;
	int deltay=abs(y1-y0);
	int error=deltax/2;
	int ystep = y0<y1 ? 1 : -1;
	int y=y0;
	for(int x=x0;x<=x1;++x){
		if(steep){
			if(isInside(y,x)){
				image_buffer(ch,y,x)=value;
			}
		}else{
			if(isInside(x,y)){
				image_buffer(ch,x,y)=value;
			}
		}
		error-=deltay;
		if(error<0){
			y+=ystep;
			error+=deltax;
		}
	}
}

double R4AgentSample01S::calcRHead(const MotionState& myMotion,const Track3D& track){
	std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	Eigen::Vector3d rt=track.posI();
	Eigen::Vector3d vt=track.velI();
	Eigen::Vector3d rs=myMotion.pos;
	Eigen::Vector3d vs=myMotion.vel;
	return f->getRmax(rs,vs,rt,vt,0.0);
}
double R4AgentSample01S::calcRTail(const MotionState& myMotion,const Track3D& track){
	std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	Eigen::Vector3d rt=track.posI();
	Eigen::Vector3d vt=track.velI();
	Eigen::Vector3d rs=myMotion.pos;
	Eigen::Vector3d vs=myMotion.vel;
	return f->getRmax(rs,vs,rt,vt,M_PI);
}
double R4AgentSample01S::calcRNorm(const MotionState& myMotion,const Track3D& track){
	double RHead=calcRHead(myMotion,track);
	double RTail=calcRTail(myMotion,track);
	Eigen::Vector3d rs=myMotion.pos;
	Eigen::Vector3d rt=track.posI();
	double r=(rs-rt).norm()-RTail;
	double delta=RHead-RTail;
	double outRangeScale=100000.0;
	if(delta==0){
		if(r<0){
			r=r/outRangeScale;
		}else if(r>0){
			r=1+r/outRangeScale;
		}else{
			r=0;
		}
	}else{
		if(r<0){
			r=r/outRangeScale;
		}else if(r>delta){
			r=1+(r-delta)/outRangeScale;
		}else{
			r/=delta;
		}
	}
	return r;
}

void exportR4AgentSample01S(py::module &m)
{
	using namespace pybind11::literals;
	EXPOSE_LOCAL_CLASS(R4AgentSample01S)
	DEF_FUNC(R4AgentSample01S,validate)
	DEF_FUNC(R4AgentSample01S,observation_space)
	DEF_FUNC(R4AgentSample01S,makeObs)
	DEF_FUNC(R4AgentSample01S,flatten_action)
	DEF_FUNC(R4AgentSample01S,unflatten_action)
	DEF_FUNC(R4AgentSample01S,deploy)
	DEF_FUNC(R4AgentSample01S,control)
	DEF_FUNC(R4AgentSample01S,convertActionFromAnother)
	.def("convertActionFromAnother",[](R4AgentSample01S& v,const py::object& decision,const py::object& command){
		return v.convertActionFromAnother(decision,command);
	})
	;
}

