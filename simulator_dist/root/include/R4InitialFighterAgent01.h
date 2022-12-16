// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Track.h"
#include "Agent.h"
#include "Sensor.h"
#include "FlightControllerUtility.h"

class PYBIND11_EXPORT TrackInfo:public Track3D{
    public:
    enum class SensingState{
        INSIDE,
        LIMIT,
        OUTSIDE
    };
    enum class UpdateState{
        TRACK,
        MEMORY,
        LOST
    };
    int idx;
    double distance,myRHead,myRTail,hisRHead,hisRTail;
    SensingState inOurSensor,inMySensor;
    std::size_t numTracker,numTrackerLimit;
    std::vector<std::string> trackers,limitTrackers,nonLimitTrackers;
    UpdateState state;
    double memoryStartTime;
    public:
    //constructors & destructor
    TrackInfo();
    TrackInfo(const Track3D& original_,int idx_=-1);
    TrackInfo(const TrackInfo& other);
    TrackInfo(const nl::json& j_);
    virtual ~TrackInfo();
    //functions
    TrackInfo copy() const;
    virtual void update(const TrackInfo& other,bool isFull=false);
    nl::json to_json() const;//for use from python side
};
template<class Base=TrackInfo>
class TrackInfoWrap:public Track3DWrap<Base>{
    public:
    using Track3DWrap<Base>::Track3DWrap;
    virtual void update(const TrackInfo& other,bool isFull) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,update,other,isFull);
    }
};
void PYBIND11_EXPORT to_json(nl::json& j,const TrackInfo& t);
void PYBIND11_EXPORT from_json(const nl::json& j,TrackInfo& t);

DECLARE_CLASS_WITHOUT_TRAMPOLINE(R4InitialFighterAgent01,SingleAssetAgent)
    public:
    enum class TransitionCause{
        FUEL_SHORTAGE,//1
        NO_ENEMY,//2
        TRY_BREAK,//3
        INTERCEPT_LEAKER,//3
        TRACKING_BY_ALLY,//4
        TRACKING_BY_NOBODY,//5
        TRACKING_BY_MYSELF,//6
        ABNORMAL,//0
        NONE//0
    };
    enum class State{
        ADVANCE,
        APPROACH_TARGET,
        KEEP_SENSING,
        RTB,
        WITHDRAW,
        EVADE,
        NONE
    };
    //失探後の航跡の保持に関するパラメータ
	double tMaxMemory;
    //探知状況の分類に関するパラメータ
	double sensorInRangeLimit,sensorInCoverageLimit;
	//(c1)射撃条件に関するパラメータ
	double kShoot;
    int nMslSimul;
	//(c2)離脱条件に関するパラメータ
	double kBreak;
    //(c3)離脱終了条件に関するパラメ−タ
    double tWithdraw;
    //(s1)通常時の行動選択に関するパラメータ
    //advance,approach,keepSensing
	double pAdvanceAlly,pApproachAlly,pKeepSensingAlly,pApproachMyself,pKeepSensingMyself;
    //(s1-1)前進に関するパラメータ
    double dPrioritizedAdvance,thetaPrioritizedAdvance;
    //(s1-3)横行に関するパラメータ
    double thetaKeepSensing;
    //(s1-4)後退に関するパラメータ
    double RTBEnterMargin,RTBExitMargin,RTBAltitude,RTBVelocity,RTBPitchLimit;
    //(s3)回避に関するパラメータ
	double thetaEvasion,hEvasion;
    //(o1)副目標の追尾に関するパラメータ
    double thetaModForSensing;
    //(o2)高度維持に関するパラメータ
	double thetaStable,hNormal;
    //(o3)場外の防止に関するパラメータ
	double dOutLimit,dOutLimitTurnAxis,dOutLimitKeepSensing;
	//目標選択に関するパラメータ
	double dPrioritizedAimLeaker,thetaPrioritizedAimLeaker;
    //速度維持のためのパラメータ
    double minimumV,minimumRecoveryV,minimumRecoveryDstV,nominalThrottle;
    //内部変数
    State state;
    std::vector<TrackInfo> track,additionalTargets,allEnemies,withdrawnTrigger;
    TrackInfo target;
    bool isAssignedAsInterceptor;
    bool launchFlag,velRecovery;;
    double launchedTime;
    TrackInfo launchedTarget;
    TrackInfo fireTgt;
    double withdrawnTime,estEnemyNoDetectionTime;
    bool waitingTransition;
    double transitionDelay,transitionTriggeredTime;
    State transitionBefore,transitionAfter;
    TransitionCause transitionCause;
    bool isAfterDeploy;
    Eigen::Vector3d forward,rightSide;
    double dOut,dLine;
    int fgtrID;
    Eigen::Vector3d pos,vel;
    Eigen::Vector3d dstDir,omegaI,omegaB,omegaScale;
    AltitudeKeeper altitudeKeeper;
    public:
    R4InitialFighterAgent01(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~R4InitialFighterAgent01();
    void validate() override;
    virtual void deploy(py::object action) override;
    void perceive(bool inReset) override;
    virtual void control() override;
    //情報の更新
    void updateMyInfo();
    void updateTracks(bool isFull);
    void updateTrackInfo(TrackInfo& ti,bool isFull);
    void updateTargetStatus(bool isFull);
    void updateTargetStatusSub(TrackInfo& tgt,bool isFull);

    bool chkFuelSufficiency();//余剰燃料のチェック
    //(c1)射撃条件
    bool chkLaunchable();
    std::pair<bool,std::pair<TrackInfo,double>> chkConditionForShoot(TrackInfo& tgt);
    //(c2)離脱条件
    std::vector<bool> chkBreak();
    //(s3)回避条件
    bool chkMWS();
    bool chkTransitionCount();
    void reserveTransition(const State& dst,double delay);
    void completeTransition();
    void cancelTransition();
    void immidiateTransition(const State& dst);
    void selectTarget();
    void decideShoot(TrackInfo& tgt);
    bool chooseTransitionNormal();
    bool deploySub();
    void makeCommand();
    void overrideFlightControl();
    protected:
    double getT();
    double calcRHead(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
    double calcRTail(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
};

void exportR4InitialFighterAgent01(py::module& m);
