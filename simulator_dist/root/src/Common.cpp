// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Common.h"
#include "PhysicalAsset.h"
#include "CommunicationBuffer.h"
#include "Missile.h"
#include "Track.h"
#include "SimulationManager.h"
void exportCommon(py::module &m){
    BIND_VECTOR(std::string,true);
    BIND_MAP(std::string,std::string,true);
    BIND_MAP(std::string,int,true);
    BIND_MAP(std::string,double,true);
    BIND_MAP(std::string,Eigen::Vector2d,true);
    BIND_MAP(std::string,std::shared_ptr<PhysicalAssetAccessor>,true);
    BIND_MAP(std::string,std::weak_ptr<CommunicationBuffer>,true);
    BIND_MAP(std::string,bool,true);
    BIND_VECTOR(std::weak_ptr<Missile>,true);
    typedef std::pair<Track3D,bool> track3d_bool_pair;
    BIND_VECTOR_NAME(track3d_bool_pair,"std::vector<std::pair<Track3D,bool>>",true);
    BIND_VECTOR(Track3D,true);
    typedef std::pair<Track2D,bool> track2d_bool_pair;
    BIND_VECTOR_NAME(track2d_bool_pair,"std::vector<std::pair<Track2D,bool>>",true);
    BIND_VECTOR(Track2D,true);
    BIND_VECTOR(std::vector<std::string>,true);
    BIND_VECTOR(Eigen::VectorXd,true);
    BIND_VECTOR(Eigen::VectorXi,true);
}