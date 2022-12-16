// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <boost/uuid/uuid.hpp>
#include "MathUtility.h"
#include "PhysicalAsset.h"
namespace py=pybind11;
namespace nl=nlohmann;

class PYBIND11_EXPORT Track3D{
    public:
    boost::uuids::uuid truth;
    double time;
    Eigen::Vector3d pos,vel;
    std::vector<Track3D> buffer;
    public:
    //constructors & destructor
    Track3D();
    Track3D(std::weak_ptr<PhysicalAsset> truth_);
    Track3D(std::weak_ptr<PhysicalAsset> truth_,const Eigen::Vector3d &pos_,const Eigen::Vector3d  &vel_,const double& time_);
    Track3D(const boost::uuids::uuid& truth_,const Eigen::Vector3d &pos_,const Eigen::Vector3d  &vel_,const double& time_);
    Track3D(const nl::json& j_);
    Track3D(const Track3D& other);
    virtual ~Track3D();
    //functions
    virtual Eigen::Vector3d posI() const;
    virtual Eigen::Vector3d velI() const;
    virtual bool is_none() const;
    Track3D copy() const;
    virtual bool isSame(const Track3D& other) const;
    virtual bool isSame(const boost::uuids::uuid& other) const;
    virtual bool isSame(const std::weak_ptr<Asset> other) const;
    virtual void clearBuffer();
    virtual void addBuffer(const Track3D& other);
    virtual void merge();
    virtual void update(const Track3D& other);
    virtual void updateByExtrapolation(const double& dt);
    virtual Track3D extrapolate(const double& dt);
    virtual Track3D extrapolateTo(const double& dstTime);
    nl::json to_json() const;//for use from python side
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Track3D,truth,time,pos,vel,buffer)

template<class Base=Track3D>
class Track3DWrap:public Base{
    public:
    using Base::Base;
    virtual Eigen::Vector3d posI() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,posI);
    }
    virtual Eigen::Vector3d velI() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,velI);
    }
    virtual bool is_none() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,is_none);
    }
    virtual bool isSame(const Track3D& other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const boost::uuids::uuid& other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const std::weak_ptr<Asset> other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual void clearBuffer() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,clearBuffer);
    }
    virtual void addBuffer(const Track3D& other) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,addBuffer,other);
    }
    virtual void merge() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,merge);
    }
    virtual void update(const Track3D& other) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,update,other);
    }
    virtual void updateByExtrapolation(const double& dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,updateByExtrapolation,dt);
    }
    virtual Track3D extrapolate(const double& dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Track3D,Base,extrapolate,dt);
    }
    virtual Track3D extrapolateTo(const double& dstTime) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Track3D,Base,extrapolateTo,dstTime);
    }
};
class PYBIND11_EXPORT Track2D{
    public:
    boost::uuids::uuid truth;
    double time;
    Eigen::Vector3d dir,origin,omega;
    std::vector<Track2D> buffer;
    public:
    //constructors & destructor
    Track2D();
    Track2D(std::weak_ptr<PhysicalAsset> truth_,const Eigen::Vector3d  &origin_);
    Track2D(std::weak_ptr<PhysicalAsset> truth_,const Eigen::Vector3d &dir_,const Eigen::Vector3d  &origin_,const Eigen::Vector3d  &omega_,const double& time_);
    Track2D(const boost::uuids::uuid& truth_,const Eigen::Vector3d &dir_,const Eigen::Vector3d  &origin_,const Eigen::Vector3d  &omega_,const double& time_);
    Track2D(const nl::json& j_);
    Track2D(const Track2D& other);
    virtual ~Track2D();
    //functions
    virtual Eigen::Vector3d dirI() const;
    virtual Eigen::Vector3d originI() const;
    virtual Eigen::Vector3d omegaI() const;
    virtual bool is_none() const;
    Track2D copy() const;
    virtual bool isSame(const Track2D& other) const;
    virtual bool isSame(const boost::uuids::uuid& other) const;
    virtual bool isSame(const std::weak_ptr<Asset> other) const;
    virtual void clearBuffer();
    virtual void addBuffer(const Track2D& other);
    virtual void merge();
    virtual void update(const Track2D& other);
    virtual void updateByExtrapolation(const double& dt);
    virtual Track2D extrapolate(const double& dt);
    virtual Track2D extrapolateTo(const double& dstTime);
    nl::json to_json() const;//for use from python side
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Track2D,truth,time,dir,origin,omega,buffer)

template<class Base=Track2D>
class Track2DWrap:public Base{
    public:
    using Base::Base;
    virtual Eigen::Vector3d dirI() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,dirI);
    }
    virtual Eigen::Vector3d originI() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,originI);
    }
    virtual Eigen::Vector3d omegaI() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,omegaI);
    }
    virtual bool is_none() const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,is_none);
    }
    virtual bool isSame(const Track2D& other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const boost::uuids::uuid& other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const std::weak_ptr<Asset> other) const override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual void clearBuffer() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,clearBuffer);
    }
    virtual void addBuffer(const Track2D& other) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,addBuffer,other);
    }
    virtual void merge() override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,merge);
    }
    virtual void update(const Track2D& other) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,update,other);
    }
    virtual void updateByExtrapolation(const double& dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(void,Base,updateByExtrapolation,dt);
    }
    virtual Track2D extrapolate(const double& dt) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Track2D,Base,extrapolate,dt);
    }
    virtual Track2D extrapolateTo(const double& dstTime) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(Track2D,Base,extrapolateTo,dstTime);
    }
};

void exportTrack(py::module& m);