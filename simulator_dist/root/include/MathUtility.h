// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>

namespace py=pybind11;

template<typename T>
using VecX=Eigen::Matrix<T,-1,1>;
template<typename T>
using Vec3=Eigen::Matrix<T,3,1>;

double PYBIND11_EXPORT sinc(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT sincMinusOne(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT sincMinusOne_x2(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT oneMinusCos(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT oneMinusCos_x2(const double &x,const double &eps=1e-3);
Eigen::Vector3d PYBIND11_EXPORT getOrthogonalVector(const Eigen::Vector3d &v);
template<typename T>
Eigen::Matrix<T,3,3> skewMatrix(const Eigen::Matrix<T,3,1>& v){
    return Eigen::Matrix<T,3,3>({
        {0,-v(2),v(1)},
        {v(2),0,-v(0)},
        {-v(1),v(0),0}});
}

template<typename T>
T polyval(const VecX<T>& p,const T &x){
    Eigen::Index n=p.size();
    T xx=1.0;
    T ret=0;
    for(int i=0;i<n;++i){
        ret+=xx*p(n-1-i);
        xx*=x;
    }
    return ret;
}
template<typename T>
VecX<T> polyderiv(const VecX<T>& p,const T &x){
    Eigen::Index n=p.size()-1;
    VecX<T> ret;
    for(int i=0;i<n;++i){
        ret(i)=p(i)*(n-1-i);
    }
    return ret;
}
template<typename T>
std::pair<Eigen::Index,T> _search1d(const VecX<T> &x,const T &p){
    Eigen::Index lb=0;
    Eigen::Index ub=x.size()-2;
    if(p<x(lb)){
        return std::make_pair(lb,(p-x(lb))/(x(lb+1)-x(lb)));
    }
    if(x(ub)<=p){
        return std::make_pair(ub,(p-x(ub))/(x(ub+1)-x(ub)));
    }
    Eigen::Index i;
    while(ub-lb>1){
        i=(lb+ub)/2;
        if(x(i)<=p){
            lb=i;
        }else{// if(p<x(i)){
            ub=i;
        }
    }
    return std::make_pair(lb,(p-x(lb))/(x(lb+1)-x(lb)));
}
template<typename T,int nDim,int TableOption_,int XOption_>
VecX<T> interpn(const std::vector<VecX<T>>& point,const Eigen::Tensor<T,nDim,TableOption_> &table,const Eigen::Matrix<T,-1,-1,XOption_> &x){
    Eigen::Index nx=x.rows();
    Eigen::Index nt=table.dimension(nDim-1);
    Eigen::Index m=table.size();
    Eigen::Tensor<T,3,TableOption_> store;
    for(int d=nDim-1;d>=0;--d){
        m/=nt;
        Eigen::Tensor<T,2,TableOption_> tmp2(nx,m);
        if(d==nDim-1){
            Eigen::Tensor<T,2,TableOption_> first=table.reshape(Eigen::array<Eigen::Index,2>{{m,nt}});
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,T> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    tmp2(i,j)=idx.second*(first(j,idx.first+1)-first(j,idx.first))+first(j,idx.first);
                }
            }
        }else{
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,T> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    tmp2(i,j)=idx.second*(store(i,j,idx.first+1)-store(i,j,idx.first))+store(i,j,idx.first);
                }
            }
        }
        nt=(d>=1)?table.dimension(d-1):1;
        store=tmp2.reshape(Eigen::array<Eigen::Index,3>{{nx,m/nt,nt}});
    }
    typedef Eigen::Map<VecX<T>> maptype;
    return maptype(store.data(),nx);
}
template<typename T,int nDim,int TableOption_,int XOption_>
Eigen::Matrix<T,-1,-1,TableOption_&Eigen::RowMajor> interpgradn(const std::vector<VecX<T>>& point,const Eigen::Tensor<T,nDim,TableOption_> &table,const Eigen::Matrix<T,-1,-1,XOption_> &x){
    Eigen::Index nx=x.rows();
    Eigen::Index nt=table.dimension(nDim-1);
    Eigen::Index m=table.size();
    Eigen::Tensor<T,4,TableOption_> store;
    for(int d=nDim-1;d>=0;--d){
        m/=nt;
        Eigen::Tensor<T,3,TableOption_> tmp2(nDim,nx,m);
        if(d==nDim-1){
            Eigen::Tensor<T,2,TableOption_> first=table.reshape(Eigen::array<Eigen::Index,2>{{m,nt}});
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,T> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    for(Eigen::Index g=0;g<nDim;++g){
                        if(d==g){
                            tmp2(g,i,j)=(first(j,idx.first+1)-first(j,idx.first))/(point[d](idx.first+1)-point[d](idx.first));
                        }else{
                            tmp2(g,i,j)=idx.second*(first(j,idx.first+1)-first(j,idx.first))+first(j,idx.first);
                        }
                    }
                }
            }
        }else{
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,T> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    for(Eigen::Index g=0;g<nDim;++g){
                        if(d==g){
                            tmp2(g,i,j)=(store(g,i,j,idx.first+1)-store(g,i,j,idx.first))/(point[d](idx.first+1)-point[d](idx.first));
                        }else{
                            tmp2(g,i,j)=idx.second*(store(g,i,j,idx.first+1)-store(g,i,j,idx.first))+store(g,i,j,idx.first);
                        }
                    }
                }
            }
        }
        nt=(d>=1)?table.dimension(d-1):1;
        store=tmp2.reshape(Eigen::array<Eigen::Index,4>{{nDim,nx,m/nt,nt}});
    }
    typedef Eigen::Map<Eigen::Matrix<T,-1,-1,TableOption_&Eigen::RowMajor>> maptype;
    return maptype(store.data(),nDim,nx);
}

//nloptの最適化対象にstd::functionを与えるためのdelegator
struct nlopt_delegator{
    std::function<double(unsigned int,const double*,double*,void*)>* func;
    void* data;
    nlopt_delegator(std::function<double(unsigned int,const double*,double*,void*)>* func_,void* data_);
};
double nlopt_delegatee(unsigned int n,const double* x,double* grad,void* data);

Eigen::MatrixXd ArimotoPotter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

Eigen::VectorXd RK4(const Eigen::VectorXd& x0,double dt,std::function<Eigen::VectorXd (const Eigen::VectorXd& x)> deriv);

void exportMathUtility(py::module &m);
