// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "MathUtility.h"
#include <pybind11/operators.h>
#include <Eigen/Eigenvalues>
namespace py=pybind11;

double sinc(const double &x,const double &eps){
    if(abs(x)>eps){
        return sin(x)/x;
    }else{
        double x2=x*x;
        return 1.-x2/6.+x2*x2/120.-x2*x2*x2/5040.;//+O(x8)
    }
}
double sincMinusOne(const double &x,const double &eps){
    if(abs(x)>eps){
        return sin(x)/x-1.;
    }else{
        double x2=x*x;
        return -x2/6.+x2*x2/120.-x2*x2*x2/5040.;//+O(x8)
    }
}
double sincMinusOne_x2(const double &x,const double &eps){
    if(abs(x)>eps){
        return (sin(x)/x-1.)/(x*x);
    }else{
        double x2=x*x;
        return -1./6.+x2/120.-x2*x2/5040.+x2*x2*x2/362880.;//+O(x8)
    }
}
double oneMinusCos(const double &x,const double &eps){
    if(abs(x)>eps){
        return 1-cos(x);
    }else{
        double x2=x*x;
        return x2/2.-x2*x2/24.+x2*x2*x2/720.;//+O(x8)
    }
}
double oneMinusCos_x2(const double &x,const double &eps){
    if(abs(x)>eps){
        return (1-cos(x))/(x*x);
    }else{
        double x2=x*x;
        return 1./2.-x2/24.+x2*x2/720.-x2*x2*x2/40320.;//+O(x8)
    }
}
Eigen::Vector3d getOrthogonalVector(const Eigen::Vector3d &v){
    double n=v.norm();
    if(n==0){
        return Eigen::Vector3d(1,0,0);
    }
    int idx;
    v.maxCoeff(&idx);
    int tmp=(idx+1)%3;
    return v.cross(Eigen::Vector3d(0==tmp,1==tmp,2==tmp)).normalized();
}

nlopt_delegator::nlopt_delegator(std::function<double(unsigned int,const double*,double*,void*)>* func_,void* data_){
    func=func_;
    data=data_;
}
double nlopt_delegatee(unsigned int n,const double* x,double* grad,void* data){
    auto d=reinterpret_cast<nlopt_delegator*>(data);
    double ret=(*d->func)(n,x,grad,d->data);
    return ret;
};

Eigen::MatrixXd ArimotoPotter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R){
    Eigen::Index dim_x=A.rows();
    Eigen::Index dim_u=B.cols();
    Eigen::MatrixXd H=Eigen::MatrixXd::Zero(2*dim_x,2*dim_x);
    H<<A,-B*R.inverse()*B.transpose(),-Q,-A.transpose();
    Eigen::EigenSolver<Eigen::MatrixXd> eig(H);
    Eigen::MatrixXcd eigvec=Eigen::MatrixXcd::Zero(2*dim_x,dim_x);
    int j=0;
    for(int i=0;i<2*dim_x;++i){
        if(eig.eigenvalues()[i].real()<0.){
            eigvec.col(j)=eig.eigenvectors().block(0,i,2*dim_x,1);
            ++j;
        }
    }
    Eigen::MatrixXcd Vs_1,Vs_2;
    Vs_1=eigvec.block(0,0,dim_x,dim_x);
    Vs_2=eigvec.block(dim_x,0,dim_x,dim_x);
    return (Vs_2*Vs_1.inverse()).real();
}

Eigen::VectorXd RK4(const Eigen::VectorXd& x0,double dt,std::function<Eigen::VectorXd (const Eigen::VectorXd& x)> deriv){
    Eigen::VectorXd k1=deriv(x0);
    //return x0+dt*k1;
    Eigen::VectorXd k2=deriv(x0+dt*0.5*k1);
    Eigen::VectorXd k3=deriv(x0+dt*0.5*k2);
    Eigen::VectorXd k4=deriv(x0+dt*k3);
    return x0+dt/6.0*(k1+2.0*k2+2.0*k3+k4);
}

void exportMathUtility(py::module &m)
{
    using namespace pybind11::literals;
    m.def("sinc",&sinc,"x"_a,"eps"_a=1e-3);
    m.def("sincMinusOne",&sincMinusOne,"x"_a,"eps"_a=1e-3);
    m.def("sincMinusOne_x2",&sincMinusOne_x2,"x"_a,"eps"_a=1e-3);
    m.def("oneMinusCos",&oneMinusCos,"x"_a,"eps"_a=1e-3);
    m.def("oneMinusCos_x2",&oneMinusCos_x2,"x"_a,"eps"_a=1e-3);
    m.def("getOrthogonalVector",&getOrthogonalVector);
    m.def("ArimotoPotter",&ArimotoPotter);
    m.def("skewMatrix",&skewMatrix<double>);
}
