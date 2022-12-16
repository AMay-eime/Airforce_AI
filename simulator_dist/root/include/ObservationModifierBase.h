#pragma once
#include <pybind11/pybind11.h>
#include "Utility.h"
#include "Callback.h"
#include "Agent.h"
namespace py=pybind11;
namespace nl=nlohmann;

DECLARE_CLASS_WITH_TRAMPOLINE(ObservationModifierBase,Callback)
	/*observationを上書きするCallbackの基底クラス。
    基底クラスでは、ExpertWrapperとSimpleMultiPortCombinerに対して子Agentに対する再帰的な上書き処理を提供しており、
    具象クラスでは通常のAgentに対する上書き処理と、上書き対象となるAgentの判定方法を実装すればよい。
	*/
    public:
    //constructors & destructor
    ObservationModifierBase(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~ObservationModifierBase();
    //functions
    virtual void onGetObservationSpace();
    virtual void onMakeObs();
    virtual py::object overrideObservationSpace(std::shared_ptr<Agent> agent,py::object oldSpace);
    virtual py::object overrideObservationSpaceImpl(std::shared_ptr<Agent> agent,py::object oldSpace)=0;
    virtual py::object overrideObservation(std::shared_ptr<Agent> agent,py::object oldObs);
    virtual py::object overrideObservationImpl(std::shared_ptr<Agent> agent,py::object oldObs)=0;
    virtual bool isTarget(std::shared_ptr<Agent> agent)=0;
};

DECLARE_TRAMPOLINE(ObservationModifierBase)
    virtual py::object overrideObservationSpace(std::shared_ptr<Agent> agent,py::object oldSpace) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(py::object,Base,overrideObservationSpace,agent,oldSpace);
    }
    virtual py::object overrideObservationSpaceImpl(std::shared_ptr<Agent> agent,py::object oldSpace) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(py::object,Base,overrideObservationSpaceImpl,agent,oldSpace);
    }
    virtual py::object overrideObservation(std::shared_ptr<Agent> agent,py::object oldObs) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE(py::object,Base,overrideObservation,agent,oldObs);
    }
    virtual py::object overrideObservationImpl(std::shared_ptr<Agent> agent,py::object oldObs) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(py::object,Base,overrideObservationImpl,agent,oldObs);
    }
    virtual bool isTarget(std::shared_ptr<Agent> agent) override{
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(bool,Base,isTarget,agent);
    }
};

void exportObservationModifierBase(py::module& m);
