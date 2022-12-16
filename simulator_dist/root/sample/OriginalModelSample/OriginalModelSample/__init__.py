# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os,json
import ASRCAISim1
from ASRCAISim1.libCore import Factory
from ASRCAISim1.common import addPythonClass
if(os.name=="nt"):
    libName="OriginalModelSample"
    pyd=os.path.join(os.path.dirname(__file__),"lib"+libName+".pyd")
    if(not os.path.exists(pyd) or os.path.getsize(pyd)==0):
        print("Info: Maybe the first run after install. A hardlink to a dll will be created.")
        if(os.path.exists(pyd)):
            os.remove(pyd)
        dll=os.path.join(os.path.dirname(__file__),"lib"+libName+".dll")
        if(not os.path.exists(dll)):
            dll=os.path.join(os.path.dirname(__file__),""+libName+".dll")
        if(not os.path.exists(dll)):
            raise FileNotFoundError("There is no lib"+libName+".dll or "+libName+".dll.")
        import subprocess
        subprocess.run([
            "fsutil",
            "hardlink",
            "create",
            pyd,
            dll
        ])
try:
    from OriginalModelSample.libOriginalModelSample import *
except ImportError as e:
    if(os.name=="nt"):
        print("Failed to import the module. If you are using Windows, please make sure that: ")
        print('(1) If you are using conda, CONDA_DLL_SEARCH_MODIFICATION_ENABLE should be set to 1.')
        print('(2) dll dependencies (such as nlopt) are located appropriately.')
    raise e

from OriginalModelSample.R4PyAgentSample01S import R4PyAgentSample01S
from OriginalModelSample.R4PyAgentSample01M import R4PyAgentSample01M
from OriginalModelSample.R4PyRewardSample01 import R4PyRewardSample01
from OriginalModelSample.R4PyRewardSample02 import R4PyRewardSample02

addPythonClass('Agent','R4PyAgentSample01S',R4PyAgentSample01S)
addPythonClass('Agent','R4PyAgentSample01M',R4PyAgentSample01M)
addPythonClass('Reward','R4PyRewardSample01',R4PyRewardSample01)
addPythonClass('Reward','R4PyRewardSample02',R4PyRewardSample02)

Factory.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/sampleConfig.json"))
