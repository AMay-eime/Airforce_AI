# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os,json
import ASRCAISim1
from ASRCAISim1.libCore import Factory
from ASRCAISim1.common import addPythonClass
if(os.name=="nt"):
    libName="AgentIsolation"
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
    from .libAgentIsolation import *
except ImportError as e:
    if(os.name=="nt"):
        print("Failed to import the module. If you are using Windows, please make sure that: ")
        print('(1) If you are using conda, CONDA_DLL_SEARCH_MODIFICATION_ENABLE should be set to 1.')
        print('(2) dll dependencies (such as nlopt) are located appropriately.')
    raise e
from .AgentDelegator import AgentDelegator
from .AgentDelegatee import AgentDelegatee
from .PolicyDelegator import PolicyDelegator
from .PolicyDelegatee import PolicyDelegatee
addPythonClass('Agent','AgentDelegator',AgentDelegator)
