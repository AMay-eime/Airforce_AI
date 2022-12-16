# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os,json
from ASRCAISim1.libCore import *

def addPythonClass(baseName,className,clsObj):
    def creator(modelConfig,instanceConfig):
        ret=clsObj(modelConfig,instanceConfig)
        return Factory.keepAlive(ret)
    Factory.addClass(baseName,className,creator)

