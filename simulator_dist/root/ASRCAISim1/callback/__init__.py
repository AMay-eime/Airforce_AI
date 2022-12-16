# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import json
from ASRCAISim1.common import addPythonClass


#Callbackの登録
from ASRCAISim1.callback.EpisodeMonitor import EpisodeMonitor
addPythonClass('Callback','EpisodeMonitor',EpisodeMonitor)

