# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import os,json
from ASRCAISim1.libCore import Factory,Viewer
from ASRCAISim1.common import addPythonClass
try:
	from ASRCAISim1.viewer.GodView import GodView

	addPythonClass('Viewer','GodView',GodView)
except Exception as ex:
	ex_str=str(ex)
	class DummyGodView(Viewer):
		def __init__(self,modelConfig,instanceConfig):
			super().__init__(modelConfig,instanceConfig)
			self.exception=ex_str
			print("============================")
			print("Warining: GodView was not imported collectly. The exception occured is,")
			print(self.exception)
			print("To continue simulation, a dummy no-op Viewer is used instead.")
			print("============================")
	addPythonClass('Viewer','GodView',DummyGodView)
