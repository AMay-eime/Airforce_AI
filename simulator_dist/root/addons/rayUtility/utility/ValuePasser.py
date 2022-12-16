# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import ray

@ray.remote
class ValuePasser:
	def __init__(self):
		self.__value=None
		self.__readBy=[]
		self.__empty=True
	def setValue(self,value):
		self.__value=value
		self.__readBy=[]
		self.__empty=False
	def isReady(self,readerName=None):
		if self.__empty:
			return False
		if(readerName is None):
			return len(self.__readBy)==0
		else:
			return not readerName in self.__readBy
	def getValue(self,readerName=None):
		assert(not self.__empty)
		if(not readerName in self.__readBy):
			self.__readBy.append(readerName)
		return self.__value
@ray.remote
class KeyValuePasser:
	def __init__(self):
		self.__value={}
		self.__readBy={}
	def setValue(self,key,value):
		self.__value[key]=value
		self.__readBy[key]=[]
	def isReady(self,key,readerName=None):
		if(key in self.__readBy):
			if(readerName is None):
				return len(self.__readBy[key])==0
			else:
				return not readerName in self.__readBy[key]
		else:
			return False
	def getValue(self,key,readerName=None):
		assert(key in self.__value)
		if(not readerName in self.__readBy[key]):
			self.__readBy[key].append(readerName)
		return self.__value[key]

class ValuePasserWrapper:
	def __init__(self,passer):
		self.__passer=passer
	def setValue(self,value):
		ray.get(self.__passer.setValue.remote(value))
	def isReady(self,readerName=None):
		return ray.get(self.__passer.isReady.remote(readerName))
	def getValue(self,readerName=None):
		return ray.get(self.__passer.getValue.remote(readerName))
class KeyValuePasserWrapper:
	def __init__(self,passer):
		self.__passer=passer
	def setValue(self,key,value):
		ray.get(self.__passer.setValue.remote(key,value))
	def isReady(self,key,readerName=None):
		return ray.get(self.__passer.isReady.remote(key,readerName))
	def getValue(self,key,readerName=None):
		return ray.get(self.__passer.getValue.remote(key,readerName))
