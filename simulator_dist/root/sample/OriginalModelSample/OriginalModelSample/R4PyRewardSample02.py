# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
from collections import defaultdict
import sys
import numpy as np
from ASRCAISim1.libCore import *
from OriginalModelSample.libOriginalModelSample import *

class R4PyRewardSample02(AgentReward):
	"""いくつかの観点に基づいた個別報酬の実装例。
	(1)敵ミサイルの回避成功
	(2)被撃墜
	(3)墜落
	(4)撃墜
	(5)場外
	(6)終了時の生存
	(7)余剰燃料の有無
	"""
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		self.pAvoid=getValueFromJsonKRD(self.modelConfig,"pAvoid",self.randomGen,0.0)
		self.pHitE_head=getValueFromJsonKRD(self.modelConfig,"pHitE_head",self.randomGen,0.0)
		self.pHitE_tail=getValueFromJsonKRD(self.modelConfig,"pHitE_tail",self.randomGen,0.0)
		self.pCrash=getValueFromJsonKRD(self.modelConfig,"pCrash",self.randomGen,0.0)
		self.pHit_head=getValueFromJsonKRD(self.modelConfig,"pHit_head",self.randomGen,0.0)
		self.pHit_tail=getValueFromJsonKRD(self.modelConfig,"pHit_tail",self.randomGen,0.0)
		self.pOut=getValueFromJsonKRD(self.modelConfig,"pOut",self.randomGen,0.0)
		self.pAlive=getValueFromJsonKRD(self.modelConfig,"pAlive",self.randomGen,0.0)
		self.pFuelShortage=getValueFromJsonKRD(self.modelConfig,"pFuelShortage",self.randomGen,0.0)
	def onEpisodeBegin(self):#初期化
		self.j_target="All"#個別のconfigによらず強制的に対象を指定する
		super().onEpisodeBegin()
		o=self.manager.getRuler()().observables
		self.westSider=o["westSider"]()
		self.eastSider=o["eastSider"]()
		self.dOut=o["dOut"]()
		self.dLine=o["dLine"]()
		self.forwardAx=o["forwardAx"]()
		self.sideAx=o["sideAx"]()
		self.fuelMargin=o["fuelMargin"]()
		self.distanceFromBase=o["distanceFromBase"]()
		self.manager.addEventHandler("Crash", self.onCrash)  # 墜落監視用
		self.manager.addEventHandler("Hit", self.onHit)  # 撃墜監視用
		self.parents=defaultdict(lambda:[])
		self.missiles=defaultdict(lambda:[])
		self.checked={}
		self.deadFighters=[]
		self.assetToTargetAgent={}
		for asset in self.manager.getAssets():
			asset=asset()
			if(isinstance(asset,Missile)):
				self.missiles[asset.getTeam()].append(asset)
				self.checked[asset.getFullName()]=False
		for key in self.reward:
			agent=self.manager.getAgent(key)()
			for p in agent.parents.values():
				f=self.manager.getAsset(p.getFullName())
				self.parents[key].append(f)
				self.assetToTargetAgent[p.getFullName()]=key
	def onCrash(self, args):
		"""墜落の監視用コールバック
		"""
		asset=PhysicalAsset.from_json_weakref(args)()
		name=asset.getFullName()
		if not name in self.deadFighters:
			if(name in self.assetToTargetAgent):
				self.reward[self.assetToTargetAgent[name]]+=self.pCrash
			self.deadFighters.append(name)
	def onHit(self, args):
		"""撃墜の監視用コールバック
		"""
		wpn=PhysicalAsset.from_json_weakref(args["wpn"])()
		tgt=PhysicalAsset.from_json_weakref(args["tgt"])()
		if not tgt.getFullName() in self.deadFighters:
			vel=tgt.motion.vel
			vel=vel/np.linalg.norm(vel)
			dir=(wpn.motion.pos-tgt.motion.pos)
			dir/=np.linalg.norm(dir)
			aa=acos(np.dot(vel,dir))
			name=tgt.getFullName()
			if(name in self.assetToTargetAgent):
				self.reward[self.assetToTargetAgent[name]]+=((self.pHitE_head+self.pHitE_tail)+cos(aa)*(self.pHitE_head-self.pHitE_tail))/2
			name=wpn.parent().getFullName()
			if(name in self.assetToTargetAgent):
				self.reward[self.assetToTargetAgent[name]]+=((self.pHit_head+self.pHit_tail)+cos(aa)*(self.pHit_head-self.pHit_tail))/2
			self.deadFighters.append(tgt.getFullName())
	def onInnerStepEnd(self):
		track=[]
		for asset in self.manager.getAssets():
			asset=asset()
			if(isinstance(asset,Missile)):
				if(not self.checked[asset.getFullName()] and asset.hasLaunched and not asset.isAlive()):
					tgt=asset.target
					for name in self.reward:
						for p in self.parents[name]:
							if(tgt.isSame(p)):
								self.reward[name]+=self.pAvoid
					self.checked[asset.getFullName()]=True
			elif(isinstance(asset,Fighter)):
				if(asset.isAlive()):
					team=asset.getTeam()
					outDist=max(0.0,abs(np.dot(self.sideAx[team],asset.posI()[0:2]))-self.dOut)
					name=asset.getFullName()
					if(name in self.assetToTargetAgent):
						self.reward[self.assetToTargetAgent[name]]+=(outDist/1000.)*self.pOut*self.interval[SimPhase.ON_INNERSTEP_END]*self.manager.getBaseTimeStep()

						if(asset.optCruiseFuelFlowRatePerDistance>0.0):
							maxReachableRange=asset.fuelRemaining/asset.optCruiseFuelFlowRatePerDistance
							distanceFromLine=np.dot(self.forwardAx[team],asset.posI()[0:2])+self.dLine
							excess=maxReachableRange/(1+self.fuelMargin)-(distanceFromLine+self.distanceFromBase[team])
						if(excess<0):
							self.reward[self.assetToTargetAgent[name]]+=self.pFuelShortage*self.interval[SimPhase.ON_INNERSTEP_END]*self.manager.getBaseTimeStep()
	def onStepEnd(self):
		ruler=self.manager.getRuler()()
		if(ruler.endReason!=ruler.EndReason.NOTYET):
			for asset in self.manager.getAssets():
				asset=asset()
				if(asset.isAlive()):
					name=asset.getFullName()
					if(isinstance(asset,Fighter)):
						if(name in self.assetToTargetAgent):
							if(ruler.isReturnableToBase(asset)):
								#帰還可能なら生存点
								self.reward[self.assetToTargetAgent[name]]+=self.pAlive
							else:
								#帰還不可能なら墜落ペナルティ
								self.reward[self.assetToTargetAgent[name]]+=self.pCrash
		super().onStepEnd()
