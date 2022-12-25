#-*-coding:utf-8-*-
import pybind11
from math import *
import sys
import os
import numpy as np
from ASRCAISim1.libCore import *
from OriginalModelSample.libOriginalModelSample import *
from ASRCAISim1.common import addPythonClass

class BabelAgentReward(AgentReward):
	"""Agentベースのもの。
	(1)仲間との距離が近いと減点
	(2)前進後退のポテンシャル報酬
	(3)エネルギー維持報酬
	(4)beBite減点
	(5)被撃墜減点
	(6)射撃距離報酬
    (7)上下南北境界付近ポテンシャル減点
	(8)直線移動減点
	"""
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		self.pDistance=getValueFromJsonKRD(self.modelConfig,"pDistance",self.randomGen,0.01)
		self.distanceLimit=getValueFromJsonKRD(self.modelConfig,"distanceLimit",self.randomGen,10000)
		self.pLine=getValueFromJsonKRD(self.modelConfig,"pLine",self.randomGen,0.1)
		self.pDetectBonus=getValueFromJsonKRD(self.modelConfig,"pDetectBonus",self.randomGen,0.0)
		self.pEnergy=getValueFromJsonKRD(self.modelConfig,"pEnergy",self.randomGen,5e-7)
		self.pBeBite=getValueFromJsonKRD(self.modelConfig,"pBeBite",self.randomGen,0.2)
		self.pDownReward=getValueFromJsonKRD(self.modelConfig,"pDownReward",self.randomGen,3.0)
		self.pShotRangeReward=getValueFromJsonKRD(self.modelConfig,"pShotRangeReward",self.randomGen,0.2)
		self.shotRangeLimit=getValueFromJsonKRD(self.modelConfig,"shotRangeLimit",self.randomGen,75000)
		self.stateMemoryLength=getValueFromJsonKRD(self.modelConfig,"stateMemoryLength",self.randomGen,30)
		self.estLimit=getValueFromJsonKRD(self.modelConfig,"estLimit",self.randomGen,40000)
		self.pEst=getValueFromJsonKRD(self.modelConfig,"pEst",self.randomGen,1e-4)
		self.pOut=getValueFromJsonKRD(self.modelConfig,"pOut",self.randomGen,1)
		self.minHeightLimit=getValueFromJsonKRD(self.modelConfig,"minHeightLimit",self.randomGen,5000)
		self.maxSideLimit=getValueFromJsonKRD(self.modelConfig,"maxSideLimit",self.randomGen,70000)
		self.gravity=9.8
	def onEpisodeBegin(self):#初期化
		self.j_target="All"#個別のconfigによらず強制的に対象を指定する
		super().onEpisodeBegin()
		o=self.manager.getRuler().observables
		self.westSider=o["westSider"]()
		self.eastSider=o["eastSider"]()
		self.forwardAx=o["forwardAx"]()
		self.friends={
			team:[
				f for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter))
			]
			for team in self.manager.getTeams()
		}
		self.distancePotential={}
		for f in self.reward:
			agent=self.manager.getAgent(f)
			team=agent.parent.getTeam()
			obs=agent.parent.observables
			motion=MotionState(obs["motion"])
			distance=0
			potential=0
			for friend in self.friends[team]:
				myPos=motion.pos
				friendPos=friend.posI()
				dist= np.linalg.norm(np.array(myPos-friendPos))
				distance=max(distance,dist)
			if distance<self.distanceLimit:
				potential=-self.pDistance*(self.distanceLimit-distance)/self.distanceLimit
			else:
				potential=0
			self.distancePotential[f]=potential

		self.firstY={
			f:MotionState(self.manager.getAgent(f).parent.observables["motion"]).pos[1]
			for f in self.reward
		}
		self.linePotential={
			f:0.0
			for f in self.reward
		}
		self.energyPotential={
			f:self.pEnergy*(np.linalg.norm(MotionState(self.manager.getAgent(f).parent.observables["motion"]).vel)**2/2 \
			-2*self.gravity*MotionState(self.manager.getAgent(f).parent.observables["motion"]).pos[2])
			for f in self.reward
		}
		self.enemies={
			team:[
				f for f in self.manager.getAssets(lambda a:a.getTeam()!=team and isinstance(a,Fighter))
			]
			for team in self.manager.getTeams()
		}
		self.enemyMsls={
			team:[
				f for f in self.manager.getAssets(lambda a:a.getTeam()!=team and isinstance(a,Missile))
			]
			for team in self.manager.getTeams()
		}
		self.numMissiles={team:len(self.enemyMsls[team]) for team in self.manager.getTeams()}
		self.biteFlag={team:np.full(self.numMissiles[team],False)
			for team in self.manager.getTeams()}
		self.beBiteRewards={
			f:0.0
			for f in self.reward
		}
		self.aliveFlag={
			f:True
			for f in self.reward
		}
		self.numMyMsls={}
		for f in self.reward:
			num=self.manager.getAgent(f).parent.observables["spec"]["weapon"]["numMsls"]
			self.numMyMsls[f]=num
		self.shotRewards={
			f:0.0
			for f in self.reward
		}
		self.estMemory={
			f:[np.zeros(3) for t in range(self.stateMemoryLength)]
			for f in self.reward
		}
		for f in self.reward:
			motion=MotionState(self.manager.getAgent(f).parent.observables["motion"])
			self.estMemory[f].append(motion.pos+motion.vel*self.stateMemoryLength)
			del self.estMemory[f][0]
		self.outLinePotential={
			f:-self.pOut*(1+max(-1,self.gravity*MotionState(self.manager.getAgent(f).parent.observables["motion"]).pos[2]/self.minHeightLimit))\
			+self.pOut*(1-max(1,abs(MotionState(self.manager.getAgent(f).parent.observables["motion"]).pos[0])/self.maxSideLimit))
			for f in self.reward
		}
		
		self.estRewards={
			f:0.0
			for f in self.reward
		}
		self.memoryTrackFlag={team:np.full(self.numMissiles[team],False)
			for team in self.manager.getTeams()}
	
	#敵との実質距離を計算する計算式、肝要。
	def calcPseudoRelativeRange(self,mPos,mVel,ePos,eVel):
		toEnem=ePos-mPos
		directLength=np.linalg.norm(toEnem)
		toEnem_norm=toEnem/directLength
		mVel_norm=mVel/np.linalg.norm(mVel)
		relVel=mVel+mVel_norm*3*150.0-eVel
		velEffect=np.dot(toEnem_norm,relVel)/150.0
		climbLange=ePos[2]-mPos[2]
		pseudoLength=directLength-velEffect*6000-5*climbLange
		#print(self.getFullName(),velEffect,",",pseudoLength)
		return pseudoLength
	def onStepEnd(self):	
		delta={f:0.0 for f in self.reward}
		for f in self.reward:
			agent=self.manager.getAgent(f)
			obs=agent.parent.observables

			#(5)被撃墜報酬
			#撃墜されればpDownReward
			if not(obs["isAlive"]):
				if self.aliveFlag[f]:
					self.aliveFlag[f]=False
					self.reward[f]-=self.pDownReward
					delta[f]-=self.pDownReward
					#print("DownReward")

			#最終ステップはmotionが欠落するらしいからカット。
			if "motion" in obs.keys():
				motion=MotionState(obs["motion"])
			else:
				#print("最終ステップと思われる。")
				continue
			team=agent.parent.getTeam()
			#(1)仲間との距離が近いと減点。線形テンシャル
			distance=0
			for friend in self.friends[team]:
				myPos=motion.pos
				friendPos=friend.posI()
				dist= np.linalg.norm(np.array(myPos-friendPos))
				distance=max(distance,dist)
			if distance<self.distanceLimit:
				potential=-self.pDistance*(self.distanceLimit-distance)/self.distanceLimit
			else:
				potential=0
			self.reward[f]+=potential-self.distancePotential[f]
			delta[f]+=potential-self.distancePotential[f]
			self.distancePotential[f]=potential
			#print("distancepotential",potential)

			#(2)前進後退すると加減点
			#100000ごとにpLineスコア
			myPos=motion.pos
			assend=(myPos[1]-self.firstY[f])/self.forwardAx[team][1]
			potential=self.pLine*assend/100000
			self.reward[f]+=potential-self.linePotential[f]
			delta[f]+=potential-self.linePotential[f]
			self.linePotential[f]=potential
			#print("linePtential",potential)

			#(3)保持エネルギーによってポテンシャルを得る。
			#最大高度速度なしでおおむね0.1(pEn=5e-7)
			vel=motion.vel
			potential=self.pEnergy*(np.linalg.norm(vel)**2/2-2*self.gravity*myPos[2])
			self.reward[f]+=potential-self.energyPotential[f]
			delta[f]+=potential-self.energyPotential[f]
			self.energyPotential[f]=potential
			#print("energyPotential",potential)

			#(4)beBite減点
			#pBitedはセンサーに引っ掛かったら。pBited*(2-(速度ベクトルと方向ベクトルの内積))/2
			for i,m in enumerate(self.enemyMsls[team]):
				if(m.hasLaunched and m.isAlive):

					if(m.mode==Missile.Mode.SELF and not(self.biteFlag[team][i]) \
						and np.linalg.norm(m.target.pos-myPos)<1000):
						mPos=m.posI()
						dot=np.dot(vel,myPos-mPos)
						normDot=dot/np.linalg.norm(vel)/np.linalg.norm(myPos-mPos)
						rew=self.pBeBite*(2+normDot)/3
						#print(f,"Bited!",rew)
						self.reward[f]-=rew
						delta[f]-=rew
						self.biteFlag[team][i]=True
						self.beBiteRewards[f]-=rew
			#print("biteReward",self.beBiteRewards[f])

			#(6)発射距離報酬
			#発射した際に距離がshotRange以下なら片側2乗報酬
			nowMslNum=obs["weapon"]["remMsls"]
			if float(nowMslNum)<float(self.numMyMsls[f]):
				self.numMyMsls[f]=nowMslNum
				dist=self.shotRangeLimit
				for fighter in self.enemies[team]:
					enemyPos=fighter.posI()
					enemyVel=fighter.velI()
					pseudoRange=self.calcPseudoRelativeRange(myPos,motion.vel,enemyPos,enemyVel)
					dist=min(dist,pseudoRange)
				rew=self.pShotRangeReward*((self.shotRangeLimit-dist)/self.shotRangeLimit)
				self.reward[f]+=rew
				delta[f]+=rew
				self.shotRewards[f]+=rew
				#print("shotReward",rew)

			#上下南北境界付近減点
			#境界付近outLimit以内で最大pOutの線形減点
			potential=-self.pOut*(1+max(-1,myPos[2]/self.minHeightLimit))+self.pOut*(1-max(1,abs(myPos[0])/self.maxSideLimit))
			self.reward[f]+=potential-self.outLinePotential[f]
			delta[f]+=potential-self.outLinePotential[f]
			self.outLinePotential[f]=potential

			#(8)直線移動減点
			#memoryLength秒前の予測位置から距離がestLimit以内だと片側2乗報酬
			#最大pEstポイント失点
			estPos=self.estMemory[f][0]
			distance=np.linalg.norm(myPos-estPos)
			if(distance<self.estLimit):
				rew=self.pEst*pow(1-distance/self.estLimit,2)
				self.reward[f]-=rew
				delta[f]-=rew
				self.estRewards[f]-=rew
				#print("estReward",-rew)
		super().onStepEnd()

	def onEpisodeEnd(self):
		for f in self.reward:
			print(f," total(fromManager):",self.manager.totalRewards[f],\
				" DistanceP:",self.distancePotential[f],\
				" LineP:",self.linePotential[f],\
				" energyP:",self.energyPotential[f],\
				" beBiteT:",self.beBiteRewards[f],\
				" DownedS:",self.aliveFlag[f],\
				" shotRangeT:",self.shotRewards[f],\
				" estimateT:",self.estRewards[f],\
				" outlineP:",self.outLinePotential[f])

#Factoryへの登録
addPythonClass('Reward', 'BabelAgentReward', BabelAgentReward)
