#-*-coding:utf-8-*-
from math import *
import os
import sys
import numpy as np
from ASRCAISim1.libCore import *
from OriginalModelSample.libOriginalModelSample import *
from ASRCAISim1.common import addPythonClass

class BabelTeamReward(TeamReward):
	"""以下の観点に基づいた報酬。
	(1)敵機追跡
	(2)Bite報酬
	(3)Approach,EnemyApproach報酬
	(4)敵機撃墜(未実装。ScoreRewardで代用)
	(5)ライン突破報酬(未実装。ScoreRewardで代用)
	(6)撃墜観測報酬
	"""
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		self.pBite=getValueFromJsonKRD(self.modelConfig,"pBite",self.randomGen,0.3)
		self.pDetect=getValueFromJsonKRD(self.modelConfig,"pDetect",self.randomGen,0.05)
		self.pDetectBonus=getValueFromJsonKRD(self.modelConfig,"pDetectBonus",self.randomGen,0.05)
		self.maxFI=getValueFromJsonKRD(self.modelConfig,"maxFI",self.randomGen,300)
		self.pSelfLine=getValueFromJsonKRD(self.modelConfig,"pSelfLine",self.randomGen,1)
		self.apLength_self=getValueFromJsonKRD(self.modelConfig,"apLength_self",self.randomGen,50000)
		self.pEnemyLine=getValueFromJsonKRD(self.modelConfig,"pEnemyLine",self.randomGen,3)
		self.apLength_enemy=getValueFromJsonKRD(self.modelConfig,"apLength_enemy",self.randomGen,100000)
		self.pDetectDown=getValueFromJsonKRD(self.modelConfig,"pDetectDown",self.randomGen,0.3)
	def calcSig(self,x,maxValue,maxInput):
		return maxValue*(exp(4*x/maxInput)-1)/(exp(4*x/maxInput)+1)
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
			for team in self.reward
		}
		self.enemies={
			team:[
				f for f in self.manager.getAssets(lambda a:a.getTeam()!=team and isinstance(a,Fighter))
			]
			for team in self.reward
		}
		self.totalFIs={
			team: 0.0
			for team in self.reward
		}
		self.totalFI_rewards={
			team: 0.0
			for team in self.reward
		}
		"""self.totalBiteReward={
			team: 0.0
			for team in self.reward
		}"""
		self.goalPos_ys={}
		for team in self.reward:
			gl=self.forwardAx[team][1]*1000000
			for asset in self.enemies[team]:
				if abs(asset.posI()[1])<abs(gl):
					gl=asset.posI()[1]
			self.goalPos_ys[team]=gl
		self.self_yPotentialReward={
			team: 0.0
			for team in self.reward
		}
		self.enemy_yPotentialReward={
			team: 0.0
			for team in self.reward
		}
		self.enemyAliveFrag={
			team:[
				True for f in self.manager.getAssets(lambda a:a.getTeam()!=team and isinstance(a,Fighter))
			]
			for team in self.reward
		}
		self.lastTrack=[]
		self.friendMsls={
			team:[
				f for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Missile))
			]
			for team in self.reward
		}
		self.numMissiles={team:len(self.friendMsls[team]) for team in self.reward}
		self.biteFlag={team:np.full(self.numMissiles[team],False)
			for team in self.reward}
	def onStepEnd(self):
		delta={t:0.0 for t in self.reward}
		for team in self.reward:
			#(2)Biteへの加点
			#pBiteはそのまま。トラックモード変更されたら加点。
			for i,m in enumerate(self.friendMsls[team]):
				if(m.hasLaunched and m.isAlive):
					if(m.mode==Missile.Mode.SELF and not self.biteFlag[team][i]):
						self.reward[team]+=self.pBite
						delta[team]+=self.pBite
						self.biteFlag[team][i]=True
						#self.totalBiteReward[team]+=self.pBite
			#(1)敵機追跡への加点。
			#航跡情報量(fi)=航跡数*秒数とし、その積算値(totalFI)に対して
			#シグモイドで計算。
			#maxFIはポイント取得可能なおよそのFI上限で固定(暫定1000)。
			#重みづけを変更したければpDetect(最大値になる)で。
			track=[]
			for f in self.friends[team]:
				if(f.isAlive()):
					track.extend([Track3D(t) for t in f.observables["sensor"]["track"]])
					break
			track=list(set(track))
			numAlive=0
			numTracked=0
			for f in self.enemies[team]:
				if(f.isAlive()):
					numAlive+=1
					for t in track:
						if(np.linalg.norm(t.pos-f.posI())<500):
							numTracked+=1
							break
			if(numAlive>0):
				self.totalFIs[team]+=numTracked
				self.reward[team]+=self.calcSig(self.totalFIs[team],self.pDetect,self.maxFI)-self.totalFI_rewards[team]
				delta[team]+=self.calcSig(self.totalFIs[team],self.pDetect,self.maxFI)-self.totalFI_rewards[team]
				self.totalFI_rewards[team]=self.calcSig(self.totalFIs[team],self.pDetect,self.maxFI)
			#print(team,"totalFIReward",self.totalFI_rewards[team])
			#(6)観測している対象が撃墜された時に発生する報酬(未実装)
			
			
			potential=0
			for f in self.friends[team]:
				pos=f.posI()
				vel=f.velI()
				omega=f.omegaI()
				forward=self.forwardAx[team][1]
				#(3)味方が境界付近にいる場合の加点
				#AgentRewardと同様の計算式である。重要な方が前に出やすくする。
				#2乗ポテンシャル
				p_Point=self.goalPos_ys[team]-self.apLength_self*forward
				if forward*(pos[1]-p_Point) > 0:
					pot=self.pSelfLine*pow((pos[1]-p_Point)/p_Point,2)
					potential=max(potential,pot)
			#print("selfpotential",potential)
			self.reward[team]+=potential-self.self_yPotentialReward[team]
			delta[team]+=potential-self.self_yPotentialReward[team]
			self.self_yPotentialReward[team]=potential

			potential=0
			for f in self.enemies[team]:
				pos=f.posI()
				vel=f.velI()
				omega=f.omegaI()
				#(3)敵が境界付近にいる場合の加点
				#上と同様の計算式。こちらの方がデフォルトでは重い。
				#中心から発生する2乗ポテンシャル。pEnemyLineにapLength_enemyでなる。
				forward=self.forwardAx[team][1]
				if forward*pos[1]<0:
					pot=-self.pEnemyLine*pow((pos[1]/self.apLength_enemy),2)
					potential=min(potential,pot)
			#print("enemypotential",potential)
			self.reward[team]+=potential-self.enemy_yPotentialReward[team]
			delta[team]+=potential-self.enemy_yPotentialReward[team]
			self.enemy_yPotentialReward[team]=potential
		super().onStepEnd()


#Factoryへの登録
addPythonClass('Reward', 'BabelTeamReward', BabelTeamReward)
Factory.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/BabelRewardConfig.json"))
