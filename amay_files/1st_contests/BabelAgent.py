#-*-coding:utf-8-*-
import pybind11
from math import *
from gym import spaces
import numpy as np
import sys
import os
import random
from ASRCAISim1.libCore import *
from OriginalModelSample.libOriginalModelSample import *
from ASRCAISim1.common import addPythonClass

print("BabelAgent_Thor_1.2")
class BabelAgent(SingleAssetAgent):
	"""行動の解像度を上昇させ、最終観測点を記憶するようにしたAgent。
	1機につき1つのAgentを割り当てる、分散方式での行動判断モデルの実装例。
	時系列情報の活用については、RNNを使わずとも、キーフレームでの値を並列で吐き出せる形で実装している。
	もしRNNを使う場合は、キーフレームを指定せず、瞬時値をそのまま観測として出力すればよい。
	場外・墜落の回避については、学習に委ねず南北・高度方向の移動に制限をかけることで対処している。
	もし一時的に場外に出ることが勝利に繋がりうると思えば、制限をなくしてもよい。
	速度については、遅くなりすぎると機動力が低下するため、下限速度を設定できるようにしている。
	射撃については、全弾連射する等ですぐに攻撃能力を喪失するような状況を予め回避するため、同時射撃数を制限できるようにしている。
	1. 観測データについて
		* 東側でも西側でも同じになるように、西側のベクトルはx,y軸について反転させる。
		* キーフレームは「n秒前のフレーム」として、そのnのlistをconfigで"pastPoints"キーにより与える
		* 各フレームの観測データの内訳
			1. 自機情報
				1. 位置・・・x,y成分をRulerのdOutとdLineの大きい方で除し、z成分をRulerのhLimで除して正規化したもの。
				2. 速度・・・速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
				3. 残弾数・・・intそのまま。（ただし一括してBoxの値として扱われる）
			2. 味方機情報
				1. 位置・・・距離をrangeNormalizerで正規化したものと、相対位置方向の単位ベクトルの4次元に分解したもの。
				2. 速度・・・自機と同じ
				3.残弾数・・・自機と同じ
			3. 敵機情報・・・見えている敵機航跡のうち、近いものから順に最大maxTrackNum機分。無い場合は0埋め。
				1. 位置・・・味方機と同じ
				2. 速度・・・味方機と同じ
			4. 味方誘導弾情報(自機も含む)
				1. 位置・・・味方機と同じ
				2. 速度・・・自機と同じ
				3. 誘導状態・・・guided,self,memoryの3通りについて、one-hot形式で与える。
			5. 敵誘導弾情報・・・見えている誘導弾航跡のうち、近いものから順に最大maxMissileNum発分。無い場合は0埋め。
				1. 到来方向・・・慣性座標系での到来方向を表した単位ベクトル。
	2. 行動の形式について
		左右旋回、上昇・下降、加減速、射撃対象の4種類を離散化したものをMultiDiscreteで与える。
		1. 左右旋回・・・自機正面を0とした「行きたい方角(右を正)」で指定。
		2. 上昇・下降・・・水平を0とし、上昇・下降角度(下降を正)で指定。
		3. 加減速・・・目標速度を基準速度(≒現在速度)+ΔVとすることで表現し、ΔVで指定。
		4. 射撃対象・・・一番近いやつに射撃をする。射撃間隔は最短でも30秒必要になる。
	Attributes:
		* configで指定するもの
		turnScale (double): 最大限に左右旋回する際の目標方角の値。degで指定。
		turnDim (int): 左右旋回の離散化数。0を用意するために奇数での指定を推奨。
		pitchScale (double): 最大限に上昇・下降する際の目標角度の値。degで指定。
		pitchDim (int): 上昇・下降の離散化数。0を用意するために奇数での指定を推奨。
		accelScale (double): 最大限に加減速する際のΔVの絶対値。
		accelDim (int): 加減速の離散化数。0を用意するために奇数での指定を推奨。
		hMin (double): 高度下限。下限を下回ったら下回り具合に応じて上昇方向に針路を補正する。
		hMax (double): 高度上限。上限を上回ったら上回り具合に応じて下降方向に針路を補正する。
		dOutLimitRatio (double): 南北方向への戦域逸脱を回避するための閾値。中心からdOut×dOutLimitRatio以上外れたら外れ具合に応じて中心方向に針路を補正する。
		rangeNormalizer (float): 距離の正規化のための除数
		fgtrVelNormalizer (float): 機体速度の正規化のための除数
		mslVelNormalizer (float): 誘導弾速度の正規化のための除数
		maxTrackNum (dict): 観測データとして使用する味方、敵それぞれの航跡の最大数。{"Friend":3,"Enemy":4}のようにdictで指定する。
		maxMissileNum (dict): 観測データとして使用する味方、敵それぞれの誘導弾情報の最大数。書式はmaxTrackNumと同じ。
		pastPoints (list of int): キーフレームのリスト。「nステップ前のフレーム」としてnのリストで与える。空で与えれば瞬時値のみを使用する。RNNにも使える(はず)
		pastData (list of numpy.ndarray): 過去のフレームデータを入れておくリスト。キーフレームの間隔が空いていても、等間隔でなければ全フレーム分必要なので、全フレーム部分用意し、リングバッファとして使用する。
		minimumV (double): 下限速度。この値を下回ると指定した目標速度に戻るまで強制的に加速する。
		minimumRecoveryV (Double): 速度下限からの回復を終了する速度。下限速度に達した場合、この値に達するまで強制的に加速する。
		minimumRecoveryDstV (Double): 速度下限からの回復目標速度。下限速度に達した場合、この値を目標速度として加速する。
		maxSimulShot (int): 同時射撃数の制限。自身が発射した、飛翔中の誘導弾がこの数以下のときのみ射撃可能。
	"""
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		self.turnScale=deg2rad(getValueFromJsonKRD(self.modelConfig,"turnScale",self.randomGen,90.0))
		self.turnDim=getValueFromJsonKRD(self.modelConfig,"turnDim",self.randomGen,11)
		self.pitchScale=deg2rad(getValueFromJsonKRD(self.modelConfig,"pitchScale",self.randomGen,60))
		self.pitchDim=getValueFromJsonKRD(self.modelConfig,"pitchDim",self.randomGen,17)
		self.accelScale=getValueFromJsonKRD(self.modelConfig,"accelScale",self.randomGen,30.0)
		self.accelDim=getValueFromJsonKRD(self.modelConfig,"accelDim",self.randomGen,7)
		self.hMin=getValueFromJsonKRD(self.modelConfig,"hMin",self.randomGen,1000.0)
		self.hMax=getValueFromJsonKRD(self.modelConfig,"hMax",self.randomGen,200000.0)
		self.dOutLimitRatio=getValueFromJsonKRD(self.modelConfig,"dOutLimitRatio",self.randomGen,0.95)
		self.rangeNormalizer=getValueFromJsonKRD(self.modelConfig,"rangeNormalizer",self.randomGen,100000.0)
		self.velocityNormalizer=getValueFromJsonKRD(self.modelConfig,"velocityNormalizer",self.randomGen,100.0)
		self.fgtrVelNormalizer=getValueFromJsonKRD(self.modelConfig,"fgtrVelNormalizer",self.randomGen,300.0)#正規化用
		self.mslVelNormalizer=getValueFromJsonKRD(self.modelConfig,"mslVelNormalizer",self.randomGen,2000.0)#正規化用
		self.maxTrackNum=getValueFromJsonKRD(self.modelConfig,"maxTrackNum",self.randomGen,{"Friend":1,"Enemy":2})#味方(自分以外)及び敵の最大の航跡数
		self.selfHistoryTrackNum = getValueFromJsonKRD(self.modelConfig,"selfHistoryTrackNum",self.randomGen,5)
		self.enemyHistoryTrackNum = getValueFromJsonKRD(self.modelConfig,"selfHistoryTrackNum",self.randomGen,4)
		self.maxMissileNum=getValueFromJsonKRD(self.modelConfig,"maxMissileNum",self.randomGen,{"Friend":3,"Enemy":1})#考慮する誘導弾の最大数
		self.pastPoints=getValueFromJsonKRD(self.modelConfig,"pastPoints",self.randomGen,[3,9,30])
		self.minimumV=getValueFromJsonKRD(self.modelConfig,"minimumV",self.randomGen,150.0)
		self.minimumRecoveryV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryV",self.randomGen,180.0)
		self.minimumRecoveryDstV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryDstV",self.randomGen,200.0)
		self.maxSimulShot=getValueFromJsonKRD(self.modelConfig,"maxSimulShot",self.randomGen,2)
		self.singleDim=11+11*self.maxTrackNum["Friend"]+6*self.maxTrackNum["Enemy"]+6*self.maxMissileNum["Friend"]+6*self.maxMissileNum["Enemy"]+1+7*self.maxTrackNum["Enemy"]
		self.pastObsDim=18
		self.rMaxs=getValueFromJsonKRD(self.modelConfig,"rMaxs",self.randomGen,[50000, 25000, 0])
		self.fineSearchArea=getValueFromJsonKRD(self.modelConfig,"fineSearchArea",self.randomGen,[95000,80])
		if(len(self.pastPoints)>0):
			self.pastData=[np.zeros(self.singleDim) for i in range(max(self.pastPoints))]#過去の観測情報を入れるためのリスト
		else:
			self.pastData=[]
		self.lastTrackInfo=[] #makeObsで出力したときの航跡情報。(deployと)controlで使用するために保持しておく必要がある。
		self.launchFlag=False
		self.velRecovery=False
		self.target=Track3D()
		#ここから自分で定義した変数。
		self.enemyIDs = []
		self.lostTime={}
		self.lostPos={}
		self.lostVel={}
		self.matchProcess=0
		self.preTracks=[]
		self.shootTimer=0
		self.missleAlert=False
		self.skillCoolTime=100
		self.skillOriginalV=np.array([0,0,0])
		self.distRecovery=False
		self.breakTime=100
		self.warnTime=0
	def validate(self):
		rulerObs=self.manager.getRuler().observables()
		self.dOut=rulerObs["dOut"]
		self.dLine=rulerObs["dLine"]
		self.hLim=rulerObs["hLim"]
		self.xyInv=np.array([1,1,1]) if (self.getTeam()==rulerObs["eastSider"]) else np.array([-1,-1,1])
		self.xyQuatInv=np.array([1,1,1,1]) if (self.getTeam()==rulerObs["eastSider"]) else np.array([1,-1,-1,1])
		if(self.parent.isinstance(CoordinatedFighter)):
			self.parent.setFlightControllerMode("fromDirAndVel")
		else:
			fSpec=self.parent.observables["spec"]["dynamics"]
			self.omegaScale=np.array([1.0/fSpec["rollMax"](),1.0/fSpec["pitchMax"](),1.0/fSpec["yawMax"]()])
		myMotion=MotionState(self.parent.observables["motion"])
		self.baseV=np.linalg.norm(myMotion.vel)
		self.dstV=self.baseV
		self.dstDir=np.array([0,-self.xyInv[1],0])
		self.lastAction=np.array([self.turnDim//2,self.pitchDim//2,self.accelDim//2,0])
		self.preGuardian=False
	def observation_space(self):
		#自機(3+3+4)dim、自機履歴(3+3)、味方機(3+3+4)dim、彼機(3+3)dim、味方誘導弾(3+3)dim、彼誘導弾(2+2)dim、
		#メタ情報(ゲーム進行度)1dim、最終航跡情報(7+7)dim、計127dim
		floatLow,floatHigh=-sys.float_info.max,sys.float_info.max
		self_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow,floatLow,-1,-1,-1,0])
		self_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,1,1,1,floatHigh])
		self_history_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow])
		self_history_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh])
		friend_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow,floatLow,-1,-1,-1])
		friend_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,1,1,1])
		enemy_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow])
		enemy_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh])
		msl_friend_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow])
		msl_friend_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh])
		msl_enemy_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow])
		msl_enemy_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatLow,floatLow])
		meta_memory_low=np.array([floatLow])
		meta_memory_high=np.array([floatHigh])
		lost_info_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow,0])
		lost_info_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh])
		obs_low=np.concatenate((
			self_low,
			np.concatenate([friend_low]*self.maxTrackNum["Friend"]),
			np.concatenate([enemy_low]*self.maxTrackNum["Enemy"]),
			np.concatenate([msl_friend_low]*self.maxMissileNum["Friend"]),
			np.concatenate([msl_enemy_low]*self.maxMissileNum["Enemy"]),
			np.concatenate([meta_memory_low]),
			np.concatenate([lost_info_low]*self.maxTrackNum["Enemy"])
		))
		obs_high=np.concatenate((
			self_high,
			np.concatenate([friend_high]*self.maxTrackNum["Friend"]),
			np.concatenate([enemy_high]*self.maxTrackNum["Enemy"]),
			np.concatenate([msl_friend_high]*self.maxMissileNum["Friend"]),
			np.concatenate([msl_enemy_high]*self.maxMissileNum["Enemy"]),
			np.concatenate([meta_memory_high]),
			np.concatenate([lost_info_high]*self.maxTrackNum["Enemy"])
		))
		#過去データは一部の利用となる。
		past_low=np.array([floatLow,floatLow,floatLow,floatLow,floatLow,floatLow])
		past_high=np.array([floatHigh,floatHigh,floatHigh,floatHigh,floatHigh,floatHigh])
		obs_low=np.concatenate((obs_low, np.concatenate([past_low]*len(self.pastPoints))))
		obs_high=np.concatenate((obs_high,np.concatenate([past_high]*len(self.pastPoints))))
		return spaces.Box(low=obs_low,high=obs_high,dtype=np.float32)
	def makeObs(self):
		#何回目の観測かを計算(初回を0とする)
		count=round(self.manager.getTickCount()/self.manager.getAgentInterval())
		current, memory=self.makeSingleObs()
		if(len(self.pastPoints)==0):
			return current
		if(count==0):
			#初回は過去の仮想データを生成(誘導弾なし、敵側の情報なし)
			myMotion=MotionState(self.parent.observables["motion"])
			myPos0=myMotion.pos
			myVel=myMotion.vel
			friendPos0=[np.zeros([3])]*self.maxTrackNum["Friend"]
			friendVel=[np.zeros([3])]*self.maxTrackNum["Friend"]
			idx=0
			for n,f in self.parent.observables["shared"]["fighter"].items():
				if(n==self.parent.getFullName()):#自分を除く
					continue
				if(idx>=self.maxTrackNum["Friend"]):#既に最大機数分記録した
					break
				friendMotion=MotionState(f["motion"])
				friendPos0[idx]=friendMotion.pos
				friendVel[idx]=friendMotion.vel
				idx+=1
			dt=self.manager.getAgentInterval()*self.manager.getBaseTimeStep()
			for t in range(max(self.pastPoints)):
				obs_m=np.zeros([self.pastObsDim],dtype=np.float32)
				ofs=0
				delta=(t+1)*dt
				#自機
				vel=myVel
				myPos=pos=myPos0-vel*delta
				V=np.linalg.norm(vel)
				obs_m[ofs+0:ofs+3]=pos*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
				#変更点。velは絶対値と単位ベクトルの入力だったのをベクトル生入力に。
				obs_m[ofs+3:ofs+6]=vel*self.xyInv/self.velocityNormalizer
				ofs+=6
				idx=0
				#味方機
				for n,f in self.parent.observables["shared"]["fighter"].items():
					if(n==self.parent.getFullName()):#自分を除く
						continue
					if(idx>=self.maxTrackNum["Friend"]):#既に最大機数分記録した
						break
					vel=friendVel[idx]
					pos=friendPos0[idx]-vel*delta
					dr=pos
					dv=vel
					R=np.linalg.norm(dr)
					V=np.linalg.norm(dv)
					#変更点。味方の位置と速度を単位ベクトル表記から変更
					#obs[ofs+0:ofs+3]=dr*self.xyInv
					#obs[ofs+3:ofs+6]=dv*self.xyInv
					#obs[ofs+6]=f["weapon"]["remMsls"]()
					ofs+=7
					idx+=1
				self.pastData[t]=obs_m
		bufferSize=max(self.pastPoints)
		idx=bufferSize-1-(count%bufferSize)
		totalObs=np.concatenate((current,np.concatenate([self.pastData[(idx+i)%bufferSize] for i in self.pastPoints])))
		self.pastData[idx]=memory
		#print("totalObs",totalObs)
		return totalObs
	def makeSingleObs(self):
		"""1フレーム分の観測データを生成する。
		"""
		obs=np.zeros([self.singleDim],dtype=np.float32)
		obsMemory=np.zeros([self.pastObsDim],dtype=np.float32)
		#自機
		myMotion=MotionState(self.parent.observables["motion"])
		pos=myMotion.pos
		vel=myMotion.vel
		V=np.linalg.norm(vel)
		quat=myMotion.q
		ofs=0
		ofs_m=0
		obs[ofs+0:ofs+3]=pos*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
		#変更点。velは絶対値と単位ベクトルの入力だったのをベクトル生入力に。
		obs[ofs+3:ofs+6]=vel*self.xyInv/self.velocityNormalizer
		obs[ofs+6:ofs+10]=np.array([quat.w,quat.x,quat.y,quat.z])*self.xyQuatInv
		obs[ofs+10]=self.parent.observables["weapon"]["remMsls"]()
		ofs+=11
		obsMemory[ofs_m+0:ofs_m+3]=pos*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
		obsMemory[ofs_m+3:ofs_m+6]=vel*self.xyInv/self.velocityNormalizer
		ofs_m+=6
		#味方機
		idx=0
		for n,f in self.parent.observables["shared"]["fighter"].items():
			if(n==self.parent.getFullName()):#自分を除く
				continue
			if(idx>=self.maxTrackNum["Friend"]):#既に最大機数分記録した
				break
			if(f["isAlive"]):#生存しているもののみ値を入れる
				fm=MotionState(f["motion"])
				dr=fm.pos
				dv=fm.vel
				R=np.linalg.norm(dr)
				V=np.linalg.norm(dv)
				quat=fm.q
				#変更点。味方の位置と速度を単位ベクトル表記から変更
				obs[ofs+0:ofs+3]=dr*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
				obs[ofs+3:ofs+6]=dv*self.xyInv/self.velocityNormalizer
				obs[ofs+6:ofs+10]=np.array([quat.w,quat.x,quat.y,quat.z])*self.xyQuatInv
				obs[ofs+10]=f["weapon"]["remMsls"]()
				ofs+=11
				idx+=1
		while(idx<self.maxTrackNum["Friend"]):#0埋め
			ofs+=11
			idx+=1
		#彼機(味方の誰かが探知しているものだけ諸元入り)
		#自分のtrackを近い方から順に読んで入れていく
		def distance(track):
			return np.linalg.norm(myMotion.pos-track.pos)
		self.lastTrackInfo=sorted([Track3D(t) for t in self.parent.observables["sensor"]["track"]],key=distance)
		idx=0
		#まずは観測時間を足してしまう(秒間0.01)
		for id in self.lostTime.keys():
			self.lostTime[id]+=0.01
		#各トラック情報を更新していく。
		for t in self.lastTrackInfo:
			if(idx>=self.maxTrackNum["Enemy"]):
				break
			idIndex = 0
			if t.truth in self.enemyIDs:
				idIndex = self.enemyIDs.index(t.truth)
			else:
				idIndex = len(self.enemyIDs)
				self.enemyIDs.append(t.truth)
			#最終観測情報の更新
			self.lostTime[t.truth]=0
			self.lostPos[t.truth]=t.pos
			self.lostVel[t.truth]=t.vel
				
			dr=t.posI()
			dv=t.velI()
			R=np.linalg.norm(dr)
			V=np.linalg.norm(dv)
			#変更点。単位ベクトル表示から通常表示に
			obs[ofs+idIndex*6+0:ofs+idIndex*6+3]=dr*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
			obs[ofs+idIndex*6+3:ofs+idIndex*6+6]=dv*self.xyInv/self.velocityNormalizer
			obsMemory[ofs_m+idIndex*6+0:ofs_m+idIndex*6+3]=dr*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
			obsMemory[ofs_m+idIndex*6+3:ofs_m+idIndex*6+6]=dv*self.xyInv/self.velocityNormalizer
		ofs+=6*self.maxTrackNum["Enemy"]
		ofs_m+=6*self.maxTrackNum["Enemy"]
		#味方誘導弾(射撃時刻が古いものから最大N発分)
		def launchedT(m):
			return m["launchedT"]() if m["isAlive"]() and m["hasLaunched"]() else np.inf
		msls=[m for m in self.parent.observables["weapon"]["missiles"]]
		for n,f in self.parent.observables["shared"]["fighter"].items():
			if(n!=self.parent.getFullName()):
				msls.extend(f["weapon"]["missiles"])
		msls=sorted(msls,key=launchedT)
		idx=0
		for m in msls:
			if(idx>=self.maxMissileNum["Friend"] or not (m["isAlive"]() and m["hasLaunched"]())):
				break
			mm=MotionState(m["motion"])
			dr=mm.pos
			dv=mm.vel
			R=np.linalg.norm(dr)
			V=np.linalg.norm(dv)
			#変更点。位置情報と速度情報を単位ベクトル表記から標準に。
			obs[ofs+0:ofs+3]=dr*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
			obs[ofs+3:ofs+6]=dv*self.xyInv/self.velocityNormalizer
			ofs+=6
			idx+=1
		while(idx<self.maxMissileNum["Friend"]):#0埋め
			ofs+=6
			idx+=1
		#彼誘導弾(MWSで探知したもののうち自身の正面に近いものから最大N発)
		def angle(track):
			return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
		mws=sorted([Track2D(t) for t in self.parent.observables["sensor"]["mws"]["track"]],key=angle)
		idx=0
		for m in mws:
			if(idx>=self.maxMissileNum["Enemy"]):
				break
			#変更点。ミサイルの角速度も考慮に入れる。
			obs[ofs+0:ofs+3]=myMotion.relPtoB(m.dir)
			obs[ofs+3:ofs+6]=myMotion.relPtoB(m.omega)
			ofs+=6
			idx+=1
		while(idx<self.maxMissileNum["Enemy"]):#0埋め
			ofs+=6
			idx+=1
		#メタデータの挿入。
		obs[ofs+0]=self.matchProcess
		ofs+=1
		#ロスト情報の挿入。
		idx=0
		for id in self.enemyIDs:
			if idx<self.maxTrackNum["Enemy"]:
				idx+=1
			else:
				break
			obs[ofs+0:ofs+3]=self.lostPos[id]*self.xyInv/np.array([max(self.dOut,self.dLine),max(self.dOut,self.dLine),self.hLim])
			obs[ofs+3:ofs+6]=self.lostVel[id]*self.xyInv/self.velocityNormalizer
			obs[ofs+6]=self.lostTime[id]
			ofs+=7
		while idx<self.maxTrackNum["Enemy"]:
			idx+=1
			ofs+=7
		return obs, obsMemory
	def action_space(self):
		self.turnTable=np.linspace(-self.turnScale,self.turnScale,self.turnDim)
		self.pitchTable=np.linspace(-self.pitchScale,self.pitchScale,self.pitchDim)
		self.accelTable=np.linspace(-self.accelScale,self.accelScale,self.accelDim)
		self.fireTable=list(range(-1,self.maxTrackNum["Enemy"]))
		if(self.turnDim%2!=0):self.turnTable[self.turnDim//2]=0.0#force center value strictly to be zero
		if(self.pitchDim%2!=0):self.pitchTable[self.pitchDim//2]=0.0#force center value strictly to be zero
		if(self.accelDim%2!=0):self.accelTable[self.accelDim//2]=0.0#force center value strictly to be zero
		nvec=np.array([len(self.turnTable),len(self.pitchTable),len(self.accelTable),len(self.fireTable)])
		return spaces.MultiDiscrete(nvec)
	#敵との実質距離を計算する計算式、肝要。
	def calcPseudoRelativeRange(self,mPos,mVel,ePos,eVel):
		toEnem=ePos-mPos
		directLength=np.linalg.norm(toEnem)
		toEnem_norm=toEnem/directLength
		mVel_norm=mVel/np.linalg.norm(mVel)
		relVel=mVel+mVel_norm*3*self.minimumV-eVel
		velEffect=np.dot(toEnem_norm,relVel)/self.minimumV
		climbLange=ePos[2]-mPos[2]
		pseudoLength=directLength-velEffect*6000-5*climbLange
		#print(self.getFullName(),velEffect,",",pseudoLength)
		return pseudoLength

	def deploy(self,action):
		myMotion=MotionState(self.parent.observables["motion"])
		pAZ=myMotion.az
		turn=self.turnTable[action[0]]
		pitch=self.pitchTable[action[1]]
		#print("pitchDist",pitch)
		self.dstDir=np.array([cos(pAZ+turn)*cos(pitch),sin(pAZ+turn)*cos(pitch),sin(pitch)])
		if(not(self.accelTable[self.lastAction[2]]==0.0 and self.accelTable[action[2]]==0.0)):
			self.baseV=np.linalg.norm(myMotion.vel)
		self.dstV=self.baseV+self.accelTable[action[2]]
		if(self.baseV<self.minimumV):
			self.velRecovery=True
		if(self.baseV>=self.minimumRecoveryV):
			self.velRecovery=False
		if(self.velRecovery):
			self.dstV=self.minimumRecoveryDstV

		

		#会敵前後の行動原理（優先度低め、有利になるための行動）
		if(self.matchProcess==0):
			self.dstDir=self.dstDir*self.xyInv
			#横45度以上の進行を許さない(直進に変換してしまう)
			if(abs(self.dstDir[0])>abs(self.dstDir[1])):
				self.dstDir[0]=0
			#下降を許さない
			if(self.dstDir[2]>0):
				self.dstDir[2]=0
			#目標速度が300を下回るのを許さない
			if(self.dstV<450):
				self.dstV=450
			
			#正規化の後に変換しなおし
			self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
			self.dstDir=self.dstDir*self.xyInv
		else:
			#lostPosから十分離れて前進していたらauto_pilot（前進）
			distance=1000000
			allChanged=True
			for fighterID in self.enemyIDs:
				ePos=self.lostPos[fighterID]
				lostT=self.lostTime[fighterID]
				relatPos=(myMotion.pos-ePos)*self.xyInv
				if(relatPos[1]<0 or abs(relatPos[1]<abs(relatPos[0]))):
					distance=min(distance,np.linalg.norm(relatPos))
				elif(lostT<0.3):
					allChanged=False
			if(distance>30000 and allChanged):
				#print("ap Ref")
				self.dstDir=np.array([0,-1,0])
				#正規化の後に変換しなおし
				self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
				self.dstDir=self.dstDir*self.xyInv
				self.dstV=600
				
			#100秒に一回発動できる航跡変換の術（見られていない場合は予測航跡をずらす）
			self.skillCoolTime+=1
			if(len(self.enemyIDs)>=self.maxTrackNum["Enemy"]):
				notFound=True
				for t in self.enemyIDs:
					vDot=np.dot(self.lostVel[t],myMotion.pos-self.lostPos[t])
					vDot_norm=vDot/(np.linalg.norm(self.lostVel[t])*np.linalg.norm(myMotion.pos-self.lostPos[t]))
					if(vDot_norm>-0.01 or self.lostTime[t]>0):
						notFound=False
						break
				if(notFound):
					if(self.skillCoolTime>99):
						self.skillCoolTime=0
						self.skillOriginalV=myMotion.vel
					if(self.skillCoolTime<6):#6秒間の強制操作
						if(myMotion.pos[2]>-5000):
							self.dstDir[2]=0
							self.dstDir[2]=-np.linalg.norm(self.dstDir)
							self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
						elif(myMotion.pos[2]<-19000):
							self.dstDir[2]=0
							self.dstDir[2]=np.linalg.norm(self.dstDir)
							self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
						else:
							if(self.skillOriginalV[2]<0):
								self.dstDir[2]=0
								self.dstDir[2]=np.linalg.norm(self.dstDir)
								self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
							else:
								self.dstDir[2]=0
								self.dstDir[2]=-np.linalg.norm(self.dstDir)
								self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
			
			#防衛線付近での防衛コントロール
			guardian=False
			closestY=0
			targetPos=np.array([0,0,0])
			targetVel=np.array([0,0,0])
			targetRelP=np.array([0,0,0])
			for fighterID in self.enemyIDs:
				ePos=self.lostPos[fighterID]
				eVel=self.lostVel[fighterID]
				lostT=self.lostTime[fighterID]
				relatPos=(myMotion.pos-ePos)*self.xyInv
				if(relatPos[1]<0 or abs(relatPos[1]<abs(relatPos[0])) and self.lostTime[fighterID]<0.1):
					if(((myMotion.pos+ePos)*self.xyInv/2)[1]>10000 and (ePos*self.xyInv)[1]>closestY and (myMotion.pos*self.xyInv)[1]>0):
						guardian=True
						closestY=(ePos*self.xyInv)[1]
						targetPos=ePos
						targetVel=eVel
						targetRelP=relatPos
			if(guardian):
				self.preGuardian=True
				destination=targetPos*self.xyInv+np.array([0,1,0])*np.linalg.norm(targetRelP)
				relatPos=myMotion.pos*self.xyInv-destination
				self.dstDir=-relatPos
				#正規化の後に変換しなおし
				self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
				self.dstDir=self.dstDir*self.xyInv
				self.dstV=600
			else:
				if(self.preGuardian):
					self.distRecovery=True
				self.preGuardian=False

			#中心から65km以上離れようとすることはできない。
			if(abs(myMotion.pos[0])>65000):
				if(myMotion.pos[0]*self.dstDir[0]>0):
					self.dstDir[0]=0
					self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)

		#基本戦略(優先度高い、決定的不利を背負わないための原則)
		self.dstDir=self.dstDir*self.xyInv
		#後退を許さない、回避成功時はしばらく直進で様子見
		normVel=myMotion.vel/np.linalg.norm(myMotion.vel)
		normVel=normVel*self.xyInv
		"""if(normVel[1]>0):
			self.distRecovery=True"""
		if(normVel[1]<-0.99):
			self.distRecovery=False
		if(self.distRecovery):
			if(self.breakTime>6):
				if(not self.preGuardian):
					self.dstDir=np.array([0,-1,0])
					self.dstV=600
			else:
				self.dstDir=myMotion.vel/np.linalg.norm(myMotion.vel)
				#縦方向慣性が大きければ航跡も変える。
				self.dstDir[2]=-self.dstDir[2]
				self.dstV=500
		#正規化の後に変換しなおし
		self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
		self.dstDir=self.dstDir*self.xyInv

		#変更点。ミサイルは一定の距離なら即発射という形。ポリシーで決まるのは発射閾値の距離。R1maxとR2Maxの想定で境界付する。最大値はR3Max
		#さらに、対象が複数いる場合は発射間隔が半分になり交互に打つようになる。
		shoot=int(action[3])
		flyingMsls=0
		for msl in self.parent.observables.at_p("/weapon/missiles"):
			if(msl.at("isAlive")() and msl.at("hasLaunched")()):
				flyingMsls+=1
		if len(self.lastTrackInfo) == 0 or self.shootTimer < 30/2:
			shoot=-1
			self.shootTimer+=1
		if(shoot>=0):
			#ここの計算式は再考の余地あり。
			minimumRange = 2000000
			tempTarget = Track3D()
			for track in self.lastTrackInfo:
				if(track.truth==self.target.truth and self.shootTimer < 30):
					continue
				pseudoRelPos = self.calcPseudoRelativeRange(myMotion.pos, myMotion.vel, track.pos, track.vel)
				if np.linalg.norm(pseudoRelPos) < minimumRange:
					minimumRange = np.linalg.norm(pseudoRelPos)
					tempTarget = track
			if(minimumRange < self.rMaxs[shoot]):
				self.launchFlag=True
				self.target=tempTarget
				#print(self.shootTimer)
				self.shootTimer=0
			else:
				self.launchFlag=False
				self.shootTimer+=1
		else:
			self.launchFlag=False
		self.observables[self.parent.getFullName()]["decision"]={
			"Roll":("Don't care"),
			"Horizontal":("Az_BODY",turn),
			"Vertical":("El",-pitch),
			"Throttle":("Vel",self.dstV),
			"Fire":(self.launchFlag,self.target.to_json())
		}
		self.lastAction=action[:]
	
	def control(self):
		"""高度と南北方向位置について可動範囲を設定し、逸脱する場合は強制的に復元
		"""
		myMotion=MotionState(self.parent.observables["motion"])
		pos=myMotion.pos
		vel=myMotion.vel
		#まずは反射処理
		if(len(self.parent.observables["sensor"]["mws"]["track"])>0 and (not self.missleAlert)):
			#ミサイル回避の反射
			missileTrack=Track2D(self.parent.observables["sensor"]["mws"]["track"][0])
			self.dstDir=-missileTrack.dir
			self.distRecovery=True
			self.breakTime=0
			self.warnTime+=0.1
			if(self.dstDir[2]<0 or vel[2]*(20000+pos[2])>3000000):
				self.dstDir[2]=0
				self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
			if(self.warnTime > 20 and np.dot(vel/np.linalg.norm(vel),missileTrack.dir)<0):
				self.dstDir[2]=-sqrt(pow(self.dstDir[0],2)+pow(self.dstDir[1],2))
				self.dstDir=self.dstDir/np.linalg.norm(self.dstDir)
			if(np.dot(vel/np.linalg.norm(vel),missileTrack.dir)>0.85):
				proDst=self.dstDir
				norVel=vel/np.linalg.norm(vel)
				self.dstDir=vel/np.linalg.norm(vel)-missileTrack.dir
				self.dstDir[2]=0
				self.dstDir=self.dstDir/(np.linalg.norm(self.dstDir)+1e-10)
				self.dstDir=self.dstDir*sqrt(pow(proDst[0],2)+pow(proDst[1],2))
				self.dstDir[2]=proDst[2]
				self.dstV=np.linalg.norm(vel)-30
		else:
			self.breakTime+=0.1
			self.warnTime=0
		invPos=pos*self.xyInv
		if(invPos[1]>70000 or invPos[1]<-97500):
			#境界近くでは前進を強制
			self.dstDir=np.array([0,-1,0])*self.xyInv
		#境界調整
		if(abs(pos[0])>=self.dOutLimitRatio*self.dOut):
			#戦域逸脱を避けるための方位補正
			#判定ラインの超過具合に応じて復帰角度を変化させる。(無限遠でラインに直交、ライン上でラインと平行)
			over=abs(pos[0])/self.dOut-self.dOutLimitRatio
			n=sqrt(self.dstDir[0]*self.dstDir[0]+self.dstDir[1]*self.dstDir[1])
			theta=atan(over)
			cs=cos(theta)
			sn=sin(theta)
			xyMagni=sqrt(pow(self.dstDir[0],2)+pow(self.dstDir[1],2))
			if(pos[0]>0):#北側
				if(self.dstDir[1]>0):#東向き
					self.dstDir=np.array([-xyMagni/sqrt(2),xyMagni/sqrt(2),self.dstDir[2]])
				else:#西向き
					self.dstDir=np.array([-xyMagni/sqrt(2),-xyMagni/sqrt(2),self.dstDir[2]])
			else:#南側
				if(self.dstDir[1]>0):#東向き
					self.dstDir=np.array([xyMagni/sqrt(2),xyMagni/sqrt(2),self.dstDir[2]])
				else:#西向き
					self.dstDir=np.array([xyMagni/sqrt(2),-xyMagni/sqrt(2),self.dstDir[2]])
		if(-pos[2]<self.hMin):
			#高度下限を下回った場合
			over=self.hMin+pos[2]
			n=sqrt(self.dstDir[0]*self.dstDir[0]+self.dstDir[1]*self.dstDir[1])
			theta=atan(over)
			cs=cos(theta)
			sn=sin(theta)
			self.dstDir=np.array([self.dstDir[0]/n*cs,self.dstDir[1]/n*cs,-sn])
		elif(-pos[2]>self.hMax):
			#高度上限を上回った場合
			over=-pos[2]-self.hMax
			n=sqrt(self.dstDir[0]*self.dstDir[0]+self.dstDir[1]*self.dstDir[1])
			theta=atan(over)
			cs=cos(theta)
			sn=sin(theta)
			self.dstDir=np.array([self.dstDir[0]/n*cs,self.dstDir[1]/n*cs,sn])
		if(self.parent.isinstance(CoordinatedFighter)):
			self.commands[self.parent.getFullName()]={
				"motion":{
					"dstDir":self.dstDir,
					"dstV":self.dstV
				},
				"weapon":{
					"launch":self.launchFlag,
					"target":self.target.to_json()
				}
			}
	#ゲーム進行度の管理をする。
	def perceive(self, b_arg):
		myMotion=MotionState(self.parent.observables["motion"])
		if "sensor" in self.parent.observables.keys():
			def distance(track):
				return np.linalg.norm(myMotion.pos-track.pos)
			tracks=sorted([Track3D(t) for t in self.parent.observables["sensor"]["track"]],key=distance)
		else:
			tracks=[]

		if len(tracks)>0 and self.matchProcess==0:
			self.matchProcess+=1
		for pTrack in self.preTracks:
			notFound=True
			for track in tracks:
				if pTrack.truth==track.truth:
					notFound=False
			if notFound:
				self.matchProcess+=1
				#print("missle hit confirmed")
		#次回のためにpreTrackの生成。
		tempTracks=[]
		for track in tracks:
			msls=[]
			for n,f in self.parent.observables["shared"]["fighter"].items():
				msls.extend(f["weapon"]["missiles"])
			launched_m=[]
			for m in msls:
				if m["isAlive"]() and m["hasLaunched"]():
					launched_m.append(m)
			for l_m in launched_m:
				t_Pos=track.pos
				m_Pos=MotionState(l_m["motion"]).pos
				m_distance=np.linalg.norm(t_Pos-m_Pos)
				if m_distance<700:
					tempTracks.append(track)
					break
			
		self.preTracks=tempTracks

#importした時に登録はしておく(Configから呼び出すために必要)
addPythonClass('Agent', 'BabelAgent', BabelAgent)
Factory.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/BabelAgentConfig.json"))
