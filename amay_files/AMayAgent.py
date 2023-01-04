# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
from gym import spaces
import numpy as np
import sys
from utils import *
from ASRCAISim1.libCore import *

class AMayAgent(Agent):
	"""編隊全体で1つのAgentを割り当てる、中央集権方式での行動判断モデルの実装例。
	モデルの内容はほぼR4PyAgentSample01Sに準拠しているが、
	* observationの自機と味方機を区別していた部分を「自陣営」として一纏めに
	* imageの描画座標系を自陣営座標系に限定して回転を無効化
	* vectorの彼機航跡に対する射程情報を味方全機分格納
	* actionを全機分の行動とする
	の変更を加えている。
	"""
	class InstantInfo:
		#軌跡描画用のフレームデータ
		def __init__(self):
			self.friend_pos=[]
			self.friend_msl_pos=[]
			self.enemy_pos=[]
	class ActionInfo:
		#機体に対するコマンドを生成するための変数をまとめた構造体
		def __init__(self):
			self.roll = 0
			self.pitch = 0
			self.yaw = 0
			self.throttle = 1
			self.use_fromDirandVel = False #操作形式をどうするか
			self.dstDir=np.array([1.0,0.0,0.0]) #目標進行方向
			self.dstAlt=10000.0 #目標高度
			self.velRecovery=False #下限速度制限からの回復中かどうか
			self.asThrottle=True #加減速についてスロットルでコマンドを生成するかどうか
			self.keepVel=False #加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
			self.dstThrottle=1.0 #目標スロットル
			self.dstV=300 #目標速度
			self.launchFlag=False #射撃するかどうか
			self.target=Track3D() #射撃対象
			self.lastShotTimes={} #各Trackに対する直前の射撃時刻
	class TeamOrigin:
		#陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス。
		#MotionStateを使用しても良いがクォータニオンを経由することで浮動小数点演算に起因する余分な誤差が生じるため、もし可能な限り対称性を求めるのであればこの例のように符号反転で済ませたほうが良い。
		#ただし、機体運動等も含めると全ての状態量に対して厳密に対称なシミュレーションとはならないため、ある程度の誤差は生じる。
		def __init__(self,isEastSider_,dLine):
			self.isEastSider=isEastSider_
			if(self.isEastSider):
				self.pos=np.array([0.,dLine,0.])
			else:
				self.pos=np.array([0.,-dLine,0.])
		def relBtoP(self,v):
			#陣営座標系⇛慣性座標系
			if(self.isEastSider):
				return np.array([v[1],-v[0],v[2]])
			else:
				return np.array([-v[1],v[0],v[2]])
		def relPtoB(self,v):
			#慣性座標系⇛陣営座標系
			if(self.isEastSider):
				return np.array([-v[1],v[0],v[2]])
			else:
				return np.array([v[1],-v[0],v[2]])
	def __init__(self,modelConfig,instanceConfig):
		super().__init__(modelConfig,instanceConfig)
		if(self.isDummy):
			return
		#modelConfigの読み込み
		#observation spaceの設定
		self.maxTrackNum=getValueFromJsonKRD(self.modelConfig,"maxTrackNum",self.randomGen,{"Friend":4,"Enemy":4})
		assert(self.maxTrackNum["Friend"]>=len(self.parents))
		self.maxMissileNum=getValueFromJsonKRD(self.modelConfig,"maxMissileNum",self.randomGen,{"Friend":8,"Enemy":1})
		self.horizontalNormalizer=getValueFromJsonKRD(self.modelConfig,"horizontalNormalizer",self.randomGen,100000.0)
		self.verticalNormalizer=getValueFromJsonKRD(self.modelConfig,"verticalNormalizer",self.randomGen,20000.0)
		self.fgtrVelNormalizer=getValueFromJsonKRD(self.modelConfig,"fgtrVelNormalizer",self.randomGen,300.0)
		self.mslVelNormalizer=getValueFromJsonKRD(self.modelConfig,"mslVelNormalizer",self.randomGen,1200.0)
		self.use_image_observation=getValueFromJsonKRD(self.modelConfig,"use_image_observation",self.randomGen,False)
		self.use_vector_observation=getValueFromJsonKRD(self.modelConfig,"use_vector_observation",self.randomGen,True)
		assert(self.use_image_observation or self.use_vector_observation)
		#  2次元画像として
		if(self.use_image_observation):
			self.image_longitudinal_resolution=getValueFromJsonKRD(self.modelConfig,"image_longitudinal_resolution",self.randomGen,32)
			self.image_lateral_resolution=getValueFromJsonKRD(self.modelConfig,"image_lateral_resolution",self.randomGen,32)
			self.image_front_range=getValueFromJsonKRD(self.modelConfig,"image_front_range",self.randomGen,200000.0)
			self.image_back_range=getValueFromJsonKRD(self.modelConfig,"image_back_range",self.randomGen,0.0)
			self.image_side_range=getValueFromJsonKRD(self.modelConfig,"image_side_range",self.randomGen,100000.0)
			self.image_horizon=getValueFromJsonKRD(self.modelConfig,"image_horizon",self.randomGen,120)
			self.image_interval=getValueFromJsonKRD(self.modelConfig,"image_interval",self.randomGen,1)
		#  実数値ベクトルとして
		if(self.use_vector_observation):
			self.include_last_action=getValueFromJsonKRD(self.modelConfig,"include_last_action",self.randomGen,True)
			self.vector_past_points=sorted(getValueFromJsonKRD(self.modelConfig,"vector_past_points",self.randomGen,[10, 30, 60, 120]))
		# action spaceの設定
		self.flatten_action_space=getValueFromJsonKRD(self.modelConfig,"flatten_action_space",self.randomGen,False)
		# 左右旋回に関する設定
		self.dstAz_relative=getValueFromJsonKRD(self.modelConfig,"dstAz_relative",self.randomGen,True)
		self.turnTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"turnTable",self.randomGen,
			[-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0])),dtype=np.float64)
		self.turnTable*=deg2rad(1.0)
		self.use_override_evasion=getValueFromJsonKRD(self.modelConfig,"use_override_evasion",self.randomGen,True)
		if(self.use_override_evasion):
			self.evasion_turnTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"evasion_turnTable",self.randomGen,
				[-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0])),dtype=np.float64)
			self.evasion_turnTable*=deg2rad(1.0)
			assert(len(self.turnTable)==len(self.evasion_turnTable))
		else:
			self.evasion_turnTable=self.turnTable
		# 上昇・下降に関する設定
		self.use_altitude_command=getValueFromJsonKRD(self.modelConfig,"use_altitude_command",self.randomGen,False)
		if(self.use_altitude_command):
			self.altTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"altTable",self.randomGen,
				[-8000.0,-4000.0,-2000.0,-1000.0,0.0,1000.0,2000.0,4000.0,8000.0])),dtype=np.float64)
			self.refAltInterval=getValueFromJsonKRD(self.modelConfig,"refAltInterval",self.randomGen,1000.0)
		else:
			self.pitchTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"pitchTable",self.randomGen,
				[-45.0,-20.0,-10.0,-5.0,0.0,5.0,10.0,20.0,45.0])),dtype=np.float64)
			self.pitchTable*=deg2rad(1.0)
			self.refAltInterval=1.0
		# 加減速に関する設定
		self.accelTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"accelTable",self.randomGen,[-20.0,0.0,20.0])),dtype=np.float64)
		self.always_maxAB=getValueFromJsonKRD(self.modelConfig,"always_maxAB",self.randomGen,False)
		# 射撃に関する設定
		self.use_Rmax_fire=getValueFromJsonKRD(self.modelConfig,"use_Rmax_fire",self.randomGen,True)
		self.rMaxs=getValueFromJsonKRD(self.modelConfig,"rMaxs",self.randomGen,[50000, 25000, 0])
		if(self.use_Rmax_fire):
			self.shotIntervalTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"shotIntervalTable",self.randomGen,
				[5.0,10.0,20.0,40.0,80.0])),dtype=np.float64)
			self.shotThresholdTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"shotThresholdTable",self.randomGen,
				[0.0,0.25,0.5,0.75,1.0])),dtype=np.float64)
		#行動制限に関する設定
		#  高度制限に関する設定
		self.altMin=getValueFromJsonKRD(self.modelConfig,"altMin",self.randomGen,2000.0)
		self.altMax=getValueFromJsonKRD(self.modelConfig,"altMax",self.randomGen,15000.0)
		self.altitudeKeeper=AltitudeKeeper(modelConfig().get("altitudeKeeper",{}))
		# 場外制限に関する設定
		self.dOutLimitRatio=getValueFromJsonKRD(self.modelConfig,"dOutLimitRatio",self.randomGen,0.9)
		self.dOutLimitScale=getValueFromJsonKRD(self.modelConfig,"dOutLimitScale",self.randomGen,50.0)
		# 同時射撃数の制限に関する設定
		self.maxSimulShot=getValueFromJsonKRD(self.modelConfig,"maxSimulShot",self.randomGen,4)
		# 下限速度の制限に関する設定
		self.minimumV=getValueFromJsonKRD(self.modelConfig,"minimumV",self.randomGen,150.0)
		self.minimumRecoveryV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryV",self.randomGen,180.0)
		self.minimumRecoveryDstV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryDstV",self.randomGen,200.0)

		#内部変数の初期化
		#  observationに関するもの
		floatLow=-sys.float_info.max
		floatHigh=sys.float_info.max
		#	2次元画像として
		if(self.use_image_observation):
			self.numChannels=6
			#画像データバッファの生成
			self.image_buffer=np.zeros([self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],dtype=np.float32)
			self.image_buffer_coords=np.zeros([2,self.image_longitudinal_resolution,self.image_lateral_resolution])
			for lon in range(self.image_longitudinal_resolution):
				for lat in range(self.image_lateral_resolution):
					self.image_buffer_coords[0,lon,lat]=(self.image_front_range+self.image_back_range)*(0.5+lat)/self.image_longitudinal_resolution-self.image_back_range
					self.image_buffer_coords[1,lon,lat]=2.0*self.image_side_range*(0.5+lon)/self.image_lateral_resolution-self.image_side_range
			#軌跡描画用の過去データバッファの生成
			self.image_past_data=[self.InstantInfo() for i in range(self.image_horizon)]
			#spaceの生成
			image_space=spaces.Box(floatLow,floatHigh,
				shape=[self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],
				dtype=np.float32)
		#	実数値ベクトルとして
		self.last_action_dim=3+(1+self.maxTrackNum["Enemy"])
		if(self.use_vector_observation):
			#次元の計算
			self.friend_dim=3+4+2+3+1+3*self.maxMissileNum["Enemy"]
			self.enemy_dim=3+4+4+3*self.maxTrackNum["Friend"]
			self.friend_msl_dim=3+4+1+3+1
			#データバッファの生成
			self.friend_temporal=np.zeros([self.maxTrackNum["Friend"],self.friend_dim],dtype=np.float32)
			self.enemy_temporal=np.zeros([self.maxTrackNum["Enemy"],self.enemy_dim],dtype=np.float32)
			self.friend_msl_temporal=np.zeros([self.maxMissileNum["Friend"],self.friend_msl_dim],dtype=np.float32)
			self.last_action_obs=np.zeros([self.maxTrackNum["Friend"],self.last_action_dim],dtype=np.float32)
			self.vector_single_dim=(
				(self.maxTrackNum["Friend"]*self.last_action_dim if self.include_last_action else 0)+
				self.maxTrackNum["Friend"]*self.friend_dim+
				self.maxTrackNum["Enemy"]*self.enemy_dim+
				self.maxMissileNum["Friend"]*self.friend_msl_dim
			)
			#過去データバッファ及びspaceの生成
			if(len(self.vector_past_points)>0):
				self.vector_past_data=[np.zeros([self.vector_single_dim],dtype=np.float32)
					for i in range(self.vector_past_points[-1])]
				vector_space=spaces.Box(floatLow,floatHigh,
					shape=[self.vector_single_dim*(1+len(self.vector_past_points))],
					dtype=np.float32)
			else:
				vector_space=spaces.Box(floatLow,floatHigh,
					shape=[self.vector_single_dim],
					dtype=np.float32)
		#spaceの確定
		if(self.use_image_observation and self.use_vector_observation):
			self._observation_space=spaces.Dict({
				"image":image_space,
				"vector":vector_space})
		elif(self.use_image_observation):
			self._observation_space=image_space
		elif(self.use_vector_observation):
			self._observation_space=vector_space
		else:
			raise ValueError("Either use_image_observation or use_vector_observation must be true.")

		#  actionに関するもの
		nvec=[]
		for pIdx,parent in enumerate(self.parents.values()):
			nvec.append(len(self.turnTable))
			nvec.append(len(self.altTable) if self.use_altitude_command else len(self.pitchTable))
			if(not self.always_maxAB):
				nvec.append(len(self.accelTable))
			nvec.append(self.maxTrackNum["Enemy"]+1)
			if(self.use_Rmax_fire):
				if(len(self.shotIntervalTable)>1):
					nvec.append(len(self.shotIntervalTable))
				if(len(self.shotThresholdTable)>1):
					nvec.append(len(self.shotThresholdTable))
		self.totalActionDim=1
		self.actionDims=np.zeros([len(nvec)])
		for i in range(len(nvec)):
			self.totalActionDim*=nvec[i]
			self.actionDims[i]=nvec[i]
		if(self.flatten_action_space):
			self._action_space=spaces.Discrete(self.totalActionDim)
		else:
			self._action_space=spaces.MultiDiscrete(nvec)
		self.actionInfos=[self.ActionInfo() for _ in self.parents]
	def validate(self):
		#Rulerに関する情報の取得
		rulerObs=self.manager.getRuler()().observables()
		self.dOut=rulerObs["dOut"]
		self.dLine=rulerObs["dLine"]
		eastSider=rulerObs["eastSider"]
		self.xyInv=np.array([1,1,1]) if (self.getTeam()==eastSider) else np.array([-1,-1,1])
		self.xyQuatInv=np.array([1,1,1,1]) if (self.getTeam()==eastSider) else np.array([1,-1,-1,1])
		self.matchProcess = 0
		self.shot_interval = 0
		self.teamOrigin=self.TeamOrigin(self.getTeam()==eastSider,self.dLine)
		for pIdx,parent in enumerate(self.parents.values()):
			if(parent.isinstance(CoordinatedFighter) or parent.isinstance(SixDoFFighter)):
				#飛行制御は全てデフォルトをdirectにしておく
				parent.setFlightControllerMode("direct")
			else:
				raise ValueError("R4PyAgentSample01M accepts only SixDoFFighter or CoordinatedFighter for its parent.")
			actionInfo=self.actionInfos[pIdx]
			actionInfo.dstDir=np.array([0., -1. if self.getTeam()==eastSider else +1., 0.])
			self.last_action_obs[pIdx,0]=atan2(actionInfo.dstDir[1],actionInfo.dstDir[0])
			actionInfo.lastShotTimes={}
	def observation_space(self):
		return self._observation_space
	def makeObs(self):
		#observablesの収集
		self.extractObservables()

		#observationの生成
		if(self.use_image_observation):
			self.makeImageObservation()
		if(self.use_vector_observation):
			if(len(self.vector_past_points)>0):
				#過去のobsをまとめる
				vector_obs=np.zeros([self.vector_single_dim*(1+len(self.vector_past_points))],dtype=np.float32)
				#現在のobs
				current=self.makeVectorObservation()
				#過去のobsの読み込み(リングバッファ)
				stepCount=round(self.manager.getTickCount()/self.manager.getAgentInterval())
				bufferSize=len(self.vector_past_data)
				bufferIdx=bufferSize-1-(stepCount%bufferSize)
				ofs = 0
				vector_obs[ofs+0:ofs+0+self.vector_single_dim]=current
				ofs+=self.vector_single_dim
				for frame in range(len(self.vector_past_points)):
					vector_obs[ofs+0:ofs+0+self.vector_single_dim]=self.vector_past_data[(bufferIdx+self.vector_past_points[frame])%bufferSize]
					ofs+=self.vector_single_dim
				#最も古いものを現在のobsに置換
				self.vector_past_data[bufferIdx]=current
			else:
				#現在のobsのみ
				vector_obs=self.makeVectorObservation()
		if(self.use_image_observation and self.use_vector_observation):
			return {
				"image":self.image_buffer,
				"vector":vector_obs
			}
		elif(self.use_image_observation):
			return self.image_buffer
		elif(self.use_vector_observation):
			return vector_obs
		else:
			raise ValueError("Either use_image_observation or use_vector_observation must be true.")
	def extractObservables(self):
		#味方機(自機含む)
		self.ourMotion=[]
		self.ourObservables=[]
		for pIdx,parent in enumerate(self.parents.values()):
			if(parent.isAlive()):
				firstAlive=parent
				break
		for pIdx,parent in enumerate(self.parents.values()):
			if(parent.isAlive()):
				self.ourMotion.append(MotionState(parent.observables["motion"]))
				#残存していればobservablesそのもの
				self.ourObservables.append(parent.observables)
			else:
				self.ourMotion.append(MotionState())
				#被撃墜or墜落済なら本体の更新は止まっているので残存している親が代理更新したものを取得(誘導弾情報のため)
				self.ourObservables.append(
					firstAlive.observables.at_p("/shared/fighter").at(parent.getFullName()))
		#彼機(味方の誰かが探知しているもののみ)
		#観測されている航跡を、自陣営の機体に近いものから順にソートしてlastTrackInfoに格納する。
		#lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
		def distance(track):
			ret=-1.0
			for pIdx,parent in enumerate(self.parents.values()):
				if(parent.isAlive()):
					myMotion=MotionState(parent.observables["motion"])
					tmp=np.linalg.norm(track.posI()-myMotion.pos)
					if(ret<0 or tmp<ret):
						ret=tmp
			return ret
		for pIdx,parent in enumerate(self.parents.values()):
			if(parent.isAlive()):
				self.lastTrackInfo=sorted([Track3D(t) for t in parent.observables.at_p("/sensor/track")],key=distance)
				break
		#味方誘導弾(射撃時刻が古いものから最大N発分)
		#味方の誘導弾を射撃時刻の古い順にソート
		def launchedT(m):
			return m["launchedT"]() if m["isAlive"]() and m["hasLaunched"]() else np.inf
		self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)
	def calcDistanceMargin(self,myMotion, myObservables):
		#正規化した余剰燃料(距離換算)の計算
		optCruiseFuelFlowRatePerDistance=myObservables.at_p("/spec/propulsion/optCruiseFuelFlowRatePerDistance")()
		fuelRemaining=myObservables.at_p("/propulsion/fuelRemaining")()
		maxReachableRange=np.inf
		if(optCruiseFuelFlowRatePerDistance>0.0):
			maxReachableRange=fuelRemaining/optCruiseFuelFlowRatePerDistance
		rulerObs=self.manager.getRuler()().observables()
		distanceFromLine=np.dot(np.array(rulerObs["forwardAx"][self.getTeam()]),myMotion.pos[0:2])+self.dLine
		distanceFromBase=rulerObs["distanceFromBase"][self.getTeam()]
		fuelMargin=rulerObs["fuelMargin"]
		return max(-1.0,min(1.0,(maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase))/(2*self.dLine)))
	def makeImageObservation(self):
		self.image_buffer=np.zeros([self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],dtype=np.float32)
		#現在の情報を収集
		current=self.InstantInfo()
		#味方機(自機含む)の諸元
		for fIdx in range(len(self.ourMotion)):
			if(self.ourObservables[fIdx]["isAlive"]()):
				current.friend_pos.append(self.ourMotion[fIdx].pos)
		#彼機(味方の誰かが探知しているもののみ)
		for tIdx,t in enumerate(self.lastTrackInfo):
			current.enemy_pos.append(t.posI())
		#味方誘導弾(射撃時刻が古いものから最大N発分)
		for mIdx,m in enumerate(self.msls):
			if(not (m["isAlive"]() and m["hasLaunched"]())):
				break
			mm=MotionState(m["motion"])
			current.friend_msl_pos.append(mm.pos)

		#画像の生成
		#ch0:味方軌跡
		#ch1:彼機軌跡
		#ch2:味方誘導弾軌跡
		#ch3:味方覆域
		#ch4:彼我防衛ライン
		#ch5:場外ライン
		stepCount=round(self.manager.getTickCount()/self.manager.getAgentInterval())
		bufferIdx=self.image_horizon-1-(stepCount%self.image_horizon)
		self.image_past_data[bufferIdx]=current
		tMax=min(
			floor(stepCount/self.image_interval)*self.image_interval,
			floor((self.image_horizon-1)/self.image_interval)*self.image_interval
		)
		for t in range(tMax,-1,-self.image_interval):
			frame=self.image_past_data[(bufferIdx+t)%self.image_horizon]
			ch=0
			#ch0:味方
			for i in range(len(frame.friend_pos)):
				self.plotPoint(
					frame.friend_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)
			ch+=1
			#ch1:彼機
			for i in range(len(frame.enemy_pos)):
				self.plotPoint(
					frame.enemy_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)
			ch+=1
			#ch2:味方誘導弾
			for i in range(1,len(frame.friend_msl_pos)):
				self.plotPoint(
					frame.friend_msl_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)

		#ch3:味方レーダ覆域
		for i in range(len(self.ourMotion)):
			if(self.ourObservables[i]["isAlive"]()):
				ex=self.ourMotion[i].relBtoP(np.array([1,0,0]))
				Lref=self.ourObservables[i].at_p("/spec/sensor/radar/Lref")()
				thetaFOR=self.ourObservables[i].at_p("/spec/sensor/radar/thetaFOR")()
				for lon in range(self.image_longitudinal_resolution):
					for lat in range(self.image_lateral_resolution):
						pos=self.gridToPos(lon,lat,10000.0)
						rPos=pos-self.ourMotion[i].pos
						L=np.linalg.norm(rPos)
						if(L<=Lref and np.dot(rPos,ex)>=L*cos(thetaFOR)):
							self.plotPoint(
								pos,
								3,
								1.0
							)

		#ch4:彼我防衛ライン
		#彼側防衛ライン
		eps=1e-10 #戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
			4,
			1.0
		)
		#我側防衛ライン
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
			4,
			-1.0
		)
		#ch5:場外ライン
		#進行方向右側
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
			5,
			1.0
		)
		#進行方向左側
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
			5,
			1.0
		)
	def makeVectorObservation(self):
		#味方機(自機含む)の諸元
		self.friend_temporal=np.zeros([self.maxTrackNum["Friend"],self.friend_dim],dtype=np.float32)
		for fIdx in range(len(self.ourMotion)):
			if(fIdx>=self.maxTrackNum["Friend"]):
				break
			if(self.ourObservables[fIdx]["isAlive"]()):
				fMotion=self.ourMotion[fIdx]
				pos=self.teamOrigin.relPtoB(fMotion.pos) #慣性座標系→陣営座標系に変換
				vel=self.teamOrigin.relPtoB(fMotion.vel) #慣性座標系→陣営座標系に変換
				V=np.linalg.norm(vel)
				#位置
				p=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
				self.friend_temporal[fIdx,0:0+3]=p
				#速度
				self.friend_temporal[fIdx,3]=V/self.fgtrVelNormalizer
				self.friend_temporal[fIdx,4:4+3]=vel/V
				#初期弾数
				self.friend_temporal[fIdx,7]=self.ourObservables[fIdx].at_p("/spec/weapon/numMsls")()
				#残弾数
				self.friend_temporal[fIdx,8]=self.ourObservables[fIdx].at_p("/weapon/remMsls")()
				#姿勢(バンク、α、β)
				vb=fMotion.relPtoB(fMotion.vel)
				ex=fMotion.relBtoP(np.array([1.,0.,0.]))
				ey=fMotion.relBtoP(np.array([0.,1.,0.]))
				horizontalY=np.cross(np.array([0.,0.,1.]),ex)
				roll=0
				N=np.linalg.norm(horizontalY)
				if(N>0):
					horizontalY/=N
					sinRoll=np.dot(np.cross(horizontalY,ey),ex)
					cosRoll=np.dot(horizontalY,ey)
					roll=atan2(sinRoll,cosRoll)
				alpha=atan2(vb[2],vb[0])
				beta=asin(vb[1]/V)
				self.friend_temporal[fIdx,9]=roll
				self.friend_temporal[fIdx,10]=alpha
				self.friend_temporal[fIdx,11]=beta
				#余剰燃料
				self.friend_temporal[fIdx,12]=self.calcDistanceMargin(fMotion,self.ourObservables[fIdx])
				#MWS検出情報
				def angle(track):
					return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
				mws=sorted([Track2D(t) for t in self.ourObservables[fIdx].at_p("/sensor/mws/track")],key=angle)
				for mIdx,m in enumerate(mws):
					if(mIdx>=self.maxMissileNum["Enemy"]):
						break
					mDir=self.teamOrigin.relPtoB(m.dirI()) #慣性座標系→陣営座標系に変換
					self.friend_temporal[fIdx,13+mIdx*3:13+mIdx*3+3]=mDir
		#彼機(味方の誰かが探知しているもののみ)
		self.enemy_temporal=np.zeros([self.maxTrackNum["Enemy"],self.enemy_dim],dtype=np.float32)
		for tIdx,t in enumerate(self.lastTrackInfo):
			if(tIdx>=self.maxTrackNum["Enemy"]):
				break
			pos=self.teamOrigin.relPtoB(t.posI()) #慣性座標系→陣営座標系に変換
			vel=self.teamOrigin.relPtoB(t.velI()) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			#位置
			p=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			self.enemy_temporal[tIdx,0:0+3]=p
			#速度
			self.enemy_temporal[tIdx,3]=V/self.fgtrVelNormalizer
			self.enemy_temporal[tIdx,4:4+3]=vel/V
			#この航跡に対し飛翔中で最も近い味方誘導弾の情報(距離、誘導状態)
			#味方誘導弾の諸元格納時に格納
			self.enemy_temporal[tIdx,7]=0#距離
			self.enemy_temporal[tIdx,8:8+3]=np.array([0,0,0])#誘導状態
			#操作可能な味方機からこの航跡への射程情報
			for pIdx,parent in enumerate(self.parents.values()):
				if(parent.isAlive()):
					myMotion=MotionState(parent.observables["motion"])
					self.enemy_temporal[tIdx,pIdx*3+11]=self.calcRHead(parent,myMotion,t)/self.horizontalNormalizer#RHead
					self.enemy_temporal[tIdx,pIdx*3+12]=self.calcRTail(parent,myMotion,t)/self.horizontalNormalizer#RTail
					self.enemy_temporal[tIdx,pIdx*3+13]=self.calcRNorm(parent,myMotion,t)#現在の距離をRTail〜RHeadで正規化したもの
		#味方誘導弾(射撃時刻が古いものから最大N発分)
		self.friend_msl_temporal=np.zeros([self.maxMissileNum["Friend"],self.friend_msl_dim],dtype=np.float32)
		for mIdx,m in enumerate(self.msls):
			if(mIdx>=self.maxMissileNum["Friend"] or not (m["isAlive"]() and m["hasLaunched"]())):
				break
			mm=MotionState(m["motion"])
			pos=self.teamOrigin.relPtoB(mm.pos) #慣性座標系→陣営座標系に変換
			vel=self.teamOrigin.relPtoB(mm.vel) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			#位置
			p=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			self.friend_msl_temporal[mIdx,0:0+3]=p
			#速度
			self.friend_msl_temporal[mIdx,3]=V/self.mslVelNormalizer
			self.friend_msl_temporal[mIdx,4:4+3]=vel/V
			#目標情報(距離、誘導状態、対応する彼機航跡ID)
			mTgt=Track3D(m["target"]).extrapolateTo(self.manager.getTime())
			self.friend_msl_temporal[mIdx,7]=1.0-min(1.0,np.linalg.norm(mTgt.posI()-mm.pos)/self.horizontalNormalizer)
			mode=m["mode"]()
			if(mode==Missile.Mode.GUIDED.name):
				self.friend_msl_temporal[mIdx,8:8+3]=np.array([1,0,0])
			elif(mode==Missile.Mode.SELF.name):
				self.friend_msl_temporal[mIdx,8:8+3]=np.array([0,1,0])
			else:#if(mode==Missile.Mode.MEMORY.name):
				self.friend_msl_temporal[mIdx,8:8+3]=np.array([0,0,1])
			self.friend_msl_temporal[mIdx,11]=-1
			for tIdx,t in enumerate(self.lastTrackInfo):
				if(tIdx>=self.maxTrackNum["Enemy"]):
					break
				if(t.isSame(mTgt)):
					R=np.linalg.norm(t.posI()-mm.pos)
					if(R<self.horizontalNormalizer and self.enemy_temporal[tIdx,7]<1-R/self.horizontalNormalizer):
						self.enemy_temporal[tIdx,7]=1-R/self.horizontalNormalizer
						if(mode==Missile.Mode.GUIDED.name):
							self.enemy_temporal[tIdx,8:8+3]=np.array([1,0,0])
						elif(mode==Missile.Mode.SELF.name):
							self.enemy_temporal[tIdx,8:8+3]=np.array([0,1,0])
						else:#if(mode==Missile.Mode.MEMORY.name):
							self.enemy_temporal[tIdx,8:8+3]=np.array([0,0,1])
					self.friend_msl_temporal[mIdx,11]=tIdx
		#observationの生成
		ret=np.zeros([
			(self.maxTrackNum["Friend"]*self.last_action_dim if self.include_last_action else 0)+
			self.maxTrackNum["Friend"]*self.friend_dim+
			self.maxTrackNum["Enemy"]*self.enemy_dim+
			self.maxMissileNum["Friend"]*self.friend_msl_dim
		],dtype=np.float32)
		ofs=0
		if(self.include_last_action):
			d=self.maxTrackNum["Friend"]*self.last_action_dim
			ret[ofs:ofs+d]=np.reshape(self.last_action_obs,[-1])
			ofs+=d
		d=self.maxTrackNum["Friend"]*self.friend_dim
		ret[ofs:ofs+d]=np.reshape(self.friend_temporal,[-1])
		ofs+=d
		d=self.maxTrackNum["Enemy"]*self.enemy_dim
		ret[ofs:ofs+d]=np.reshape(self.enemy_temporal,[-1])
		ofs+=d
		d=self.maxMissileNum["Friend"]*self.friend_msl_dim
		ret[ofs:ofs+d]=np.reshape(self.friend_msl_temporal,[-1])
		return ret
	def action_space(self):
		return self._action_space
	def deploy(self,action):
		self.last_action_obs=np.zeros([self.maxTrackNum["Friend"],self.last_action_dim],dtype=np.float32)
		if(self.flatten_action_space):
			unflattened_action=self.unflatten_action(action,self.actionDims)
		else:
			unflattened_action=action
		action_idx=0
		for pIdx,parent in enumerate(self.parents.values()):
			if(not parent.isAlive()):
				continue
			actionInfo=self.actionInfos[pIdx]
			myMotion=MotionState(parent.observables["motion"])
			vel=myMotion.vel
			#roll
			actionInfo.roll = 2 * unflattened_action[action_idx] - 1
			action_idx+=1
			self.last_action_obs[pIdx,0] = actionInfo.roll

			#pitch
			actionInfo.pitch = 2 * unflattened_action[action_idx] - 1
			action_idx+=1
			self.last_action_obs[pIdx,1] = actionInfo.pitch
			
			#pitch
			actionInfo.yaw = 2 * unflattened_action[action_idx] - 1
			action_idx+=1
			self.last_action_obs[pIdx,2] = actionInfo.yaw			

			#throttle
			actionInfo.throttle = unflattened_action[action_idx]
			action_idx+=1
			self.last_action_obs[pIdx,3] = actionInfo.throttle

			#下限速度の制限
			if(magnitude(vel)<self.minimumV):
				actionInfo.velRecovery=True
			if(magnitude(vel)>=self.minimumRecoveryV):
				actionInfo.velRecovery=False
			if(actionInfo.velRecovery):
				actionInfo.dstV=self.minimumRecoveryDstV
				actionInfo.throttle = 1

			#射撃
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
			shotTarget = -1
			#射撃可否の判断、射撃コマンドの生成
			flyingMsls=0
			for msl in parent.observables.at_p("/weapon/missiles"):
				if(msl.at("isAlive")() and msl.at("hasLaunched")()):
					flyingMsls+=1
			if(
				parent.observables.at_p("/weapon/remMsls")()>0 and
				parent.observables.at_p("/weapon/launchable") and
				0 < len(self.lastTrackInfo) and
				flyingMsls<self.maxSimulShot
			):
				min_tgt = -1
				min_range = 1000000 
				for i in range(self.lastTrackInfo):
					track = self.lastTrackInfo[i]
					pseudo_ = calcPseudoRelativeRange(myMotion.pos, vel, track.pos, track.vel)
					if pseudo_ < min_range:
						min_tgt = i
						min_range = pseudo_
				if min_range < self.rMaxs[0]:
					shotTarget = min_tgt
			self.last_action_obs[pIdx,4] = shotTarget
			if(shotTarget>=0 and self.shot_interval > 20):
				actionInfo.launchFlag=True
				actionInfo.target=Track3D(self.lastTrackInfo[shotTarget])
				self.shot_interval = 0
			else:
				actionInfo.launchFlag=False
				actionInfo.target=Track3D()
				self.shot_interval += 1

			#以下はILの時にしか使わない
			self.observables[parent.getFullName()]["decision"]={
				"Roll":("Don't care"),
				"Fire":(actionInfo.launchFlag,actionInfo.target.to_json())
			}
			if(len(mws)>0 and self.use_override_evasion):
				self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
			else:
				if(self.dstAz_relative):
					self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_BODY",deltaAz)
				else:
					self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
			if(self.use_altitude_command):
				self.observables[parent.getFullName()]["decision"]["Vertical"]=("Pos",-actionInfo.dstAlt)
			else:
				self.observables[parent.getFullName()]["decision"]["Vertical"]=("El",-dstPitch)
			if(actionInfo.asThrottle):
				self.observables[parent.getFullName()]["decision"]["Throttle"]=("Throttle",actionInfo.dstThrottle)
			else:
				self.observables[parent.getFullName()]["decision"]["Throttle"]=("Vel",actionInfo.dstV)
	def control(self):
		for pIdx,parent in enumerate(self.parents.values()):
			if(not parent.isAlive()):
				continue
			actionInfo=self.actionInfos[pIdx]
			myMotion=MotionState(parent.observables["motion"])
			pos=myMotion.pos
			vel=myMotion.vel
			norm_vel = norm_vec(vel)
			#デフォルトの
			#まずは反射処理
			if(len(parent.observables["sensor"]["mws"]["track"])>0 and (not self.missleAlert)):
				#ミサイル回避の反射
				missileTrack=Track2D(self.parent.observables["sensor"]["mws"]["track"][0])
				destination = -missileTrack.dir
				destination_b = myMotion.relPtoB(destination)
				if(destinatuon_b[2]<0 or -20000 > pos[2]):
					destinatuon_b[2]=0
					destinatuon_b=destinatuon_b/np.linalg.norm(destinatuon_b)
				if(self.warnTime > 20 and np.dot(vel/np.linalg.norm(vel),missileTrack.dir)<0):
					destinatuon_b[2]=-sqrt(pow(destinatuon_b[0],2)+pow(destinatuon_b[1],2))
					destinatuon_b=destinatuon_b/np.linalg.norm(destinatuon_b)
				if(np.dot(vel/np.linalg.norm(vel),missileTrack.dir)>0.85):
					proDst=destinatuon_b
					destinatuon_b=vel/np.linalg.norm(vel)-missileTrack.dir
					destinatuon_b[2]=0
					destinatuon_b=destinatuon_b/(np.linalg.norm(destinatuon_b)+1e-10)
					destinatuon_b=destinatuon_b*sqrt(pow(proDst[0],2)+pow(proDst[1],2))
					destinatuon_b[2]=proDst[2]
				if destination_b[0] < 0:
					#回避目標方向が逆方向の場合、最大制動を行う
					actionInfo.roll = 1 if destination_b[1] > 0 else -1
					actionInfo.pitch = 1 if destination_b[2] < 0 else -1
					actionInfo.yaw = 1 if destination_b[1] > 0 else -1
					actionInfo.throttle = 1
				else:
					#同一方向になったらイースをかける
					actionInfo.roll = min(max(-1, destination_b[1] * 10), 1)
					actionInfo.pitch = -min(max(-1, destination_b[2] * 10), 1)
					actionInfo.yaw = min(max(-1, destination_b[1] * 10), 1)
					actionInfo.throttle = 1
				self.distRecovery = True
				self.breakTime = 0
				self.warnTime += 0.1	
			else:
				self.breakTime+=0.1
				self.warnTime=0
			invPos=pos*self.xyInv
			max_performance = parent.obserbables["spec"]["dynamics"]["optCruiseFuelFlowRatePerDistance"]
			if(parent.obserbables["proplusion"]["fuelRemaining"] < parent.obserbables["spec"]["proplusion"]["fuelCapacity"] * 0.25 + max(0,(self.dLine - invPos[0])) * 1.1 * max_performance):
				#燃料減ってきたら帰還させる
				actionInfo.use_fromDirandVel = True
				target_vel = 250
				target_height = 9000
				if invPos[2] < -target_height * 0.9:
					actionInfo.dstDir=np.array([0,-1,-1])*self.xyInv
				elif invPos[2] < -target_height * 1.1:
					actionInfo.dstDir=np.array([0,-1,1])*self.xyInv
				else:
					actionInfo.dstDir=np.array([0,-1,0])*self.xyInv
				if magnitude(vel) < target_vel * 0.9:
					actionInfo.throttle = 1
				elif magnitude(vel) > target_vel * 1.1:
					actionInfo.throttle = 0
				else:
					actionInfo.throttle = 0.5
			#目標方位の補正
			if(abs(pos[0])>=(1-2*(1-self.dOutLimitRatio))*self.dOut):
				actionInfo.use_fromDirandVel = True
				#戦域逸脱を避けるための方位補正
				#判定ラインの超過具合に応じて制限角度を変化させる。(無限遠でラインに直交、ライン上でラインと平行)
				over=self.dOutLimitScale*(abs(pos[0])/self.dOut-self.dOutLimitRatio)
				n=sqrt(norm_vel[0]*norm_vel[0]+norm_vel[1]*norm_vel[1])
				theta=atan(over)
				cs=cos(theta)
				sn=sin(theta)
				actionInfo.throttle = 1
				if(pos[0]>0):#北側
					if(vel[1]>0):#現在東向き
						#南東側のみ許容
						if(norm_vel[1]<0):
							#西を向きたい場合、dstDirは真南を限度とする
							actionInfo.dstDir=np.array([-n,0,norm_vel[2]])
						else:
							#東を向きたい場合、角度を制限する
							if(atan2(-norm_vel[0],norm_vel[1])<theta):
								actionInfo.dstDir=np.array([-n*sn,n*cs,norm_vel[2]])
					else:#現在西向き
						#南西側のみ許容
						if(norm_vel[1]>0):
							#東を向きたい場合、dstDirは真南を限度とする
							actionInfo.dstDir=np.array([-n,0,norm_vel[2]])
						else:
							#西を向きたい場合、角度を制限する
							if(atan2(-norm_vel[0],-norm_vel[1])<theta):
								actionInfo.dstDir=np.array([-n*sn,-n*cs,norm_vel[2]])
				else:#南側
					if(vel[1]>0):#現在東向き
						#北東側のみ許容
						if(norm_vel[1]<0):
							#西を向きたい場合、dstDirは真北を限度とする
							actionInfo.dstDir=np.array([n,0,norm_vel[2]])
						else:
							#東を向きたい場合、角度を制限する
							if(atan2(norm_vel[0],norm_vel[1])<theta):
								actionInfo.dstDir=np.array([n*sn,n*cs,norm_vel[2]])
					else:#現在西向き
						#北南西側のみ許容
						if(norm_vel[1]>0):
							#東を向きたい場合、dstDirは真北を限度とする
							actionInfo.dstDir=np.array([n,0,norm_vel[2]])
						else:
							#西を向きたい場合、角度を制限する
							if(atan2(norm_vel[0],-norm_vel[1])<theta):
								actionInfo.dstDir=np.array([n*sn,-n*cs,norm_vel[2]])
			#高度方向の補正(actionがピッチ指定の場合)
			if(not self.use_altitude_command):
				n=sqrt(norm_vel[0]*norm_vel[0]+norm_vel[1]*norm_vel[1])
				dstPitch=atan2(-norm_vel[2],n)
				#高度下限側
				bottom=self.altitudeKeeper(myMotion,norm_vel,self.altMin)
				minPitch=atan2(-bottom[2],sqrt(bottom[0]*bottom[0]+bottom[1]*bottom[1]))
				#高度上限側
				top=self.altitudeKeeper(myMotion,norm_vel,self.altMax)
				maxPitch=atan2(-top[2],sqrt(top[0]*top[0]+top[1]*top[1]))
				newPitch=max(minPitch,min(maxPitch,dstPitch))
				if not newPitch == dstPitch:
					actionInfo.use_fromDirandVel = True
					cs=cos(dstPitch)
					sn=sin(dstPitch)
					actionInfo.dstDir=np.array([norm_vel[0]/n*cs,norm_vel[1]/n*cs,-sn])
					actionInfo.throttle = 1
			#会敵していないようであれば直進させる(今回は帰るラインが非常に遠いから注意)
			if self.matchProcess == 0:
				actionInfo.use_fromDirandVel = True
				if self.parent.observables["spec"]["weapon"]["numMsls"] > 4:
					target_vel = 300
					target_height = 15000
				else:
					target_vel = 250
					target_height = 7000
				if invPos[2] < -target_height * 0.9:
					actionInfo.dstDir=np.array([0,-1,-1])*self.xyInv
				elif invPos[2] < -target_height * 1.1:
					actionInfo.dstDir=np.array([0,-1,1])*self.xyInv
				else:
					actionInfo.dstDir=np.array([0,-1,0])*self.xyInv
				if magnitude(vel) < target_vel * 0.9:
					actionInfo.throttle = 1
				elif magnitude(vel) > target_vel * 1.1:
					actionInfo.throttle = 0
				else:
					actionInfo.throttle = 0.5
			#ここからコマンドの生成
			if actionInfo.use_fromDirandVel:
				parent.setFlightControllerMode("fromDirAndVel")
				self.commands[parent.getFullName()]={
					"motion":{
						"dstDir":actionInfo.dstDir
					},
					"weapon":{
						"launch":actionInfo.launchFlag,
						"target":actionInfo.target.to_json()
					}
				}
				if(self.use_altitude_command):
					self.commands[parent.getFullName()]["motion"]["dstAlt"]=actionInfo.dstAlt
				if(actionInfo.asThrottle):
					self.commands[parent.getFullName()]["motion"]["dstThrottle"] = actionInfo.throttle
				elif(actionInfo.keepVel):
					self.commands[parent.getFullName()]["motion"]["dstAccel"]=0.0
				else:
					self.commands[parent.getFullName()]["motion"]["dstV"]=actionInfo.dstV
				actionInfo.launchFlag=False
			else:
				parent.setFlightControllerMode("direct")
				self.commands[parent.getFullName()]={
					"motion":{
						"roll":actionInfo.roll,
						"pitch":actionInfo.pitch,
						"yaw":actionInfo.yaw,
						"throttle":actionInfo.throttle
					},
					"weapon":{
						"launch":actionInfo.launchFlag,
						"target":actionInfo.target.to_json()
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
	
	def convertActionFromAnother(self,decision,command):#摸倣対称の行動または制御出力と近い行動を計算する
		interval=self.manager.getAgentInterval()*self.manager.getBaseTimeStep()
		unflattened_action=np.zeros(len(self.actionDims),dtype=np.int64)
		action_idx=0
		for pIdx,parent in enumerate(self.parents.values()):
			if(not parent.isAlive()):
				#左右旋回
				tmp_d=self.turnTable-0.0
				tmp_i=np.argmin(abs(tmp_d))
				unflattened_action[action_idx]=tmp_i
				action_idx+=1
				#上昇・下降
				if(self.use_altitude_command):
					tmp_d=self.altTable-0.0
				else:
					tmp_d=self.pitchTable-0.0
				tmp_i=np.argmin(abs(tmp_d))
				unflattened_action[action_idx]=tmp_i
				action_idx+=1
				#加減速
				if(not self.always_maxAB):
					tmp_d=self.accelTable-0.0
					tmp_i=np.argmin(abs(tmp_d))
					unflattened_action[action_idx]=tmp_i
					action_idx+=1
				#射撃
				unflattened_action[action_idx]=0
				action_idx+=1
				if(self.use_Rmax_fire):
					if(len(self.shotIntervalTable)>1):
						unflattened_action[action_idx]=0
						action_idx+=1
					if(len(self.shotThresholdTable)>1):
						unflattened_action[action_idx]=len(self.shotThresholdTable)-1
						action_idx+=1
				continue
			actionInfo=self.actionInfos[pIdx]
			myMotion=MotionState(parent.observables["motion"])
			#左右旋回
			decisionType=decision[parent.getFullName()]["Horizontal"][0]()
			value=decision[parent.getFullName()]["Horizontal"][1]()
			motionCmd=command[parent.getFullName()]["motion"]()
			dstAz_ex=0.0
			if(decisionType=="Az_NED"):
				dstAz_ex=value
			elif(decisionType=="Az_BODY"):
				dstAz_ex=value+myMotion.az
			elif("dstDir" in motionCmd):
				dd=motionCmd["dstDir"]
				dstAz_ex=atan2(dd[1],dd[0])
			else:
				raise ValueError("Given unsupported expert's decision & command for this Agent class.")
			deltaAz_ex=0.0
			def angle(track):
				return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
			mws=sorted([Track2D(t) for t in parent.observables.at_p("/sensor/mws/track")],key=angle)
			if(self.use_override_evasion and len(mws)>0):
				dr=np.zeros([3])
				for m in mws:
					dr+=m.dirI()
				dr/=np.linalg.norm(dr)
				deltaAz_ex=dstAz_ex-atan2(-dr[1],-dr[0])
				deltaAz_ex=atan2(sin(deltaAz_ex),cos(deltaAz_ex))
				tmp_d=self.evasion_turnTable-deltaAz_ex
			else:
				if(self.dstAz_relative):
					dstDir_ex = myMotion.relPtoH(np.array([cos(dstAz_ex),sin(dstAz_ex),0]))
				else:
					dstDir_ex = self.teamOrigin.relPtoB(np.array([cos(dstAz_ex),sin(dstAz_ex),0]))
				deltaAz_ex = atan2(dstDir_ex[1],dstDir_ex[0])
				tmp_d=self.turnTable-deltaAz_ex
			tmp_i=np.argmin(abs(tmp_d))
			unflattened_action[action_idx]=tmp_i
			action_idx+=1

			#上昇・下降
			decisionType=decision[parent.getFullName()]["Vertical"][0]()
			value=decision[parent.getFullName()]["Vertical"][1]()
			dstPitch_ex=0.0
			deltaAlt_ex=0.0
			refAlt=round(-myMotion.pos[2]/self.refAltInterval)*self.refAltInterval
			if(decisionType=="El"):
				dstPitch_ex=-value
				deltaAlt_ex=self.altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt
			elif(decisionType=="Pos"):
				dstPitch_ex=self.altitudeKeeper.getDstPitch(myMotion,-value)
				deltaAlt_ex=-value-refAlt
			elif("dstDir" in motionCmd):
				dd=motionCmd["dstDir"]
				dstPitch_ex=-atan2(dd[2],sqrt(dd[0]*dd[0]+dd[1]*dd[1]))
				deltaAlt_ex=self.altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt
			else:
				raise ValueError("Given unsupported expert's decision & command for this Agent class.")
			dstPitch_ex=atan2(sin(dstPitch_ex),cos(dstPitch_ex))
			if(self.use_altitude_command):
				tmp_d=self.altTable-deltaAlt_ex
			else:
				tmp_d=self.pitchTable-dstPitch_ex
			tmp_i=np.argmin(abs(tmp_d))
			unflattened_action[action_idx]=tmp_i
			action_idx+=1
			#加減速
			if(not self.always_maxAB):
				#本サンプルでは「現在の速度と目標速度の差」をactionとしてdstVを機体へのコマンドとしているため、
				#教師役の行動が目標速度以外(スロットルや加速度)の場合、正確な変換は困難である。
				#そのため、最も簡易的な変換の例として、最大減速、等速、最大加速の3値への置換を実装している。
				decisionType=decision[parent.getFullName()]["Throttle"][0]()
				value=decision[parent.getFullName()]["Throttle"][1]()
				V=np.linalg.norm(myMotion.vel)
				accelIdx=-1
				if("dstV" in motionCmd):
					dstV_ex=motionCmd["dstV"]
				elif(decisionType=="Vel"):
					dstV_ex=value
				elif(decisionType=="Throttle"):
					#0〜1のスロットルで指定していた場合
					th=0.3
					if(value>1-th):
						accelIdx=len(self.accelTable)-1
					elif(value<th):
						accelIdx=0
					else:
						dstV_ex=V
				elif(decisionType=="Accel"):
					#加速度ベースの指定だった場合
					eps=0.5
					if(abs(value)<eps):
						dstV_ex=V
					elif(value>0):
						accelIdx=len(self.accelTable)-1
					else:
						accelIdx=0
				else:
					raise ValueError("Given unsupported expert's decision & command for this Agent class.")
				if(accelIdx<0):
					deltaV_ex=dstV_ex-V
					tmp_d=self.accelTable-deltaV_ex
					tmp_i=np.argmin(abs(tmp_d))
					unflattened_action[action_idx]=tmp_i
				else:
					unflattened_action[action_idx]=accelIdx
				action_idx+=1

			#射撃
			shotTarget_ex=-1
			expertTarget=Track3D(decision[parent.getFullName()]["Fire"][1])
			if(decision[parent.getFullName()]["Fire"][0]()):
				for tIdx,t in enumerate(self.lastTrackInfo):
					if(t.isSame(expertTarget)):
						shotTarget_ex=tIdx
				if(shotTarget_ex>=self.maxTrackNum["Enemy"]):
					shotTarget_ex=-1
			unflattened_action[action_idx]=shotTarget_ex+1
			action_idx+=1
			if(self.use_Rmax_fire):
				if(shotTarget_ex<0):
					#射撃なしの場合、間隔と射程の条件は最も緩いものとする
					if(len(self.shotIntervalTable)>1):
						unflattened_action[action_idx]=0
						action_idx+=1
					if(len(self.shotThresholdTable)>1):
						unflattened_action[action_idx]=len(self.shotThresholdTable)-1
						action_idx+=1
				else:
					#射撃ありの場合、間隔と射程の条件は最も厳しいものとする
					if(not expertTarget.truth in actionInfo.lastShotTimes):
						actionInfo.lastShotTimes[expertTarget.truth]=0.0
					if(len(self.shotIntervalTable)>1):
						shotInterval_ex=self.manager.getTime()-actionInfo.lastShotTimes[expertTarget.truth]
						unflattened_action[action_idx]=0
						for i in range(len(self.shotIntervalTable)-1,-1,-1):
							if(self.shotIntervalTable[i]<shotInterval_ex):
								unflattened_action[action_idx]=i
								break
						action_idx+=1
					if(len(self.shotThresholdTable)>1):
						r=self.calcRNorm(parent,myMotion,expertTarget)
						unflattened_action[action_idx]=len(self.shotThresholdTable)-1
						for i in range(len(self.shotThresholdTable)):
							if(r<self.shotThresholdTable[i]):
								unflattened_action[action_idx]=i
								break
						action_idx+=1

		if(self.flatten_action_space):
			return self.flatten_action(unflattened_action,self.actionDims)
		else:
			return unflattened_action
	def controlWithAnotherAgent(self,decision,command):
		#基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
		self.control()
		#例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
		#self.commands[parent.getFullName()]["weapon"]=command[parent.getFullName()]["weapon"]

	def flatten_action(self,original_action,original_shape):
		"""MultiDiscretionなactionをDiscreteなactionに変換する。
		"""
		ret=0
		for i in range(len(original_shape)):
			ret*=original_shape[i]
			ret+=original_action[i]
		return ret
	def unflatten_action(self,flattened_action,original_shape):
		"""Discrete化されたactionを元のMultiDiscreteなactionに変換する。
		"""
		ret=np.zeros_like(dims)
		for i in range(len(dims)-1,-1,-1):
			ret[i]=idx%dims[i]
			idx-=ret[i]
			idx/=dims[i]
		return ret

	def isInside(self,lon,lat):
		"""ピクセル座標(lon,lat)が画像の範囲内かどうかを返す。
		"""
		return 0<=lon and lon<self.image_longitudinal_resolution and 0<=lat and lat<self.image_lateral_resolution
	def rposToGrid(self,dr):
		"""基準点からの相対位置drに対応するピクセル座標(lon,lat)を返す。
		"""
		lon=floor((dr[1]+self.image_side_range)*self.image_lateral_resolution/(2*self.image_side_range))
		lat=floor((dr[0]+self.image_back_range)*self.image_longitudinal_resolution/(self.image_front_range+self.image_back_range))
		return np.array([lon,lat])
	def gridToRpos(self,lon,lat,alt):
		"""指定した高度においてピクセル座標(lon,lat)に対応する基準点からの相対位置drを返す。
		"""
		return np.array([
			self.image_buffer_coords[0,lon,lat],
			self.image_buffer_coords[1,lon,lat],
			-alt
		])
	def posToGrid(self,pos):
		"""絶対位置posに対応するピクセル座標(lon,lat)を返す。
		"""
		#自陣営防衛ライン中央が基準点
		rpos=pos-self.teamOrigin.pos
		#慣性座標系から陣営座標系に回転
		rpos=self.teamOrigin.relPtoB(rpos)
		return self.rposToGrid(rpos)
	def gridToPos(self,lon,lat,alt):
		"""指定した高度においてピクセル座標(lon,lat)に対応する絶対位置posを返す。
		"""
		rpos=self.gridToRpos(lon,lat,alt)
		#陣営座標系から慣性座標系に回転
		rpos=self.teamOrigin.relBtoP(rpos)
		#自陣営防衛ライン中央が基準点
		rpos=rpos+self.teamOrigin.pos
		return rpos
	def plotPoint(self,pos,ch,value):
		"""画像バッファのチャネルchで慣性座標系での絶対位置posに対応する点の値をvalueにする。
		"""
		g=self.posToGrid(pos)
		lon=g[0]
		lat=g[1]
		if(self.isInside(lon,lat)):
			self.image_buffer[ch,lon,lat]=value
	def plotLine(self,pBegin,pEnd,ch,value):
		"""画像バッファのチャネルchで慣性座標系での絶対位置pBeginからpEndまでの線分に対応する各点の値をvalueにする。
		線分の描画にはブレゼンハムのアルゴリズムを使用している。
		"""
		gBegin=self.posToGrid(pBegin)
		gEnd=self.posToGrid(pEnd)
		x0=gBegin[0]
		y0=gBegin[1]
		x1=gEnd[0]
		y1=gEnd[1]
		steep=abs(y1-y0)>abs(x1-x0)
		if(steep):
			x0,y0=y0,x0
			x1,y1=y1,x1
		if(x0>x1):
			x0,x1=x1,x0
			y0,y1=y1,y0
		deltax=x1-x0
		deltay=abs(y1-y0)
		error=deltax/2
		ystep = 1 if y0<y1 else -1
		y=y0
		for x in range(x0,x1+1):
			if(steep):
				if(self.isInside(y,x)):
					self.image_buffer[ch,y,x]=value
			else:
				if(self.isInside(x,y)):
					self.image_buffer[ch,x,y]=value
			error-=deltay
			if(error<0):
				y+=ystep
				error+=deltax

	def calcRHead(self,parent,myMotion,track):
		#相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程(RHead)を返す。
		rt=track.posI()
		vt=track.velI()
		rs=myMotion.pos
		vs=myMotion.vel
		return parent.getRmax(rs,vs,rt,vt,0.0)
	def calcRTail(self,parent,myMotion,track):
		#相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程(RTail)を返す。
		rt=track.posI()
		vt=track.velI()
		rs=myMotion.pos
		vs=myMotion.vel
		return parent.getRmax(rs,vs,rt,vt,np.pi)
	def calcRNorm(self,parent,myMotion,track):
		#RTail→0、RHead→1として正規化した距離を返す。
		RHead=self.calcRHead(parent,myMotion,track)
		RTail=self.calcRTail(parent,myMotion,track)
		rs=myMotion.pos
		rt=track.posI()
		r=np.linalg.norm(rs-rt)-RTail
		delta=RHead-RTail
		outRangeScale=100000.0
		if(delta==0):
			if(r<0):
				r=r/outRangeScale
			elif(r>0):
				r=1+r/outRangeScale
			else:
				r=0
		else:
			if(r<0):
				r=r/outRangeScale
			elif(r>delta):
				r=1+(r-delta)/outRangeScale
			else:
				r/=delta
		return r

#importした時に登録はしておく(Configから呼び出すために必要)
from ASRCAISim1.common import addPythonClass
import os
addPythonClass('Agent', 'AMayAgent', AMayAgent)
Factory.addDefaultModelsFromJsonFile(os.path.join(os.path.dirname(__file__),"./config/AMayAgentConfig.json"))