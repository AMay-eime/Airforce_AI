# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
from gym import spaces
import numpy as np
import sys
from ASRCAISim1.libCore import *

class R4PyAgentSample01S(SingleAssetAgent):
	"""1機につき1つのAgentを割り当てる、分散方式での行動判断モデルの実装例。
	1. observationの形式
		* Blue側でもRed側でも同じになるように、自機座標系でないベクトル量は陣営座標系(自陣営の進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)で表現する。
		* 戦域を2次元画像で捉えるimageと、単に実数ベクトルで表すvectorの2種類を用意しており、更に両方を持ったdictとすることも可能である。
		* image, vectorともに、一つのobservation内に時系列情報を埋め込める形として実装している。
		* imageデータの仕様
			* 画像の解像度はimage_longitudinal_resolution×image_lateral_resolutionとする。
			* ピクセルのインデックスは画像の左下を(0,0)とする。
			* 戦域を描画する座標系(描画座標系)の+x方向は画像の上方向、+y方向は画像の右方向とする。
			* 描画座標系の原点は、image_relative_positionをTrueとした場合は自機位置、Falseとした場合は自陣営防衛ライン中央とする。
			* 描画座標系の+x方向は、image_rotateをTrueとした場合は自機正面、Falseとした場合は自陣営の進行すべき方向とする。
			* 描画座標系の+y方向は、戦域を真上から見た図として描画されるような向きに取る。
			* x軸方向(縦方向)の描画範囲はimage_front_range+image_back_rangeとし、原点は画像下端からimage_back_range/(image_front_range+image_back_range)の位置とする。
			* y軸方向(横方向)の描画範囲は2×image_side_rangeとし、原点は画像の中央とする。
			* 各チャネルの内容は以下の通り。
				1. 自機の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
				2. 味方の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
				3. 彼機の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
				4. 味方誘導弾の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
				5. 自機のレーダ覆域…現時刻の覆域内を1とする。
				6. 味方のレーダ覆域…現時刻の覆域内を1とする。
				7. 彼我防衛ライン…彼側の防衛ラインを1、我側の防衛ラインを-1とする。
				8. 場外ライン…ライン上を1とする。
		* vectorデータの仕様
			* Atariゲームにおけるframe stackingと同様の方法で現時刻と過去の数点のデータを並べて一つのobservationとすることを可能としている。
			* 直前のstepでとった行動を表すベクトルを含めることを可能としている。
			* 各時刻のデータの内訳は以下の通り
				1. 直前の行動(4+maxTrackNum["Enemy"]、include_last_actionをTrueとした場合のみ)
					1. 左右旋回…前回の目標方位を現在の方位からの差分で格納。
					2. 上昇・下降…前回の目標高度またはピッチ角を格納。
					3. 加減速…前回の目標加速度を、accelTableの絶対値が最大の要素で正規化したものを格納。always_maxABがtrueの場合は常に1とする。
					4. 射撃…前回の射撃コマンドを、射撃なし+射撃対象IDのone hotで格納。
				2. 味方機情報(1機あたり13+3×maxMissileNum["Enemy"])…自機の情報を先頭に、他の味方機の情報を合わせてmaxTrackNum["Friend"]機分格納する。存在しない部分は0埋め。
					1. 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					2. 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					3. 初期弾数(1)
					4. 残弾数(1)
					5. 姿勢(3)…バンク角、α(迎角)、β(横滑り角)の3次元をradで表現したもの。
					6. 余剰燃料(1)…現在の余剰燃料を距離に換算したものを、2*dLineで正規化して-1〜+1にクリッピングしたもの。
					7. MWS検出情報(3×maxMissileNum["Enemy"])…MWSが検出した誘導弾のうち正面に最も近いものからmaxMissileNum["Enemy"]発分、方位情報を格納。存在しない部分は0埋め。
				3. 彼機情報(1機あたり14)…自陣営が探知している彼機の航跡を、自機に近いものから順にmaxTrackNum["Enemy"]機分格納する。存在しない部分は0埋め。
					1. 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					2. 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					3. 最近傍の味方誘導弾(4)…この航跡に対し飛翔中で最近傍の誘導弾の距離と誘導状態(one hot)を格納。
					4. 射程情報(3)…この航跡に対する自機のRHead、RTailをhorizontalNormalizerで正規化したものと、自機との距離をRTail→0、RHead→1となるように正規化したものを格納。
				4. 味方誘導弾情報(1発あたり)…自陣営の飛翔中誘導弾の情報を、射撃時刻が古いものからmaxMissileNum["Friend"]発分格納する。存在しない部分は0埋め。
					1. 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					2. 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					3. 目標情報(5)…目標との距離、誘導状態(one hot)、対応する彼機情報のIDを格納。
	2. 行動の形式について
		左右旋回、上昇・下降、加減速、射撃の4種類を離散化したものを全機分並べたものをMultiDiscreteで与える。
		1. 左右旋回
		ある基準方位を0として目標方位(右を正)で指定する。
		基準方位は、dstAz_relativeフラグをTrueとした場合、自機正面となり、Falseとした場合、自陣営の進行すべき方向となる。
		目標方位の選択肢はturnTableで与える。
		また、use_override_evasionフラグをTrueとした場合、MWS検出時の基準方位と目標方位テーブルを上書きすることが可能。
		基準方位は検出された誘導弾の到来方位の平均値と逆向きとし、目標方位テーブルはevasion_turnTableで与える。
		2. 上昇・下降
		use_altitude_commandフラグをTrueとした場合、基準高度からの目標高度差で指定する。
		基準高度はrefAltInterval間隔で最寄りの高度とし、目標高度差の選択肢はaltTableで与える。
		use_altitude_commandフラグをFalseとした場合、水平を0とした目標ピッチ角(上昇を正))で指定する。
		目標ピッチ角の選択肢はpitchTableで与える。
		3. 加減速
		always_maxABをTrueとした場合は常時最大推力を出力するものとし、spaceからは削除する。
		Falseとした場合は基準速度(=現在速度)に対する速度差として目標速度を指定する。
		速度差の選択肢はaccelTableで与える。
		4. 射撃
		射撃対象を、0を射撃なし、1〜maxTrackNum["Enemy"]を対応するlastTrackInfoのTrackとしたone hot形式で指定する。
		射撃有無については、use_Rmax_fireフラグをTrueとした場合、誘導弾モデルの射程テーブルと射撃間隔の指定を用いて行う。
		Falseの場合、射撃対象でTrackを指定していた場合は射撃ありとして扱う。
		射撃間隔は各Trackごとに固有のタイマーで、前回の射撃からの必要経過時間を指す。
		射程閾値はRTailを0、RHeadを1として距離を正規化した値で、閾値を下回った場合に射撃を行う。
		射撃間隔の選択肢はshotIntervalTableで、射程閾値の選択肢はshotThresholdTableで与える。
	3. 行動制限について
		このサンプルでは、いくつかの観点でAIの行動判断を上書きして、一般に望ましくないと思われる挙動を抑制する例を実装している。
		1. 高度の制限
		altMinを下限、altMaxを上限とした高度範囲内で行動するように、それを逸脱しそうな場合に制限をかける。
		spaceが目標高度の場合は単純なクリッピングで範囲を限定する。
		spaceが目標ピッチ角の場合は、簡易な高度制御則を用いて制限ピッチ角を計算し、それを用いてクリッピングを行う。
		2. 場外の制限
		戦域中心からdOutLimitRatio×dOutの位置に引いた基準ラインの超過具合に応じて目標方位に制限をかける。
		無限遠で基準ラインに直交、基準ライン上で基準ラインと平行になるようなatanスケールでの角度補正を行う。
		補正を行う範囲は、基準ラインよりも更にdOutLimitRatio×dOut内側のラインからとする。
		3. 同時射撃数の制限
		自機の飛翔中誘導弾がmaxSimulShot発以上のときは新たな射撃を行えないようにしている。
		4. 下限速度の制限
		速度がminimumVを下回った場合、minimumRecoveryVを上回るまでの間、
		目標速度をminimumRecovertDstVに固定することで、低速域での飛行を抑制している。
	Attributes:
		* modelConfigで設定するもの
			* observation spaceの設定
			maxTrackNum (dict[str,int]): vectorのobservationとして考慮する彼我の戦闘機の航跡数。{"Friend":3,"Enemy":4}のようにdictで指定する。
			maxMissileNum (dict[str,int]): vectorのobservationとして考慮する彼我の誘導弾の航跡数。書式はmaxTrackNumと同じ。
			horizontalNormalizer (float): 水平方向の位置・距離の正規化のための除数
			verticalNormalizer (float): 高度方向の正規化のための除数
			fgtrVelNormalizer (float): 機体速度の正規化のための除数
			mslVelNormalizer (float): 誘導弾速度の正規化のための除数
				* 2次元画像としてのobservation
				use_image_observation (bool): 戦域を2次元画像で表現したobservationを使用するかどうか
				image_longitudinal_resolution(int): 前後方向の解像度
				image_lateral_resolution(int): 左右方向の解像度
				image_front_range(float): 前方の描画範囲(m)
				image_back_range(float): 後方の描画範囲(m)
				image_side_range(float): 側方の描画範囲(m)
				image_horizon(int): 軌跡を描画する秒数
				image_interval(int): 軌跡を描画する間隔
				image_rotate(bool): 画像化の基準座標系。trueの場合自機正面を-x軸とするように回転。falseの場合、慣性座標系そのまま(赤青の陣営による反転は行う)。
				image_relative_position(bool): 画像化の基準位置。trueの場合自機位置、falseの場合自陣営防衛ライン中央とする。
				* 実数値ベクトルとしてのobservation
				use_vector_observation(bool): 戦域を実数値ベクトルで表現したobservationを使用するかどうか
				include_last_action(bool): 前回の行動に関する情報を含めるかどうか
				vector_past_points(list[int]): キーフレームとしてobservationに追加する過去の時刻
			* action spaceの設定
				* spaceの形式に関する設定
				flatten_action_space(bool): 上記のaction spaceをDiscreteにflattenするかどうか。falseにした場合はMultiDiscreteとして扱う。
				* 左右旋回に関する設定
				dstAz_relative(bool): 旋回の原点に関する設定。trueの場合は自機正面、falseの場合は自陣営の進行方向が原点となる。
				turnTable(list[float]): 通常時の目標方位(deg)テーブル(右が正)
				evasion_turnTable(list[float]): MWS作動時の目標方位(deg)テーブル(検出した誘導弾を背にした方位を0とし、右が正)
				use_override_evasion(bool): MWS作動時の目標方位テーブルを専用のものに置き換えるかどうか
				* 上昇・下降に関する設定
				pitchTable(list[float]): 目標ピッチ角(deg)テーブル(上昇が正)
				altTable(list[float]): 目標高度(m)テーブル(最寄りの基準高度からの高度差で指定)
				refAltInterval(float): 基準高度グリッドの間隔(m)
				use_altitude_command(bool): ピッチ角と高度のどちらを使用するか。trueでピッチ、falseで高度を使用する。
				* 加減速に関する設定
				accelTable(list[float]): 目標加速度テーブル
				always_maxAB(bool): 常時maxABするかどうか。
				* 射撃に関する設定
				shotIntervalTable(list[float]) 同一目標に対する射撃間隔(秒)テーブル
				shotThresholdTable(list[float]): 射撃閾値テーブル(RTailを0、RHeadを1とした線形空間での閾値)
				use_Rmax_fire(bool): 既存の射程テーブルを用いて射撃判定を行うかどうか。
			* 行動制限に関する設定
				* 高度制限に関する設定(ピッチ角をactionにしている場合のみ有効)
				altMin(float): 下限高度
				altMax(float): 上限高度
				altitudeKeeper(dict): ピッチ角制限を計算するための高度制御則。サンプルではFlightControllerUtility.hのAltitudeKeeperクラスを使用しており、configではそのパラメータをdictで指定する。
				* 場外制限に関する設定
				dOutLimitRatio(float): 場外防止を行う閾値。RulerのdOutに対する割合で指定。
				dOutLimitScale(float): 場外防止の復元力に関する係数。
				* 同時射撃数の制限に関する設定
				maxSimulShot(int): 自身が同時に射撃可能な誘導弾数の上限
				* 下限速度の制限に関する設定
				minimumV(float): 低速域からの回復を開始する速度
				minimumRecoveryV(float): 低速域からの回復を終了する速度
				minimumRecoveryDstV(float): 低速域からの回復時に設定する目標速度
		* 内部変数
			* observationに関するもの
			 _observation_space(gym.spaces.Space): 観測空間を表すgym.spaces.Space
			ourMotion(list[MotionState]): 味方各機のMotionState
			ourObservables(list[nl::json]): 味方各機のobservables
			lastTrackInfo(list[Track3D]): 自陣営が捉えている彼機航跡
			msls(list[nl::json]): 自陣営の誘導弾(のobservables)
				* 2次元画像としてのobservation
				numChannels(int): チャネル数
				image_buffer(3-dim ndarray): 画像データバッファ。shape=[numChannels,image_longitudinal_resolution,image_lateral_resolution]
				image_buffer_coords(3-dim ndarray): ピクセル座標(lon,lat)と戦域座標(x,y)のマッピング。shape=[2,image_longitudinal_resolution,image_lateral_resolution]
				image_past_data(list[InstantInfo]): 軌跡描画用の過去データのバッファ
				* 実数値ベクトルとしてのobservation
				last_action_dim(int): 前回の行動の次元
				friend_dim(int): 味方諸元1機あたりの次元
				enemy_dim(int): 彼機諸元1機あたりの次元
				friend_msl_dim(int): 味方誘導弾諸元1発あたりの次元
				vector_single_dim(int): 1時刻分のvector observationの次元
				last_action_obs(1-dim ndarray): 前回の行動を格納するバッファ。shape=[last_action_dim]
				friend_temporal(2-dim ndarray): 味方諸元を格納するバッファ。shape=[maxTrackNum["Friend"],friend_dim]
				enemy_temporal(2-dim ndarray): 彼機諸元を格納するバッファ。shape=[maxTrackNum["Enemy"],enemy_dim]
				friend_msl_temporal(2-dim ndarray): 味方誘導弾諸元を格納するバッファ。shape=[maxMissileNum["Friend"],friend_msl_dim]
				vector_past_data(list[1-dim ndarray]): 過去のvectorデータのバッファ。各要素のshapeは[vector_single_dim]
			* actionに関するもの
			_action_space(gym.spaces.Space): 行動空間を表すgym.spaces.Space
			actionDims(list[int]): actionの各要素の次元
			totalActionDim(int): actionの総次元数
				* 機体に対するコマンドを生成するための構造体
				class ActionInfo
					dstDir(3-elem 1-dim ndarray): 目標進行方向
					dstAlt(float): 目標高度
					velRecovery(bool): 下限速度制限からの回復中かどうか
					asThrottle(bool): 加減速についてスロットルでコマンドを生成するかどうか
					keepVel(bool): 加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
					dstThrottle(float): 目標スロットル
					dstV(float): 目標速度
					launchFlag(bool): /射撃するかどうか
					target(Track3D): 射撃対象
					lastShotTimes(dict[boost::uuids::uuid,double]): 各Trackに対する直前の射撃時刻
				actionInfo(ActionInfo): 自機の行動を表す変数
			* Ruler、陣営に関するもの
			dOut(float): 戦域中心から場外ラインまでの距離
			dLine(float): 戦域中心から防衛ラインまでの距離
			teamOrigin(MotionState): 陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すMotionState
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
			self.dstDir=np.array([1.0,0.0,0.0]) #目標進行方向
			self.dstAlt=10000.0 #目標高度
			self.velRecovery=False #下限速度制限からの回復中かどうか
			self.asThrottle=False #加減速についてスロットルでコマンドを生成するかどうか
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
		self.maxMissileNum=getValueFromJsonKRD(self.modelConfig,"maxMissileNum",self.randomGen,{"Friend":8,"Enemy":1})
		self.horizontalNormalizer=getValueFromJsonKRD(self.modelConfig,"horizontalNormalizer",self.randomGen,100000.0)
		self.verticalNormalizer=getValueFromJsonKRD(self.modelConfig,"verticalNormalizer",self.randomGen,20000.0)
		self.fgtrVelNormalizer=getValueFromJsonKRD(self.modelConfig,"fgtrVelNormalizer",self.randomGen,300.0)
		self.mslVelNormalizer=getValueFromJsonKRD(self.modelConfig,"mslVelNormalizer",self.randomGen,1200.0)
		self.use_image_observation=getValueFromJsonKRD(self.modelConfig,"use_image_observation",self.randomGen,True)
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
			self.image_rotate=getValueFromJsonKRD(self.modelConfig,"image_rotate",self.randomGen,False)
			self.image_relative_position=getValueFromJsonKRD(self.modelConfig,"image_relative_position",self.randomGen,False)
		#  実数値ベクトルとして
		if(self.use_vector_observation):
			self.include_last_action=getValueFromJsonKRD(self.modelConfig,"include_last_action",self.randomGen,True)
			self.vector_past_points=sorted(getValueFromJsonKRD(self.modelConfig,"vector_past_points",self.randomGen,[]))
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
			self.numChannels=8
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
			self.enemy_dim=3+4+4+3
			self.friend_msl_dim=3+4+1+3+1
			#データバッファの生成
			self.friend_temporal=np.zeros([self.maxTrackNum["Friend"],self.friend_dim],dtype=np.float32)
			self.enemy_temporal=np.zeros([self.maxTrackNum["Enemy"],self.enemy_dim],dtype=np.float32)
			self.friend_msl_temporal=np.zeros([self.maxMissileNum["Friend"],self.friend_msl_dim],dtype=np.float32)
			self.last_action_obs=np.zeros([self.last_action_dim],dtype=np.float32)
			self.vector_single_dim=(
				(self.last_action_dim if self.include_last_action else 0)+
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
		self.actionInfo=self.ActionInfo()
	def validate(self):
		#Rulerに関する情報の取得
		rulerObs=self.manager.getRuler()().observables()
		self.dOut=rulerObs["dOut"]
		self.dLine=rulerObs["dLine"]
		eastSider=rulerObs["eastSider"]
		self.teamOrigin=self.TeamOrigin(self.getTeam()==eastSider,self.dLine)
		if(self.parent.isinstance(CoordinatedFighter) or self.parent.isinstance(SixDoFFighter)):
			self.parent.setFlightControllerMode("fromDirAndVel")
		else:
			raise ValueError("R4PyAgentSample01S accepts only SixDoFFighter or CoordinatedFighter for its parent.")
		self.actionInfo.dstDir=np.array([0., -1. if self.getTeam()==eastSider else +1., 0.])
		self.last_action_obs[0]=atan2(self.actionInfo.dstDir[1],self.actionInfo.dstDir[0])
		self.actionInfo.lastShotTimes={}
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
				bufferSize=len(self.turnTablevector_past_data)
				bufferIdx=bufferSize-1-(stepCount%bufferSize)
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
		myMotion=MotionState(self.parent.observables["motion"])
		self.ourMotion=[myMotion]
		self.ourObservables=[self.parent.observables]
		for n,f in self.parent.observables.at_p("/shared/fighter").items():
			if(n==self.parent.getFullName()):#自分を除く
				continue
			if(f["isAlive"]()):
				self.ourMotion.append(MotionState(f["motion"]))
			else:
				self.ourMotion.append(MotionState())
			self.ourObservables.append(f)
		#彼機(味方の誰かが探知しているもののみ)
		#観測されている航跡を、自機に近いものから順にソートしてlastTrackInfoに格納する。
		#lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
		def distance(track):
			return np.linalg.norm(myMotion.pos-track.posI())
		self.lastTrackInfo=sorted([Track3D(t) for t in self.ourObservables[0].at_p("/sensor/track")],key=distance)
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
		#ch0:自機軌跡
		#ch1:味方軌跡
		#ch2:彼機軌跡
		#ch3:味方誘導弾軌跡
		#ch4:自機覆域
		#ch5:味方覆域
		#ch6:彼我防衛ライン
		#ch7:場外ライン
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
			#ch0:自機
			self.plotPoint(
				frame.friend_pos[0],
				ch,
				1.0-1.0*t/self.image_horizon
			)
			ch+=1
			#ch1:味方
			for i in range(1,len(frame.friend_pos)):
				self.plotPoint(
					frame.friend_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)
			ch+=1
			#ch2:彼機
			for i in range(len(frame.enemy_pos)):
				self.plotPoint(
					frame.enemy_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)
			ch+=1
			#ch3:味方誘導弾
			for i in range(1,len(frame.friend_msl_pos)):
				self.plotPoint(
					frame.friend_msl_pos[i],
					ch,
					1.0-1.0*t/self.image_horizon
				)

		#ch4:自機レーダ覆域, ch5:味方レーダ覆域
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
								4 if i==0 else 5,
								1.0
							)

		#ch6:彼我防衛ライン
		#彼側防衛ライン
		eps=1e-10 #戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
			6,
			1.0
		)
		#我側防衛ライン
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
			6,
			-1.0
		)
		#ch7:場外ライン
		#進行方向右側
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
			7,
			1.0
		)
		#進行方向左側
		self.plotLine(
			self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
			self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
			7,
			1.0
		)
	def makeVectorObservation(self):
		myMotion=MotionState(self.parent.observables["motion"])
		#前回の行動は前回のdeployで計算済だが、dstAzのみ現在のazからの差に置き換える
		deltaAz=self.last_action_obs[0]-myMotion.az
		self.last_action_obs[0]=atan2(sin(deltaAz),cos(deltaAz))

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
			#自機からこの航跡への射程情報
			self.enemy_temporal[tIdx,11]=self.calcRHead(myMotion,t)/self.horizontalNormalizer#RHead
			self.enemy_temporal[tIdx,12]=self.calcRTail(myMotion,t)/self.horizontalNormalizer#RTail
			self.enemy_temporal[tIdx,13]=self.calcRNorm(myMotion,t)#現在の距離をRTail〜RHeadで正規化したもの
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
			(self.last_action_dim if self.include_last_action else 0)+
			self.maxTrackNum["Friend"]*self.friend_dim+
			self.maxTrackNum["Enemy"]*self.enemy_dim+
			self.maxMissileNum["Friend"]*self.friend_msl_dim
		],dtype=np.float32)
		ofs=0
		if(self.include_last_action):
			d=self.last_action_dim
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
		myMotion=MotionState(self.parent.observables["motion"])
		self.last_action_obs=np.zeros([self.last_action_dim],dtype=np.float32)
		if(self.flatten_action_space):
			unflattened_action=self.unflatten_action(action,self.actionDims)
		else:
			unflattened_action=action
		action_idx=0
		#左右旋回
		deltaAz=self.turnTable[unflattened_action[action_idx]]
		def angle(track):
			return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
		mws=sorted([Track2D(t) for t in self.parent.observables.at_p("/sensor/mws/track")],key=angle)
		if(len(mws)>0 and self.use_override_evasion):
			deltaAz=self.evasion_turnTable[unflattened_action[action_idx]]
			dr=np.zeros([3])
			for m in mws:
				dr+=m.dirI()
			dr/=np.linalg.norm(dr)
			dstAz=atan2(-dr[1],-dr[0])+deltaAz
			self.actionInfo.dstDir=np.array([cos(dstAz),sin(dstAz),0])
		elif(self.dstAz_relative):
			self.actionInfo.dstDir=myMotion.relHtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
		else:
			self.actionInfo.dstDir=self.teamOrigin.relBtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
		action_idx+=1
		dstAz=atan2(self.actionInfo.dstDir[1],self.actionInfo.dstDir[0])
		self.last_action_obs[0]=dstAz

		#上昇・下降
		if(self.use_altitude_command):
			refAlt=round(-myMotion.pos(2)/self.refAltInterval)*self.refAltInterval
			self.actionInfo.dstAlt=max(self.altMin,min(self.altMax,refAlt+self.altTable[unflattened_action[action_idx]]))
			dstPitch=0#dstAltをcommandsに与えればSixDoFFighter::FlightControllerのaltitudeKeeperで別途計算されるので0でよい。
		else:
			dstPitch=self.pitchTable[unflattened_action[action_idx]]
		action_idx+=1
		self.actionInfo.dstDir=np.array([self.actionInfo.dstDir[0]*cos(dstPitch),self.actionInfo.dstDir[1]*cos(dstPitch),-sin(dstPitch)])
		self.last_action_obs[1]=self.actionInfo.dstAlt if self.use_altitude_command else dstPitch

		#加減速
		V=np.linalg.norm(myMotion.vel)
		if(self.always_maxAB):
			self.actionInfo.asThrottle=True
			self.actionInfo.keepVel=False
			self.actionInfo.dstThrottle=1.0
			self.last_action_obs[2]=1.0
		else:
			self.actionInfo.asThrottle=False
			accel=self.accelTable[unflattened_action[action_idx]]
			action_idx+=1
			self.actionInfo.dstV=V+accel
			self.actionInfo.keepVel = accel==0.0
			self.last_action_obs[2]=accel/max(self.accelTable[-1],self.accelTable[0])
		#下限速度の制限
		if(V<self.minimumV):
			self.actionInfo.velRecovery=True
		if(V>=self.minimumRecoveryV):
			self.actionInfo.velRecovery=False
		if(self.actionInfo.velRecovery):
			self.actionInfo.dstV=self.minimumRecoveryDstV
			self.actionInfo.asThrottle=False

		#射撃
		#actionのパース
		shotTarget=unflattened_action[action_idx]-1
		action_idx+=1
		if(self.use_Rmax_fire):
			if(len(self.shotIntervalTable)>1):
				shotInterval=self.shotIntervalTable[unflattened_action[action_idx]]
				action_idx+=1
			else:
				shotInterval=self.shotIntervalTable[0]
			if(len(self.shotThresholdTable)>1):
				shotThreshold=self.shotThresholdTable[unflattened_action[action_idx]]
				action_idx+=1
			else:
				shotThreshold=shotThresholdTable[0]
		#射撃可否の判断、射撃コマンドの生成
		flyingMsls=0
		for msl in self.parent.observables.at_p("/weapon/missiles"):
			if(msl.at("isAlive")() and msl.at("hasLaunched")()):
				flyingMsls+=1
		if(
			self.parent.observables.at_p("/weapon/remMsls")()>0 and
			self.parent.observables.at_p("/weapon/launchable") and
			shotTarget>=0 and
			shotTarget<len(self.lastTrackInfo) and
			flyingMsls<self.maxSimulShot
		):
			if(self.use_Rmax_fire):
				rMin=np.inf
				t=self.lastTrackInfo[shotTarget]
				r=self.calcRNorm(myMotion,t)
				if(r<=shotThreshold):
					#射程の条件を満たしている
					if(not t.truth in self.actionInfo.lastShotTimes):
						self.actionInfo.lastShotTimes[t.truth]=0.0
					if(self.manager.getTime()-self.actionInfo.lastShotTimes[t.truth]>=shotInterval):
						#射撃間隔の条件を満たしている
						self.actionInfo.lastShotTimes[t.truth]=self.manager.getTime()
					else:
						#射撃間隔の条件を満たさない
						shotTarget=-1
				else:
					#射程の条件を満たさない
					shotTarget=-1
		else:
			shotTarget=-1
		self.last_action_obs[3+(shotTarget+1)]=1
		if(shotTarget>=0):
			self.actionInfo.launchFlag=True
			self.actionInfo.target=self.lastTrackInfo[shotTarget]
		else:
			self.actionInfo.launchFlag=False
			self.actionInfo.target=Track3D()

		self.observables[self.parent.getFullName()]["decision"]={
			"Roll":("Don't care"),
			"Fire":(self.actionInfo.launchFlag,self.actionInfo.target.to_json())
		}
		if(len(mws)>0 and self.use_override_evasion):
			self.observables[self.parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
		else:
			if(self.dstAz_relative):
				self.observables[self.parent.getFullName()]["decision"]["Horizontal"]=("Az_BODY",deltaAz)
			else:
				self.observables[self.parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
		if(self.use_altitude_command):
			self.observables[self.parent.getFullName()]["decision"]["Vertical"]=("Pos",-self.actionInfo.dstAlt)
		else:
			self.observables[self.parent.getFullName()]["decision"]["Vertical"]=("El",-dstPitch)
		if(self.actionInfo.asThrottle):
			self.observables[self.parent.getFullName()]["decision"]["Throttle"]=("Throttle",self.actionInfo.dstThrottle)
		else:
			self.observables[self.parent.getFullName()]["decision"]["Throttle"]=("Vel",self.actionInfo.dstV)
	def control(self):
		myMotion=MotionState(self.parent.observables["motion"])
		pos=myMotion.pos
		vel=myMotion.vel
		#目標方位の補正
		if(abs(pos[0])>=(1-2*(1-self.dOutLimitRatio))*self.dOut):
			#戦域逸脱を避けるための方位補正
			#判定ラインの超過具合に応じて制限角度を変化させる。(無限遠でラインに直交、ライン上でラインと平行)
			over=self.dOutLimitScale*(abs(pos[0])/self.dOut-self.dOutLimitRatio)
			n=sqrt(self.actionInfo.dstDir[0]*self.actionInfo.dstDir[0]+self.actionInfo.dstDir[1]*self.actionInfo.dstDir[1])
			theta=atan(over)
			cs=cos(theta)
			sn=sin(theta)
			if(pos[0]>0):#北側
				if(vel[1]>0):#現在東向き
					#南東側のみ許容
					if(self.actionInfo.dstDir[1]<0):
						#西を向きたい場合、dstDirは真南を限度とする
						self.actionInfo.dstDir=np.array([-n,0,self.actionInfo.dstDir[2]])
					else:
						#東を向きたい場合、角度を制限する
						if(atan2(-self.actionInfo.dstDir[0],self.actionInfo.dstDir[1])<theta):
							self.actionInfo.dstDir=np.array([-n*sn,n*cs,self.actionInfo.dstDir[2]])
				else:#現在西向き
					#南西側のみ許容
					if(self.actionInfo.dstDir[1]>0):
						#東を向きたい場合、dstDirは真南を限度とする
						self.actionInfo.dstDir=np.array([-n,0,self.actionInfo.dstDir[2]])
					else:
						#西を向きたい場合、角度を制限する
						if(atan2(-self.actionInfo.dstDir[0],-self.actionInfo.dstDir[1])<theta):
							self.actionInfo.dstDir=np.array([-n*sn,-n*cs,self.actionInfo.dstDir[2]])
			else:#南側
				if(vel[1]>0):#現在東向き
					#北東側のみ許容
					if(self.actionInfo.dstDir[1]<0):
						#西を向きたい場合、dstDirは真北を限度とする
						self.actionInfo.dstDir=np.array([n,0,self.actionInfo.dstDir[2]])
					else:
						#東を向きたい場合、角度を制限する
						if(atan2(self.actionInfo.dstDir[0],self.actionInfo.dstDir[1])<theta):
							self.actionInfo.dstDir=np.array([n*sn,n*cs,self.actionInfo.dstDir[2]])
				else:#現在西向き
					#北南西側のみ許容
					if(self.actionInfo.dstDir[1]>0):
						#東を向きたい場合、dstDirは真北を限度とする
						self.actionInfo.dstDir=np.array([n,0,self.actionInfo.dstDir[2]])
					else:
						#西を向きたい場合、角度を制限する
						if(atan2(self.actionInfo.dstDir[0],-self.actionInfo.dstDir[1])<theta):
							self.actionInfo.dstDir=np.array([n*sn,-n*cs,self.actionInfo.dstDir[2]])
		#高度方向の補正(actionがピッチ指定の場合)
		if(not self.use_altitude_command):
			n=sqrt(self.actionInfo.dstDir[0]*self.actionInfo.dstDir[0]+self.actionInfo.dstDir[1]*self.actionInfo.dstDir[1])
			dstPitch=atan2(-self.actionInfo.dstDir[2],n)
			#高度下限側
			bottom=self.altitudeKeeper(myMotion,self.actionInfo.dstDir,self.altMin)
			minPitch=atan2(-bottom[2],sqrt(bottom[0]*bottom[0]+bottom[1]*bottom[1]))
			#高度上限側
			top=self.altitudeKeeper(myMotion,self.actionInfo.dstDir,self.altMax)
			maxPitch=atan2(-top[2],sqrt(top[0]*top[0]+top[1]*top[1]))
			dstPitch=max(minPitch,min(maxPitch,dstPitch))
			cs=cos(dstPitch)
			sn=sin(dstPitch)
			self.actionInfo.dstDir=np.array([self.actionInfo.dstDir[0]/n*cs,self.actionInfo.dstDir[1]/n*cs,-sn])
		self.commands[self.parent.getFullName()]={
			"motion":{
				"dstDir":self.actionInfo.dstDir
			},
			"weapon":{
				"launch":self.actionInfo.launchFlag,
				"target":self.actionInfo.target.to_json()
			}
		}
		if(self.use_altitude_command):
			self.commands[self.parent.getFullName()]["motion"]["dstAlt"]=self.actionInfo.dstAlt
		if(self.actionInfo.asThrottle):
			self.commands[self.parent.getFullName()]["motion"]["dstThrottle"]=self.actionInfo.dstThrottle
		elif(self.actionInfo.keepVel):
			self.commands[self.parent.getFullName()]["motion"]["dstAccel"]=0.0
		else:
			self.commands[self.parent.getFullName()]["motion"]["dstV"]=self.actionInfo.dstV
		self.actionInfo.launchFlag=False
	def convertActionFromAnother(self,decision,command):#摸倣対称の行動または制御出力と近い行動を計算する
		interval=self.manager.getAgentInterval()*self.manager.getBaseTimeStep()
		myMotion=MotionState(self.parent.observables["motion"])
		unflattened_action=np.zeros(len(self.actionDims),dtype=np.int64)
		action_idx=0
		#左右旋回
		decisionType=decision[self.parent.getFullName()]["Horizontal"][0]()
		value=decision[self.parent.getFullName()]["Horizontal"][1]()
		motionCmd=command[self.parent.getFullName()]["motion"]()
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
		mws=sorted([Track2D(t) for t in self.parent.observables.at_p("/sensor/mws/track")],key=angle)
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
		decisionType=decision[self.parent.getFullName()]["Vertical"][0]()
		value=decision[self.parent.getFullName()]["Vertical"][1]()
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
			decisionType=decision[self.parent.getFullName()]["Throttle"][0]()
			value=decision[self.parent.getFullName()]["Throttle"][1]()
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
		expertTarget=Track3D(decision[self.parent.getFullName()]["Fire"][1])
		if(decision[self.parent.getFullName()]["Fire"][0]()):
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
				if(not expertTarget.truth in self.actionInfo.lastShotTimes):
					self.actionInfo.lastShotTimes[expertTarget.truth]=0.0
				if(len(self.shotIntervalTable)>1):
					shotInterval_ex=self.manager.getTime()-self.actionInfo.lastShotTimes[expertTarget.truth]
					unflattened_action[action_idx]=0
					for i in range(len(self.shotIntervalTable)-1,-1,-1):
						if(self.shotIntervalTable[i]<shotInterval_ex):
							unflattened_action[action_idx]=i
							break
					action_idx+=1
				if(len(self.shotThresholdTable)>1):
					r=self.calcRNorm(myMotion,expertTarget)
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
		#self.commands[self.parent.getFullName()]["weapon"]=command[self.parent.getFullName()]["weapon"]

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
		"""慣性座標系での絶対位置posに対応するピクセル座標(lon,lat)を返す。
		"""
		myMotion=self.ourMotion[0]
		if(self.image_relative_position):
			#自機位置が基準点
			rpos=pos-myMotion.pos
		else:
			#自陣営防衛ライン中央が基準点
			rpos=pos-self.teamOrigin.pos
		if(self.image_rotate):
			#慣性座標系からH座標系に回転
			rpos=myMotion.relPtoH(rpos)
		else:
			#慣性座標系から陣営座標系に回転
			rpos=self.teamOrigin.relPtoB(rpos)
		return self.rposToGrid(rpos)
	def gridToPos(self,lon,lat,alt):
		"""指定した高度においてピクセル座標(lon,lat)に対応する慣性座標系での絶対位置posを返す。
		"""
		myMotion=self.ourMotion[0]
		rpos=self.gridToRpos(lon,lat,alt)
		if(self.image_rotate):
			#H座標系から慣性座標系に回転
			rpos=myMotion.relHtoP(rpos)
		else:
			#陣営座標系から慣性座標系に回転
			rpos=self.teamOrigin.relBtoP(rpos)
		if(self.image_relative_position):
			#自機位置が基準点
			rpos=rpos+myMotion.pos
		else:
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

	def calcRHead(self,myMotion,track):
		#相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程(RHead)を返す。
		rt=track.posI()
		vt=track.velI()
		rs=myMotion.pos
		vs=myMotion.vel
		return self.parent.getRmax(rs,vs,rt,vt,0.0)
	def calcRTail(self,myMotion,track):
		#相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程(RTail)を返す。
		rt=track.posI()
		vt=track.velI()
		rs=myMotion.pos
		vs=myMotion.vel
		return self.parent.getRmax(rs,vs,rt,vt,np.pi)
	def calcRNorm(self,myMotion,track):
		#RTail→0、RHead→1として正規化した距離を返す。
		RHead=self.calcRHead(myMotion,track)
		RTail=self.calcRTail(myMotion,track)
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
