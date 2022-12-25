// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <boost/uuid/uuid.hpp>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Sensor.h>
#include <Eigen/CXX11/Tensor>
#include <ASRCAISim1/FlightControllerUtility.h>

DECLARE_CLASS_WITHOUT_TRAMPOLINE(R4AgentSample01S,SingleAssetAgent)
	/*1機につき1つのAgentを割り当てる、分散方式での行動判断モデルの実装例。
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
				struct ActionInfo
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
	*/
	public:
	//modelConfigで設定するもの
	//  observation spaceの設定
	std::map<std::string,int> maxTrackNum; //vectorのobservationとして考慮する彼我の戦闘機の航跡数
	std::map<std::string,int> maxMissileNum; //vectorのobservationとして考慮する彼我の誘導弾の航跡数
	double horizontalNormalizer; //水平方向の位置・距離の正規化のための除数
	double verticalNormalizer; //高度方向の正規化のための除数
	double fgtrVelNormalizer; //機体速度の正規化のための除数
	double mslVelNormalizer; //誘導弾速度の正規化のための除数	
	//    2次元画像としてのobservation
	bool use_image_observation; //戦域を2次元画像で表現したobservationを使用するかどうか
	int image_longitudinal_resolution; //前後方向の解像度
	int image_lateral_resolution; //左右方向の解像度
	double image_front_range; //前方の描画範囲(m)
	double image_back_range; //後方の描画範囲(m)
	double image_side_range; //側方の描画範囲(m)
	int image_horizon; //軌跡を描画する秒数
	int image_interval; //軌跡を描画する間隔
	bool image_rotate; //画像化の基準座標系。trueの場合自機正面を-x軸とするように回転。falseの場合、慣性座標系そのまま(赤青の陣営による反転は行う)。
	bool image_relative_position; //画像化の基準位置。trueの場合自機位置、falseの場合自陣営防衛ライン中央とする。
	//    実数値ベクトルとしてのobservation
	bool use_vector_observation; //戦域を実数値ベクトルで表現したobservationを使用するかどうか
	bool include_last_action; //前回の行動に関する情報を含めるかどうか
	std::vector<int> vector_past_points; //キーフレームとしてobservationに追加する過去の時刻

	//  action spaceの設定
	//    spaceの形式に関する設定
	bool flatten_action_space; //上記のaction spaceをDiscreteにflattenするかどうか。falseにした場合はMultiDiscreteとして扱う。
	//    左右旋回に関する設定
	bool dstAz_relative; //旋回の原点に関する設定。trueの場合は自機正面、falseの場合は自陣営の進行方向が原点となる。
	Eigen::VectorXd turnTable; //通常時の目標方位(deg)テーブル(右が正)
	Eigen::VectorXd evasion_turnTable; //MWS作動時の目標方位(deg)テーブル(検出した誘導弾を背にした方位を0とし、右が正)
	bool use_override_evasion; //MWS作動時の目標方位テーブルを専用のものに置き換えるかどうか
	//    上昇・下降に関する設定
	Eigen::VectorXd pitchTable; //目標ピッチ角(deg)テーブル(上昇が正)
	Eigen::VectorXd altTable; //目標高度(m)テーブル(最寄りの基準高度からの高度差で指定)
	double refAltInterval; //基準高度グリッドの間隔(m)
	bool use_altitude_command; //ピッチ角と高度のどちらを使用するか。trueでピッチ、falseで高度を使用する。
	//    加減速に関する設定
	Eigen::VectorXd accelTable; //目標加速度テーブル
	bool always_maxAB; //常時maxABするかどうか。
	//    射撃に関する設定
	Eigen::VectorXd shotIntervalTable; //同一目標に対する射撃間隔(秒)テーブル
	Eigen::VectorXd shotThresholdTable; //射撃閾値テーブル(RTailを0、RHeadを1とした線形空間での閾値)
	bool use_Rmax_fire; //既存の射程テーブルを用いて射撃判定を行うかどうか。

	//  行動制限に関する設定
	//    高度制限に関する設定(ピッチ角をactionにしている場合のみ有効)
	double altMin; //下限高度
	double altMax; //上限高度
	AltitudeKeeper altitudeKeeper; //ピッチ角制限を計算するための高度制御則。サンプルではFighterControllerのモデルと同じものを同じ設定で使用している。
	//    場外制限に関する設定
	double dOutLimitRatio; //場外防止を行う閾値。RulerのdOutに対する割合で指定。
	double dOutLimitScale; //場外防止の復元力に関する係数。
	//    同時射撃数の制限に関する設定
	int maxSimulShot; //自身が同時に射撃可能な誘導弾数の上限
	//  下限速度の制限に関する設定
	double minimumV; //低速域からの回復を開始する速度
	double minimumRecoveryV; //低速域からの回復を終了する速度
	double minimumRecoveryDstV; //低速域からの回復時に設定する目標速度

	//内部変数
	//  observationに関するもの
	py::object _observation_space; //観測空間を表すgym.spaces.Space
	std::vector<MotionState> ourMotion; //味方各機のMotionState
	std::vector<nl::json> ourObservables; //味方各機のobservables
	std::vector<Track3D> lastTrackInfo; //自陣営が捉えている彼機航跡
	std::vector<nl::json> msls; //自陣営の誘導弾(のobservables)
	//    2次元画像としてのobservation
	int numChannels;//チャネル数
	Eigen::Tensor<float,3> image_buffer; //画像データバッファ[ch×lon×lat]
	Eigen::Tensor<double,3> image_buffer_coords; //ピクセル座標(lon,lat)と戦域座標(x,y)のマッピング[2×lon×lat]
	struct InstantInfo{
		//軌跡描画用のフレームデータ
		public:
		std::vector<Eigen::Vector3d> friend_pos; //味方位置
		std::vector<Eigen::Vector3d> friend_msl_pos; //味方誘導弾位置
		std::vector<Eigen::Vector3d> enemy_pos; //彼機位置
	};
	std::vector<InstantInfo> image_past_data; //軌跡描画用の過去データのバッファ
	//    実数値ベクトルとしてのobservation
	int last_action_dim; //前回の行動の次元
	int friend_dim; //味方諸元1機あたりの次元
	int enemy_dim; //彼機諸元1機あたりの次元
	int friend_msl_dim; //味方誘導弾諸元1発あたりの次元
	int vector_single_dim; //1時刻分のvector observationの次元
	Eigen::VectorXf last_action_obs; //前回の行動を格納するバッファ
	Eigen::MatrixXf friend_temporal; //味方諸元を格納するバッファ
	Eigen::MatrixXf enemy_temporal; //彼機諸元を格納するバッファ
	Eigen::MatrixXf friend_msl_temporal; //味方誘導弾諸元を格納するバッファ
	std::vector<Eigen::VectorXf> vector_past_data; //過去のvectorデータのバッファ

	//  actionに関するもの
	py::object _action_space; //行動空間を表すgym.spaces.Space
	Eigen::VectorXi actionDims; //actionの各要素の次元
	int totalActionDim; //actionの総次元数

	struct ActionInfo{
		//    機体に対するコマンドを生成するための変数をまとめた構造体
		public:
		ActionInfo();
		Eigen::Vector3d dstDir; //目標進行方向
		double dstAlt; //目標高度
		bool velRecovery; //下限速度制限からの回復中かどうか
		bool asThrottle; //加減速についてスロットルでコマンドを生成するかどうか
		bool keepVel; //加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
		double dstThrottle; //目標スロットル
		double dstV; //目標速度
		bool launchFlag; //射撃するかどうか
		Track3D target; //射撃対象
		std::map<boost::uuids::uuid,double> lastShotTimes; //各Trackに対する直前の射撃時刻
	};
	ActionInfo actionInfo;
	
	//Ruler、陣営に関するもの
	double dOut; //戦域中心から場外ラインまでの距離
	double dLine; //戦域中心から防衛ラインまでの距離

	//陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス。
	//MotionStateを使用しても良いがクォータニオンを経由することで浮動小数点演算に起因する余分な誤差が生じるため、もし可能な限り対称性を求めるのであればこの例のように符号反転で済ませたほうが良い。
	//ただし、機体運動等も含めると全ての状態量に対して厳密に対称なシミュレーションとはならないため、ある程度の誤差は生じる。
	class TeamOrigin{
		public:
		bool isEastSider;
		Eigen::Vector3d pos;//原点
		TeamOrigin();
		TeamOrigin(bool isEastSider_,double dLine);
		~TeamOrigin();
    	Eigen::Vector3d relBtoP(const Eigen::Vector3d& v) const;//陣営座標系⇛慣性座標系
    	Eigen::Vector3d relPtoB(const Eigen::Vector3d& v) const;//慣性座標系⇛陣営座標系
	};
	TeamOrigin teamOrigin;

	public:
	R4AgentSample01S(const nl::json& modelConfig_,const nl::json& instanceConfig_);
	virtual ~R4AgentSample01S();
	virtual void validate() override;
	virtual py::object observation_space() override;
	virtual py::object makeObs() override;
	virtual void extractObservables();
	virtual double calcDistanceMargin(const MotionState& myMotion, const nl::json& myObservables);
	virtual void makeImageObservation();
	virtual Eigen::VectorXf makeVectorObservation();
	virtual py::object action_space() override;
	virtual void deploy(py::object action) override;
	virtual void control() override;
	virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command) override;
	virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command);
	//MultiDiscrete⇔Discreteの相互変換
	int flatten_action(const Eigen::VectorXi& original_action,const Eigen::VectorXi& original_shape);
	Eigen::VectorXi unflatten_action(const int& flattened_action,const Eigen::VectorXi& original_shape);
	//画像化するための関数
	bool isInside(const int& lon, const int& lat); //ピクセル座標が画像内かどうか
	Eigen::Vector2i rposToGrid(const Eigen::Vector3d& dr); //相対位置→ピクセル
	Eigen::Vector3d gridToRpos(const int& lon, const int& lat, const double& alt); //ピクセル+高度→相対位置
	Eigen::Vector2i posToGrid(const Eigen::Vector3d& pos); //絶対位置→ピクセル
	Eigen::Vector3d gridToPos(const int& lon, const int& lat, const double& alt); //ピクセル+高度→絶対位置
	void plotPoint(const Eigen::Vector3d& pos,const int& ch, const float& value); //点を描画
	void plotLine(const Eigen::Vector3d& pBegin,const Eigen::Vector3d& pEnd,const int& ch, const float& value); //線分を描画
	//射程計算に関する関数
	double calcRHead(const MotionState& myMotion,const Track3D& track);//相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程
	double calcRTail(const MotionState& myMotion,const Track3D& track);//相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程
	double calcRNorm(const MotionState& myMotion,const Track3D& track);//正規化した射程
};

void exportR4AgentSample01S(py::module &m);
