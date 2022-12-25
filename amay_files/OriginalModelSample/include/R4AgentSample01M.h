// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <boost/uuid/uuid.hpp>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Sensor.h>
#include <Eigen/CXX11/Tensor>
#include <ASRCAISim1/FlightControllerUtility.h>

DECLARE_CLASS_WITHOUT_TRAMPOLINE(R4AgentSample01M,Agent)
	/*編隊全体で1つのAgentを割り当てる、中央集権方式での行動判断モデルの実装例。
	モデルの内容はほぼR4AgentSample01Sに準拠しているが、
	* observationの自機と味方機を区別していた部分を「自陣営」として一纏めに
	* imageの描画座標系を自陣営座標系に限定して回転を無効化
	* vectorの彼機航跡に対する射程情報を味方全機分格納
	* actionを全機分の行動とする
	の変更を加えている。
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
	Eigen::MatrixXf last_action_obs; //前回の行動を格納するバッファ
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
	std::vector<ActionInfo> actionInfos;
	
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
	R4AgentSample01M(const nl::json& modelConfig_,const nl::json& instanceConfig_);
	virtual ~R4AgentSample01M();
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
	double calcRHead(std::shared_ptr<PhysicalAssetAccessor> parent,const MotionState& myMotion,const Track3D& track);//相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程
	double calcRTail(std::shared_ptr<PhysicalAssetAccessor> parent,const MotionState& myMotion,const Track3D& track);//相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程
	double calcRNorm(std::shared_ptr<PhysicalAssetAccessor> parent,const MotionState& myMotion,const Track3D& track);//正規化した射程
};

void exportR4AgentSample01M(py::module &m);
