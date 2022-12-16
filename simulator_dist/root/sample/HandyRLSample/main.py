# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
# 
# * Discrete以外のaction_spaceに対応
# * MatchMakerによる行動判断モデルの動的選択に対応
# * Imitationに対応
# * Ape-X型のε-greedyに対応
# * SummaryWriterによるTensorboard形式のログ出力に対応
# * turn_basedな環境に非対応
# * evalモードに非対応

import sys
import yaml
from ASRCAISim1.addons.HandyRLUtility.model import DummyInternalModel
import OriginalModelSample
from OriginalModelSample.R4TorchNNSampleForHandyRL import R4TorchNNSampleForHandyRL
from ASRCAISim1.addons.MatchMaker.BVRMatchMaker import BVRMatchMaker, BVRMatchMonitor
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass

custom_classes={
    # models
    "R4TorchNNSampleForHandyRL": R4TorchNNSampleForHandyRL,
    "DummyInternalModel": DummyInternalModel,
    # match maker
    "BVRMatchMaker": BVRMatchMaker,
    "BVRMatchMonitor": BVRMatchMonitor,
    # action distribution class getter
    "actionDistributionClassGetter": getActionDistributionClass
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please set config yaml path.')
        exit(1)
    with open(sys.argv[1]) as f:
        args = yaml.safe_load(f)
    #print(args)
    args["train_args"]["custom_classes"] = custom_classes

    args["train_args"]["match_maker_class"] = args["train_args"].get("match_maker_class", "BVRMatchMaker")
    args["train_args"]["match_monitor_class"] = args["train_args"].get("match_monitor_class", "BVRMatchMonitor")

    if len(sys.argv) < 3:
        print('Please set mode of HandyRL.')
        exit(1)

    mode = sys.argv[2]

    if mode == '--train' or mode == '-t':
        from handyrl.train import train_main as main
        main(args)
    elif mode == '--train-server' or mode == '-ts':
        from handyrl.train import train_server_main as main
        main(args)
    elif mode == '--worker' or mode == '-w':
        from handyrl.worker import worker_main as main
        main(args, sys.argv[3:])
    else:
        print('Not found mode %s.' % mode)
