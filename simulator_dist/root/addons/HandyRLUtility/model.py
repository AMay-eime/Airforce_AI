# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import copy
from gym import spaces
import torch
import torch.nn as nn

class ModelBase(nn.Module):
    """ASRCAISim1のHandyRLサンプルを使用する際のNNモデルの基底クラス。
    出力のキー'policy'には各action要素の出力をconcatしたものを与える。
    """
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = ac_space
        self.action_dist_class = action_dist_class
        self.action_dim = self.action_dist_class.get_param_dim(ac_space) #=出力ノード数
        self.model_config = copy.deepcopy(model_config)
    def forward(self, obs, state, seq_len=None, mask=None):
        raise NotImplementedError
        """
        return {
            'policy': p,
            'value': 0.0,
            'return': 0.0
        }
        """

def getBatchSize(obs,space):
    if(isinstance(space,spaces.Dict)):
        k=next(iter(space))
        return getBatchSize(obs[k],space[k])
    elif(isinstance(space,spaces.Tuple)):
        return  getBatchSize(obs[0],space[0])
    else:
        return obs.shape[0]

class DummyInternalModel(ModelBase):
    """ルールベースモデル等、ASRCAISim1シミュレータ内部で行動が計算されるタイプの
    Agentを使用する際のダミーモデル
    """
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        super().__init__(obs_space, ac_space, action_dist_class, model_config)
    def forward(self, obs, state, seq_len = None, mask = None):
        B=getBatchSize(obs,self.observation_space)
        p = torch.ones([B,self.action_dim],dtype=torch.float32)
        return {
            'policy': p,
            'value': torch.zeros([B,1],dtype=torch.float32),
            'return': torch.zeros([B,1],dtype=torch.float32)
        }
