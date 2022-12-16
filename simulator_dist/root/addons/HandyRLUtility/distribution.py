# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
# 様々な種類の行動空間に対し、共通のインターフェースで確率分布等を計算するためのユーティリティ
# HandyRLでの使用を想定したインターフェースとなっている。

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from ASRCAISim1.addons.torch_truncnorm import TruncatedNormal

def getActionDistributionClass(space):
    if(isinstance(space,gym.spaces.Discrete)):
        return DiscreteActionDistribution
    elif(isinstance(space,gym.spaces.MultiDiscrete)):
        return MultiDiscreteActionDistribution
    elif(isinstance(space,gym.spaces.MultiBinary)):
        return MultiBinaryActionDistribution
    elif(isinstance(space,gym.spaces.Box)):
        return BoxActionDistribution
    elif(isinstance(space,gym.spaces.Tuple)):
        return TupleActionDistribution
    elif(isinstance(space,gym.spaces.Dict)):
        return DictActionDistribution

class ActionDistributionBase:
    """様々な行動空間に対して共通のインターフェースでサンプリングや学習を可能とするための基底クラス。
    """
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        self.space=space
        if(params is not None):
            if(isinstance(params,np.ndarray)):
                self.device=torch.Tensor().device
            else:
                self.device=params.device
        else:
            self.device=torch.Tensor().device
    def log_prob(self,actions):
        if(not isinstance(actions,torch.Tensor)):
            actions=torch.tensor(actions,device=self.device)
        ret=self.dist.log_prob(actions)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def sample(self,deterministic=False,exploration=None):
        raise NotImplementedError
    def entropy(self):
        ret=self.dist.entropy()
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    #def pack(self,unpacked):
    #    raise NotImplementedError
    def unpack_policy(self,packed):
        raise NotImplementedError
    def unpack_scalar(self,packed,keepDim=True):
        raise NotImplementedError
    @staticmethod
    def get_action_dim(space):
        raise NotImplementedError
    @classmethod
    def get_param_dim(cls,space):
        raise NotImplementedError
    @staticmethod
    def apply_mask(space, policy, mask):
        if(mask is not None):
            return policy - mask
    @staticmethod
    def getDefaultAction(space):
        return np.zeros_like(space.sample())
    @staticmethod
    def getDefaultLegalActions(space):
        raise NotImplementedError
    @classmethod
    def getInvalidActionMask(cls,space):
        return np.full([cls.get_param_dim(space)], 1e32)

class DiscreteActionDistribution(ActionDistributionBase):
    """Discreteな行動空間に対する分布
    入力はlogitsとする
    """
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.Discrete))
        super().__init__(space,params)
        self.ndim=self.space.n
        self.action_mask=None
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            if(legal_actions is not None):
                if(self.asNumpy):
                    self.action_mask = np.ones_like(params) * 1e32
                else:
                    self.action_mask = torch.ones_like(params) * 1e32
                self.action_mask[legal_actions==1] = 0
                params = params - self.action_mask
            if(self.asNumpy):
                params = torch.tensor(params)
        self.dist=dist.Categorical(logits=params, validate_args=validate_args)
    def sample(self,deterministic=False,exploration=None):
        if(deterministic):
            ret=self.dist.probs.argmax(dim=-1)
        else:
            if(exploration is None):
                ret=self.dist.sample()
            else:
                #ε-greedy
                if(np.random.random()<exploration):
                    ret=dist.Categorical(logits=torch.ones_like(self.dist.logits), validate_args=self.dist._validate_args).sample()
                else:
                    ret=self.dist.sample()
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def unpack_policy(self,packed):
        return [packed]
    def unpack_scalar(self,packed,keepDim=True):
        if(keepDim):
            return [packed.unsqueeze(-1)]
        else:
            return [packed]
    @staticmethod
    def get_action_dim(space):
        return 1
    @classmethod
    def get_param_dim(cls,space):
        assert(isinstance(space,gym.spaces.Discrete))
        return space.n
    @staticmethod
    def getDefaultLegalActions(space):
        return np.ones([space.n])

class MultiDiscreteActionDistribution(ActionDistributionBase):
    """MultiDiscreteな行動空間に対する分布
    入力はconcatenateされたlogitsとする
    """
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.MultiDiscrete))
        super().__init__(space,params,legal_actions)
        self.action_mask=None
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            splitted_params=self.unpack_policy(params)
            if(legal_actions is not None):
                self.action_mask=[]
                for i in range(len(splitted_params)):
                    p=splitted_params[i]
                    if(self.asNumpy):
                        mask = np.ones_like(p) * 1e32
                    else:
                        mask = torch.ones_like(p) * 1e32
                    mask[legal_actions[i]==1] = 0
                    splitted_params[i] = p - mask
                    self.action_mask.append(mask)
            if(self.asNumpy):
                splitted_params = [torch.tensor(p) for p in splitted_params]
                if(self.action_mask is not None):
                    self.action_mask=np.concatenate(self.action_mask,axis=-1)
            else:
                if(self.action_mask is not None):
                    self.action_mask=torch.cat(self.action_mask,dim=-1)
        self.dists=[dist.Categorical(logits=p, validate_args=validate_args) for p in splitted_params]
    def log_prob(self,actions):
        if(not isinstance(actions,torch.Tensor)):
            actions=torch.tensor(actions,device=self.device)
        splitted_actions=torch.split(actions, 1, -1)
        ret=torch.stack([d.log_prob(a.squeeze(-1)) for d,a in zip(self.dists,splitted_actions)],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def sample(self,deterministic=False,exploration=None):
        if(deterministic):
            ret=torch.stack([d.probs.argmax(dim=-1) for d in self.dists],dim=-1)
        else:
            if(exploration is None):
                ret=torch.stack([d.sample() for d in self.dists],dim=-1)
            else:
                #ε-greedy
                if(np.random.random()<exploration):
                    ret=torch.stack([
                        dist.Categorical(logits=torch.ones_like(d.logits), validate_args=d._validate_args).sample()
                        for d in self.dists
                    ],dim=-1)
                else:
                    ret=torch.stack([d.sample() for d in self.dists],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def entropy(self):
        ret=torch.stack([d.entropy() for d in self.dists],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def unpack_policy(self,packed):
        if(isinstance(packed,torch.Tensor)):
            return list(torch.split(packed, [s.n for s in self.space], -1))
        else:
            dims = [s.n for s in self.space]
            return np.split(packed, [np.sum(dims[:i+1]) for i in range(len(dims)-1)], -1)
    def unpack_scalar(self,packed,keepDim=True):
        if(isinstance(packed,torch.Tensor)):
            ret = list(torch.split(packed, 1, -1))
        else:
            dims = [s.n for s in self.space]
            ret = np.split(packed, [i for i in range(len(dims)-1)], -1)
        if(not keepDim):
            ret = [v.squeeze(-1) for v in ret]
        return ret
    @staticmethod
    def get_action_dim(space):
        return len(space.nvec)
    @classmethod
    def get_param_dim(cls,space):
        assert(isinstance(space,gym.spaces.MultiDiscrete))
        return np.sum(space.nvec)
    @staticmethod
    def getDefaultLegalActions(space):
        #return [list(range(s.n)) for s in space]
        return [np.ones([s.n]) for s in space]

class MultiBinaryActionDistribution(ActionDistributionBase):
    """MultiBinaryな行動空間に対する分布
    入力はconcatenateされたlogitsとする
    """
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.MultiBinary))
        super().__init__(space,params,legal_actions)
        self.action_mask=None
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            splitted_params=self.unpack_policy(params)
            if(legal_actions is not None):
                self.action_mask=[]
                for i in range(len(splitted_params)):
                    p=splitted_params[i]
                    if(self.asNumpy):
                        mask = np.ones_like(p) * 1e32
                    else:
                        mask = torch.ones_like(p) * 1e32
                    mask[legal_actions[i]==1] = 0
                    splitted_params[i] = p - mask
                    self.action_mask.append(mask)
            if(self.asNumpy):
                splitted_params = [torch.tensor(p) for p in splitted_params]
                if(self.action_mask is not None):
                    self.action_mask=np.concatenate(self.action_mask,axis=-1)
            else:
                if(self.action_mask is not None):
                    self.action_mask=torch.cat(self.action_mask,dim=-1)
        self.dists=[dist.Categorical(logits=p, validate_args=validate_args) for p in splitted_params]
    def log_prob(self,actions):
        if(not isinstance(actions,torch.Tensor)):
            actions=torch.tensor(actions,device=self.device)
        splitted_actions=torch.split(actions, 1, -1)
        ret=torch.stack([d.log_prob(a.squeeze(-1)) for d,a in zip(self.dists,splitted_actions)],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def sample(self,deterministic=False,exploration=None):
        if(deterministic):
            ret=torch.stack([d.probs.argmax(dim=-1) for d in self.dists],dim=-1)
        else:
            if(exploration is None):
                ret=torch.stack([d.sample() for d in self.dists],dim=-1)
            else:
                #ε-greedy
                if(np.random.random()<exploration):
                    ret=torch.stack([
                        dist.Categorical(logits=torch.ones_like(d.logits), validate_args=d._validate_args).sample()
                        for d in self.dists
                    ],dim=-1)
                else:
                    ret=torch.stack([d.sample() for d in self.dists],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def entropy(self):
        ret=torch.stack([d.entropy() for d in self.dists],dim=-1)
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def unpack_policy(self,packed):
        dims = [2] * self.space.n
        if(isinstance(packed,torch.Tensor)):
            return list(torch.split(packed, dims, -1))
        else:
            return np.split(packed, [np.sum(dims[:i+1]) for i in range(len(dims)-1)], -1)
    def unpack_scalar(self,packed,keepDim=True):
        if(isinstance(packed,torch.Tensor)):
            ret = list(torch.split(packed, 1, -1))
        else:
            ret = np.split(packed, [i for i in range(self.space.n-1)], -1)
        if(not keepDim):
            ret = [v.squeeze(-1) for v in ret]
        return ret
    @staticmethod
    def get_action_dim(space):
        return space.n
    @classmethod
    def get_param_dim(cls,space):
        assert(isinstance(space,gym.spaces.MultiBinary))
        return 2 * space.n
    @staticmethod
    def getDefaultLegalActions(space):
        return [np.ones([2])] * space.n

class BoxActionDistribution(ActionDistributionBase):
    """Boxな行動空間に対する分布
    truncated gaussianとして表現する
    mean=sigmoid(y1)*(high-low)+low
    stddev=(sigmoid(y2)*upperLimit+lowerLimit)*(high-low)
    a~truncnorm(mean,stddev)
    Flattenされたparamsが入るのでspaceのshapeに復元が必要
    """
    stddev_min = 1e-2 #標準偏差の下限値(探索の余地を残すため)
    stddev_scale = 3.0 #標準偏差のsigmoidにかかる係数≒上限値
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.Box))
        super().__init__(space,params,legal_actions)
        self.low=torch.tensor(self.space.low)
        self.high=torch.tensor(self.space.high)
        self.ndim=np.product(self.space.shape)

        self.action_mask=None
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            mean_, stddev_ = self.unpack_policy(params)
            if(self.asNumpy):
                mean_ = torch.tensor(mean_)
                stddev_ = torch.tensor(stddev_)
                self.action_mask = np.zeros_like(params)
            else:
                self.action_mask = torch.zeros_like(params)
            if(legal_actions is not None):
                self.low=legal_actions[0]
                self.high=legal_actions[1]
                if(isinstance(self.low,np.ndarray)):
                    self.low=torch.tensor(self.low)
                if(isinstance(self.high,np.ndarray)):
                    self.high=torch.tensor(self.high)
            mean_, stddev_, self.high, self.low = broadcast_all(mean_, stddev_, self.high, self.low)
            self.mean=torch.sigmoid(mean_)*(self.high-self.low)+self.low
            #self.stddev=torch.exp(stddev_)+1e-6
            self.stddev=(torch.sigmoid(stddev_)*type(self).stddev_scale+type(self).stddev_min)*(self.high-self.low)
        self.dist=TruncatedNormal(self.mean,self.stddev,self.low,self.high, validate_args=validate_args)
    def sample(self,deterministic=False,exploration=None):
        if(deterministic):
            ret=self.mean
        else:
            if(exploration is None):
                ret=self.dist.sample()
            else:
                #ε-greedy
                if(np.random.random()<exploration):
                    ret=dist.Uniform(self.low,self.high,validate_args=self.dist._validate_args).sample()
                    #ret=TruncatedNormal(self.mean,self.stddev+exploration*(self.high-self.low),self.low,self.high, validate_args=self.dist._validate_args).sample()
                else:
                    ret=self.dist.sample()
        ret=torch.clamp(ret,self.low,self.high) #just for numeric stability
        if(self.asNumpy):
            ret=ret.numpy()
        return ret
    def unpack_policy(self,packed):
        dims = [self.ndim, self.ndim]
        if(isinstance(packed,torch.Tensor)):
            return [r.unflatten(-1,self.space.shape) for r in torch.split(packed, dims, -1)]
        else:
            return [np.reshape(r,packed.shape[:-1]+self.space.shape) for r in np.split(packed, [np.sum(dims[:i+1]) for i in range(len(dims)-1)], -1)]
    def unpack_scalar(self,packed,keepDim=True):
        if(keepDim):
            return [packed]
        else:
            return [packed.squeeze(-1)]
    @staticmethod
    def get_action_dim(space):
        return np.product(space.shape)
    @classmethod
    def get_param_dim(cls,space):
        return np.product(space.shape) * 2
    @staticmethod
    def apply_mask(space, policy, mask):
        return policy
    @staticmethod
    def getDefaultAction(space):
        return space.low
    @staticmethod
    def getDefaultLegalActions(space):
        return [space.low,space.high]

class TupleActionDistribution(ActionDistributionBase):
    """Tupleな行動空間に対する分布
    """
    getActionDistributionClass=getActionDistributionClass
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.Tuple))
        super().__init__(space,params,legal_actions)
        self.dist_classes=[
            type(self).getActionDistributionClass(sub) for sub in self.space.spaces
        ]
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            dims=[cl.get_param_dim(sub) for cl, sub in zip(self.dist_classes, self.space.spaces)]
            if(self.asNumpy):
                splitted_params = np.split(params, [np.sum(dims[:i+1]) for i in range(len(dims)-1)], -1)
            else:
                splitted_params = torch.split(params, dims, -1)
        legal_actions = legal_actions if legal_actions is not None else [None for sub in self.space.spaces]
        self.dists = [cl(sub, p, l, validate_args=validate_args)
            for cl, sub, p, l in zip(self.dist_classes, self.space.spaces, splitted_params, legal_actions)]
        self.action_mask = [d.action_mask for d in self.dists]
    def log_prob(self,actions):
        return [d.log_prob(a) for d,a in zip(self.dists,actions)]
    def sample(self,deterministic=False,exploration=None):
        return [d.sample(deterministic,exploration) for d in self.dists]
    def entropy(self):
        return [d.entropy() for d in self.dists]
    def unpack_policy(self,packed):
        return sum([d.unpack_policy(p) for d, p in zip(self.dists, packed)], [])
    def unpack_scalar(self,packed,keepDim=True):
        return sum([d.unpack_scalar(p, keepDim) for d, p in zip(self.dists, packed)], [])
    @classmethod
    def get_action_dim(cls,space):
        return np.sum([
            cls.getActionDistributionClass(sub).get_action_dim(sub) for sub in space.spaces
        ])
    @classmethod
    def get_param_dim(cls,space):
        return np.sum([
            cls.getActionDistributionClass(sub).get_param_dim(sub) for sub in space.spaces
        ])
    @classmethod
    def apply_mask(cls,space, policy, mask):
        return [cls.getActionDistributionClass(sub).apply_mask(sub, p, m) for sub, p, m in zip(space.spaces, policy, mask)]
    @classmethod
    def getDefaultAction(cls,space):
        return [cls.getActionDistributionClass(sub).getDefaultAction(sub) for sub in space.spaces]
    @classmethod
    def getDefaultLegalActions(cls,space):
        return [cls.getActionDistributionClass(sub).getDefaultLegalActions(sub) for sub in space.spaces]
    @classmethod
    def getInvalidActionMask(cls,space):
        return [cls.getActionDistributionClass(sub).getInvalidActionMask(sub) for sub in space.spaces]

class DictActionDistribution(ActionDistributionBase):
    """Dictな行動空間に対する分布
    """
    getActionDistributionClass=getActionDistributionClass
    def __init__(self, space, params, legal_actions=None, validate_args=None):
        assert(isinstance(space,gym.spaces.Dict))
        super().__init__(space,params,legal_actions)
        self.dist_classes={k: type(self).getActionDistributionClass(sub)
            for k, sub in self.space.spaces.items()
        }
        if(params is not None):
            self.asNumpy=isinstance(params,np.ndarray)
            dims=[self.dist_classes[k].get_param_dim(sub) for k, sub in self.space.spaces.items()]
            if(self.asNumpy):
                splitted_params = np.split(params, [np.sum(dims[:i+1]) for i in range(len(dims)-1)], -1)
            else:
                splitted_params = torch.split(params, dims, -1)
            splitted_params = {k: v for k, v in zip(self.space.spaces.keys(), splitted_params)}
        legal_actions = legal_actions if legal_actions is not None else {k: None for k, sub in self.space.spaces.items()}
        self.dists={k: self.dist_classes[k](sub, splitted_params[k], legal_actions[k], validate_args=validate_args)
            for k, sub in self.space.spaces.items()
        }
        self.action_mask = {k: d.action_mask for k, d in self.dists.items()}
    def log_prob(self,actions):
        return {k: d.log_prob(actions[k]) for (k, d) in self.dists.items()}
    def sample(self,deterministic=False,exploration=None):
        return {k: d.sample(deterministic,exploration) for k, d in self.dists.items()}
    def entropy(self):
        return {k: d.entropy() for k, d in self.dists.items()}
    def unpack_policy(self,packed):
        return sum([d.unpack_policy(packed[k]) for k, d in self.dists.items()], [])
    def unpack_scalar(self,packed,keepDim=True):
        return sum([d.unpack_scalar(packed[k], keepDim) for k, d in self.dists.items()], [])
    @classmethod
    def get_action_dim(cls,space):
        return np.sum([
            cls.getActionDistributionClass(sub).get_action_dim(sub) for sub in space.spaces.values()
        ])
    @classmethod
    def get_param_dim(cls,space):
        return np.sum([
            cls.getActionDistributionClass(sub).get_param_dim(sub) for sub in space.spaces.values()
        ])
    @classmethod
    def apply_mask(cls,space, policy, mask):
        return {k: cls.getActionDistributionClass(sub).apply_mask(sub, policy[k], mask[k]) for k, sub in space.spaces.items()}
    @classmethod
    def getDefaultAction(cls,space):
        return {k: cls.getActionDistributionClass(sub).getDefaultAction(sub) for k, sub in space.spaces.items()}
    @classmethod
    def getDefaultLegalActions(cls,space):
        return {k: cls.getActionDistributionClass(sub).getDefaultLegalActions(sub) for k, sub in space.spaces.items()}
    @classmethod
    def getInvalidActionMask(cls,space):
        return {k: cls.getActionDistributionClass(sub).getInvalidActionMask(sub) for k, sub in space.spaces.items()}

class ActionDistributionClassGetter:
    def __init__(self,classes={}):
        self._classes=classes
        class CustomTupleActionDistribution(self._classes.get("Tuple",TupleActionDistribution)):
            pass
        class CustomDictActionDistribution(self._classes.get("Dict",DictActionDistribution)):
            pass
        self._customClassForTuple=CustomTupleActionDistribution
        self._customClassForDict=CustomDictActionDistribution
        self._customClassForTuple.getActionDistributionClass=self
        self._customClassForDict.getActionDistributionClass=self
    def __call__(self,space):
        if(isinstance(space,gym.spaces.Discrete)):
            return self._classes.get("Discrete",DiscreteActionDistribution)
        elif(isinstance(space,gym.spaces.MultiDiscrete)):
            return self._classes.get("MultiDiscrete",MultiDiscreteActionDistribution)
        elif(isinstance(space,gym.spaces.MultiBinary)):
            return self._classes.get("MultiBinary",MultiBinaryActionDistribution)
        elif(isinstance(space,gym.spaces.Box)):
            return self._classes.get("Box",BoxActionDistribution)
        elif(isinstance(space,gym.spaces.Tuple)):
            return self._customClassForTuple
        elif(isinstance(space,gym.spaces.Dict)):
            return self._customClassForDict

def getDefaultLegalActions(space):
    return getActionDistributionClass(space).getDefaultLegalActions(space)
