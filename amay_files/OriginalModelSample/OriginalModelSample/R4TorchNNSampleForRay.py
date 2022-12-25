# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""ray RLlibで学習するためのtorch.nn.Moduleのサンプル。
R4AgentSample01S/Mのobservation,actionに対応したものとなっている。
"""
import copy
import logging
import numpy as np
from typing import Dict, Optional, Union
import gym

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.recurrent_net import LSTMWrapper
from ray.rllib.models.torch.torch_modelv2 import ModelV2,TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from .GenericTorchModelUtil import GenericLayers

MODEL_DEFAULTS: ModelConfigDict = {
    "use_lstm": True,
    "lstm_cell_size": 256,
    "fcnet_hiddens":[256],
    "pos_conv": {
        "layers": [
            ["Conv2d",{"kernel_size": 7,"out_channels": 64,"padding": 3,"stride": 2}],
            ["BatchNorm2d",{}],
            ["ReLU",{}],
            ["ResidualBlock",{
                "layers":[
                    ["Conv2d",{"kernel_size": 3,"out_channels": 64,"padding": 1}],
                    ["BatchNorm2d",{}],
                    ["ReLU",{}],
                    ["Conv2d",{"kernel_size": 3,"out_channels": 64,"padding": 1}],
                    ["BatchNorm2d",{}]
                ]}],
            ["ReLU",{}],
            ["Conv2d",{"kernel_size": 4,"out_channels": 128,"padding": 1,"stride": 2}],
            ["BatchNorm2d",{}],
            ["ReLU",{}],
            ["ResidualBlock",{
                "layers":[
                    ["Conv2d",{"kernel_size": 3,"out_channels": 128,"padding": 1}],
                    ["BatchNorm2d",{}],
                    ["ReLU",{}],
                    ["Conv2d",{"kernel_size": 3,"out_channels": 128,"padding": 1}],
                    ["BatchNorm2d",{}]
                ]}],
            ["ReLU",{}],
            ["Conv2d",{"kernel_size": 4,"out_channels": 256,"padding": 1,"stride": 2}],
            ["BatchNorm2d",{}],
            ["ReLU",{}],
            ["ResidualBlock",{
                "layers":[
                    ["Conv2d",{"kernel_size": 3,"out_channels": 256,"padding": 1}],
                    ["BatchNorm2d",{}],
                    ["ReLU",{}],
                    ["Conv2d",{"kernel_size": 3,"out_channels": 256,"padding": 1}],
                    ["BatchNorm2d",{}]
                ]}],
            ["ReLU",{}],
            ["AdaptiveAvgPool2d",{"output_size": 1}],
            ["Flatten",{}]
        ]
    },
    "dense": {
        "layers": [
            ["Linear",{"out_features": 256}],
            ["BatchNorm1d",{}],
            ["ReLU",{}],
            ["ResidualBlock",{
                "layers":[
                    ["Linear",{"out_features": 256}],
                    ["BatchNorm1d",{}]
                ]}],
            ["ReLU",{}],
            ["ResidualBlock",{
                "layers":[
                    ["Linear",{"out_features": 256}],
                    ["BatchNorm1d",{}]
                ]}]
        ]
    },
    "merge": {
        "layers": [
            ["ReLU",{}]
        ]
    },
    "value":{
        "layers": [
            ["Linear",{"out_features": 64}],
            ["ReLU",{}],
            ["Linear",{"out_features": 64}],
            ["ReLU",{}]
        ]
    },
    "action":{
        "layers": [
            ["Linear",{"out_features": 128}],
            ["ReLU",{}],
            ["Linear",{"out_features": 128}],
            ["ReLU",{}]
        ]
    }
}

class R4TorchNNSampleForRay(TorchModelV2, nn.Module):
    """Wrapperでなく、これ自身が中核
    """
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str,
                 **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        custom_model_config=copy.deepcopy(MODEL_DEFAULTS)
        custom_model_config.update(**kwargs)
        orig_space=getattr(obs_space, "original_space", obs_space)
        if(isinstance(orig_space,gym.spaces.Dict)):
            self.use_image_obs=True
            image_space=orig_space['image']
            self.use_vector_obs=True
            vector_space=orig_space['vector']
        elif(isinstance(orig_space,gym.spaces.Box)):
            if(len(orig_space.shape)==3):
                self.use_image_obs=True
                image_space=orig_space
                self.use_vector_obs=False
            elif(len(orig_space.shape)==1):
                self.use_vector_obs=True
                vector_space=orig_space
                self.use_image_obs=False
            else:
                raise ValueError("Invalid observation space. {}".format(orig_space))
        else:
            raise ValueError("Invalid observation space. {}".format(orig_space))

        self.core_dim=0
        if(self.use_image_obs):
            #conv branch
            cfg=custom_model_config['pos_conv']
            cfg['input_shape']=image_space.shape
            self.pos_conv=GenericLayers(cfg)
            self.core_dim+=self.pos_conv.output_shape[-1]

        if(self.use_vector_obs):
            #dense branch
            cfg=custom_model_config['dense']
            cfg['input_shape']=vector_space.shape
            self.dense=GenericLayers(cfg)
            self.core_dim+=self.dense.output_shape[-1]

        #merge branch
        cfg=custom_model_config['merge']
        cfg['input_shape']=[self.core_dim]
        self.merge=GenericLayers(cfg)
        self.merged_dim=self.merge.output_shape[-1]
        self.use_lstm = custom_model_config.get("use_lstm",False)
        if(self.use_lstm):
            in_space =gym.spaces.Box(
                float("-inf"),
                float("inf"),
                shape=(self.merged_dim, ),
                dtype=np.float32)
            self.lstm_cell_size = custom_model_config.get("lstm_cell_size",64)
            self.fcnet_hiddens = custom_model_config.get("fcnet_hiddens",[])
            class BranchClass(LSTMWrapper,FullyConnectedNetwork):
                pass
            BranchClass._wrapped_forward = FullyConnectedNetwork.forward
            self.core_recurrent_block=BranchClass(
                in_space,
                action_space,
                self.merged_dim,
                {
                    "fcnet_hiddens": self.fcnet_hiddens,
                    "_time_major":False,
                    "lstm_cell_size":self.lstm_cell_size,
                    "lstm_use_prev_action":False,
                    "lstm_use_prev_reward":False
                },
                "core_recurrent_block"
            )

        #出力層
        #value branch
        cfg=copy.deepcopy(custom_model_config['value'])
        cfg['input_shape']=[self.merged_dim]
        cfg['layers']=cfg['layers']+[['Linear',{'out_features':1}]]
        self.value_head=GenericLayers(cfg)
        #action branch
        cfg=copy.deepcopy(custom_model_config['action'])
        cfg['input_shape']=[self.merged_dim]
        cfg['layers']=cfg['layers']+[['Linear',{'out_features':self.num_outputs}]]
        self.action_head=GenericLayers(cfg)

        if(self.use_lstm):
            self.view_requirements = self.core_recurrent_block.view_requirements
            self.view_requirements["obs"].space = self.obs_space

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs=input_dict[SampleBatch.OBS]

        if(self.use_image_obs and self.use_vector_obs):
            #conv branch
            x1=self.pos_conv(obs['image'])#[B×Ch×Lon×Lat]→[B×D_image]

            #dense branch
            x2=self.dense(obs['vector'])#[B×D_vector]

            #merge branch
            x3=torch.cat([x1,x2],dim=1)#[B×D_core]
            x4=self.merge(x3)
        elif(self.use_image_obs):
            #conv branch
            x1=self.pos_conv(obs)#[B×Ch×Lon×Lat]→[B×D_image]

            #merge branch
            x4=self.merge(x1)
        else:#self.use_vector_obs
            #dense branch
            x2=self.dense(obs)#[B×D_vector]
            #merge branch
            x4=self.merge(x2)

        if(self.use_lstm):
            input_dict["obs_flat"]=input_dict["obs"]=x4
            x4,state_out=self.core_recurrent_block(input_dict,state,seq_lens)#[B×D_core]
        else:
            state_out=state

        #action branch
        action_logits=self.action_head(x4)

        #value branch
        self._value_base_features=x4

        return action_logits,state_out

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        if(self.use_lstm):
            return self.core_recurrent_block.get_initial_state()
        else:
            return []

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_base_features is not None, "must call forward() first"
        x=self._value_base_features
        value=self.value_head(x)
        return value.squeeze(1)

ModelCatalog.register_custom_model("R4TorchNNSampleForRay",R4TorchNNSampleForRay)
