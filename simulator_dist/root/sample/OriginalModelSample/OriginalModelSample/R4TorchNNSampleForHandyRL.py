# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""HandyRLで学習するためのtorch.nn.Moduleのサンプル。
R4AgentSample01S/Mのobservation,actionに対応したものとなっている。
"""
import copy
import gym
import numpy as np
import torch
import torch.nn as nn

from ASRCAISim1.addons.HandyRLUtility.model import ModelBase
from ASRCAISim1.addons.HandyRLUtility.RecurrentBlock import RecurrentBlock
from .GenericTorchModelUtil import GenericLayers

MODEL_DEFAULTS = {
    "use_lstm": False,
    "lstm_cell_size": 64,
    "lstm_num_layers": 3,
    "lstm_dropout": 0.2,
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
    "return":{
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

class R4TorchNNSampleForHandyRL(ModelBase):
    def __init__(self, obs_space, ac_space, action_dist_class, model_config):
        super().__init__(obs_space, ac_space, action_dist_class, model_config)
        self.model_config=copy.deepcopy(MODEL_DEFAULTS)
        self.model_config.update(copy.deepcopy(model_config))

        if(isinstance(obs_space,gym.spaces.Dict)):
            self.use_image_obs=True
            image_space=obs_space['image']
            self.use_vector_obs=True
            vector_space=obs_space['vector']
        elif(isinstance(obs_space,gym.spaces.Box)):
            if(len(obs_space.shape)==3):
                self.use_image_obs=True
                image_space=obs_space
                self.use_vector_obs=False
            elif(len(obs_space.shape)==1):
                self.use_vector_obs=True
                vector_space=obs_space
                self.use_image_obs=False
            else:
                raise ValueError("Invalid observation space. {}".format(obs_space))
        else:
            raise ValueError("Invalid observation space. {}".format(obs_space))

        self.core_dim=0
        if(self.use_image_obs):
            #conv branch
            cfg=self.model_config['pos_conv']
            cfg['input_shape']=image_space.shape
            self.pos_conv=GenericLayers(cfg)
            self.core_dim+=self.pos_conv.output_shape[-1]

        if(self.use_vector_obs):
            #dense branch
            cfg=self.model_config['dense']
            cfg['input_shape']=vector_space.shape
            self.dense=GenericLayers(cfg)
            self.core_dim+=self.dense.output_shape[-1]

        #merge branch
        cfg=self.model_config['merge']
        cfg['input_shape']=[self.core_dim]
        self.merge=GenericLayers(cfg)
        self.merged_dim=self.merge.output_shape[-1]
        self.use_lstm = self.model_config.get("use_lstm",False)
        if(self.use_lstm):
            self.lstm_cell_size = self.model_config.get("lstm_cell_size",64)
            self.lstm_num_layers = self.model_config.get("lstm_num_layers",1)
            self.lstm_dropout = self.model_config.get("lstm_dropout",0.1)
            self.core_recurrent_block=RecurrentBlock(nn.LSTM(
                self.merged_dim,
                self.lstm_cell_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                dropout=self.lstm_dropout
            ))
            self.merged_dim=self.lstm_cell_size

        #value branch
        cfg=self.model_config['value']
        cfg['input_shape']=[self.merged_dim]
        cfg['layers'].append(['Linear',{'out_features':1}])
        self.value_head=GenericLayers(cfg)
        #return branch
        cfg=self.model_config['return']
        cfg['input_shape']=[self.merged_dim]
        cfg['layers'].append(['Linear',{'out_features':1}])
        self.return_head=GenericLayers(cfg)

        #action branch
        cfg=self.model_config['action']
        cfg['input_shape']=[self.merged_dim]
        cfg['layers'].append(['Linear',{'out_features':self.action_dim}])
        self.action_head=GenericLayers(cfg)

    def forward(self,obs,state,seq_len=None,mask=None):
        if(self.use_image_obs and self.use_vector_obs):
            #conv branch
            x1=self.pos_conv(obs['image'])#[B×Ch×Lon×Lat]→[B×D_image]

            #dense branch
            x2=self.dense(obs['vector'])#[B×D_dense]

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
            x4,state_out=self.core_recurrent_block(x4,state,seq_len,mask)

        #action branch
        action_logits=self.action_head(x4)

        #value branch
        v=self.value_head(x4)
        r=self.return_head(x4)

        ret={
            "policy": action_logits,
            'value': v,
            'return': r
        }
        if(self.use_lstm):
            ret["hidden"]=state_out
        return ret

    def init_hidden(self, batch_size=None):
        if(self.use_lstm):
            return self.core_recurrent_block.init_hidden(batch_size)
        else:
            return None
