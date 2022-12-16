# ===== Original Version (As a part of trainer.py of HandyRL) =====
# Copyright (c) 2020 DeNA Co., Ltd.
#
# ===== Modified Version (As a separated RecurrentBlock interface) =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
import numpy as np
import torch
import torch.nn as nn

from ASRCAISim1.addons.HandyRLUtility.util import map_r, bimap_r, trimap_r
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass

class RecurrentBlock(nn.Module):
    """
    HandyRL改において、リカレント部分の逐次処理を内部で実行し、
    外部からは非リカレント部分と同様に全体をバッチとして一括で入力できるようにするためのラッパー
    HandyRLが逐次処理を必要とする理由は、「推論を実行しない時刻」が存在したときにバッチから除外せずに、
    学習時に該当時刻をマスクしてstateを更新せず次時刻に渡すことで対応されているためである。
    """
    def __init__(self, module):
        super().__init__()
        self.module=module
        assert(isinstance(self.module,nn.RNNBase))
        assert(self.module.batch_first)
        dummy=torch.zeros([1,1,self.module.input_size])
        with torch.no_grad():
            _,hidden=self.module(dummy)
        self.hidden_shape=map_r(hidden, lambda h: h.transpose(0,1))

    def forward(self,inputs,states=None,seq_len=None,mask=None):
        #通常のRNNに加えて、バッチ中の途中の時刻に「推論しない時刻」の存在を許容するもの
        if(states is None):
            states=map_r(self.init_hidden([inputs.shape[0]]), lambda h: h.to(inputs.device))
        if(seq_len is None):
            #単時刻の推論の場合、shape=[B,D]で時刻の次元が無い形で入力されることを前提とする
            x,state_out=self.module(inputs.unsqueeze(1),map_r(states, lambda s: s.transpose(0,1).contiguous()))#[B×D_core]
            x_out = x[:, 0, ...].squeeze(1)
            state_out = map_r(state_out, lambda s: s.transpose(0,1).contiguous())
        else:
            #複数時刻にわたる推論の場合、shape=[B×T,D]の形で入力されることを前提とする
            hs = states
            if(mask is not None):
                hs = bimap_r(hs, map_r(mask, lambda m:m[:,0]), lambda h, m: h * m)
            x_out=[]
            x_in=torch.reshape(inputs,(-1,seq_len)+inputs.shape[1:])
            for t in range(seq_len):
                x,state_out=self.module(x_in[:,t,...].unsqueeze(1),map_r(hs, lambda h: h.transpose(0,1).contiguous()))#[B,1,D]
                x_out.append(x)
                if(mask is not None):
                    state_out = map_r(state_out, lambda s: s.transpose(0,1).contiguous())
                    hs = trimap_r(hs, state_out, map_r(mask, lambda m:m[:,t]), lambda h, nh, m: h * (1 - m) + nh * m)
                state_out=hs
            x_out = torch.cat(x_out,dim=1).flatten(0,1)
        return x_out,state_out

    def init_hidden(self, batch_size=None):
        if batch_size is None:  # for inference
            return map_r(self.hidden_shape, lambda h: np.zeros(h.shape[1:], dtype=np.float32))
        else:  # for training
            return map_r(self.hidden_shape, lambda h: torch.zeros((*batch_size,*h.shape[1:]), dtype=torch.float32))
