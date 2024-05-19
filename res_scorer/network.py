import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from typing import List

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0) # init the bias of last layer of MLP to 0

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2) #(1,3,1024)
        return self.encoder(inputs)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layer_names = ['self', 'cross']
        self.layers=nn.ModuleList([
            AttentionalPropagation(feature_dim, num_heads = 4)
            for _ in range(len(self.layer_names))])
    
    def forward(self,feats0,feats1):
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'cross':
                src0, src1 = feats1, feats0
            else: 
                src0, src1 = feats0, feats1
            delta0, delta1 = layer(feats0, src0), layer(feats1, src1)
            desc0, desc1 = (feats0 + delta0), (feats1 + delta1)
        return desc0, desc1


class Scorer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.kps_en = KeypointEncoder(32,[32])
        self.gnn = AttentionalGNN(feature_dim=32)
        self.final_mlp = nn.Conv1d(32, 32, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1))

    def forward(self, keys0, keys1, feats0, feats1):
        coors_e0 = self.kps_en(keys0) #(b,32,1024)
        coors_e1 = self.kps_en(keys1)

        feats0 = feats0.transpose(1, 2) + coors_e0
        feats1 = feats1.transpose(1, 2) + coors_e1

        gfeats0, gfeats1 = self.gnn(feats0,feats1)

        mfeats0, mfeats1 = self.final_mlp(gfeats0), self.final_mlp(gfeats1) #(b,32,1024)

        feats = torch.cat((mfeats0,mfeats1),dim=1) #(b,64,1024)
        feats = self.pool(feats) #(b,64,1)
        feats = feats.view(feats.size(0),-1)
        feats = self.fc(feats)
        return(feats)