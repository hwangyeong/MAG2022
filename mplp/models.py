import torch
from torch import nn
import torch.nn.functional as F
from layers import (MLP, FeedForwardNet, GroupMLP, MultiHeadBatchNorm,
                    MultiHeadMLP)

################################################################
# SAGN model
class SAGN(nn.Module):
    def __init__(self, in_channels, hidden, out_feats, num_hops, n_layers, num_heads, weight_style="attention", alpha=0.5, focal="first",
                 hop_norm="softmax", dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, zero_inits=False, position_emb=False):
        super(SAGN, self).__init__()
        self._num_heads = num_heads
        self._in_feat0 = in_channels[0]
        self._hidden = hidden
        self._out_feats = out_feats
        self._weight_style = weight_style
        self._alpha = alpha
        self._hop_norm = hop_norm
        self._zero_inits = zero_inits
        self._focal = focal
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        # self.bn = nn.BatchNorm1d(hidden * num_heads)
        self.bn = MultiHeadBatchNorm(num_heads, hidden * num_heads)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.multihop_encoders = nn.ModuleList()
        self.position_emb = position_emb

        tot_hidden = 0
        for i in in_channels:
            # tot_hidden += j
            mlp = GroupMLP(i, hidden, hidden, num_heads, n_layers, dropout)
            self.multihop_encoders.append(mlp)

        # self.multihop_encoders = nn.ModuleList([GroupMLP(in_feats, hidden, hidden, num_heads, n_layers, dropout) for i in range(num_hops)])
        self.res_fc = nn.Linear(self._in_feat0, self._hidden * num_heads, bias=False)  # 给第一个原始属性做encoder
        
        if weight_style == "attention":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if position_emb:
            self.pos_emb = nn.Parameter(torch.FloatTensor(size=(num_hops, in_channels)))
        else:
            self.pos_emb = None
        
        self.post_encoder = GroupMLP(hidden, hidden, out_feats, num_heads, n_layers, dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for encoder in self.multihop_encoders:
            encoder.reset_parameters()
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if self._weight_style == "attention":
            if self._zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self.pos_emb is not None:
            nn.init.xavier_normal_(self.pos_emb, gain=gain)
        self.post_encoder.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats): # feat: [tensor, tensor, ... k-hop个]
        out = 0
        feats = [self.input_drop(feat) for feat in feats]
        if self.pos_emb is not None:
            feats = [f +self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = []
        for i in range(len(feats)):  # hidden: [tensor, tensor, ... k-hop个]
            hidden.append(self.multihop_encoders[i](feats[i]).view(-1, self._num_heads, self._hidden))
        
        a = None
        if self._weight_style == "attention": 
            if self._focal == "first": #选第几个作为求attention的目标
                focal_feat = hidden[0]
            if self._focal == "last":
                focal_feat = hidden[-1]
            if self._focal == "average":
                focal_feat = 0
                for h in hidden:
                    focal_feat += h
                focal_feat /= len(hidden)
                
            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)
            if self._hop_norm == "softmax":
                a = self.leaky_relu(astack)
                a = F.softmax(a, dim=-1)
            if self._hop_norm == "sigmoid":
                a = torch.sigmoid(astack)
            if self._hop_norm == "tanh":
                a = torch.tanh(astack)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]

        if self._weight_style == "uniform":
            for h in hidden:
                out += h / len(hidden)
        
        if self._weight_style == "exponent":
            for k, h in enumerate(hidden):
                out += self._alpha ** k * h

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)
        return out
        # return out, a.mean(1) if a is not None else None