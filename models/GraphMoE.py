# from cgitb import reset
# from turtle import forward, shape
# from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch
import math
# import torch.nn as nn
import numpy as np
import copy
import numpy as np
import torch
import os
import torch
import os
import numpy as np


# def interpolate(tensor, index, target_size, mode = 'nearest', dim = 0):
#     source_length = tensor.shape[dim]
#     if source_length > target_size:
#         raise AttributeError('no need to interpolate')
#     if dim == -1:
#         new_tensor = torch.zeros((*tensor.shape[:-1], target_size),dtype=tensor.dtype, device=tensor.device)
#     if dim == 0:
#         new_tensor = torch.zeros((target_size, *tensor.shape[1:], ),dtype=tensor.dtype, device=tensor.device)
#     scale = target_size // source_length
#     reset = target_size % source_length
#     new_index = index
#     new_tensor[new_index, :] = tensor
#     new_tensor[:new_index[0], :] = tensor[0,:].unsqueeze(0)
#     for i in range(source_length-1):
#         new_tensor[new_index[i]:new_index[i+1] , :] = tensor[i,:].unsqueeze(0)
#     new_tensor[new_index[i+1] :,:] = tensor[i+1,:].unsqueeze(0)
#     return new_tensor


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, h, A):
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))
        return h


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=None):
        super().__init__()
        intermediate_size = 4*hidden_size
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class ScaleDotProductAttention(nn.Module):
    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, mask=None, e=1e-12):
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)
        score = (q @ k_t) / math.sqrt(c)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.dropout(self.softmax(score))
        return score, k


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class GraphCrossAttention(nn.Module):
    def __init__(self, hidden_size=None,num_attention_heads=4, args=None):
        super().__init__()
        self.args = args
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = 0.2
        self.text_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.vision_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.vision_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.vision_and_text_k_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def forward(
            self,
            text_hidden_states: torch.Tensor,
            vision_hidden_states: torch.Tensor,
            output_attentions: bool = False,
            current_layer: int = None,
            text2vision_hidden_states: torch.Tensor = None,
    ):
        bsz, tgt_len, embed_dim = text_hidden_states.size()
        query_states = self.text_q_proj(text_hidden_states) * self.scale
        key_states = self._shape(self.vision_k_proj(vision_hidden_states), -1, bsz)
        value_states = self._shape(self.vision_v_proj(vision_hidden_states), -1, bsz)
        if text2vision_hidden_states is not None:
            key_states = self._shape(
                self.vision_and_text_k_proj(torch.cat((text2vision_hidden_states, text2vision_hidden_states), dim=-1)),
                -1,
                bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_tgt_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_tgt_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_tgt_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_tgt_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_tgt_len)
        else:
            attn_weights_reshaped = None
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_weights, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value =\
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class RelationalMemory(nn.Module):
    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model
        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())
        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)
    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory
    def forward(self, input, memory):
        memory = F.normalize(memory, p=2, dim=-1)
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input], 1)
        v = torch.cat([memory, input], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)
        gates = self.W(input) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        return next_memory


class GraphMoE(nn.Module):
    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF",  batch_norm=True):
        super(GraphMoE, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model=="MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        self.n_experts = 3
        self.attention = ScaleDotProductAttention(window_size*input_size)
        self.self_attn1 = GraphCrossAttention(hidden_size)
        self.self_attn2 = GraphCrossAttention(hidden_size)
        self.self_attn3 = GraphCrossAttention(hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.layer_norm4 = nn.LayerNorm(hidden_size)
        self.mlp1 = CLIPMLP(hidden_size)
        self.mlp2 = CLIPMLP(hidden_size)
        self.mlp3 = CLIPMLP(hidden_size)
        # self.activation_fn = quick_gelu
        self.expert_router = nn.Linear(self.n_experts * hidden_size, self.n_experts)
        self.notinitial = False
        self.memory = None
        # self.kd_loss_fnt = nn.KLDivLoss(reduction='batchmean')
        self.rm = RelationalMemory(num_slots=window_size, d_model=self.n_experts * hidden_size)
    def forward(self, x):
        full_shape = x.shape
        graph,_ = self.attention(x)
        self.graph = graph
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h1, _ = self.rnn(x)
        h2 = h1.reshape((full_shape[0], full_shape[1], h1.shape[1], h1.shape[2]))
        h2 = self.gcn(h2, graph)
        h1_norm = self.layer_norm1(h1)
        h2_norm = self.layer_norm2(h2)
        attentional_hidden_states1, attn_weights = self.self_attn1(
            text_hidden_states=h1_norm,
            vision_hidden_states=h2_norm,
            current_layer=1,
        )
        h3 = self.gcn(h2, graph)
        h3_norm = self.layer_norm3(h3)
        attentional_hidden_states2, attn_weights = self.self_attn2(
            text_hidden_states=h1_norm,
            vision_hidden_states=h3_norm,
            current_layer=1,
        )
        h4 = self.gcn(h3, graph)
        h4_norm = self.layer_norm4(h4)
        attentional_hidden_states3, attn_weights = self.self_attn3(
            text_hidden_states=h1_norm,
            vision_hidden_states=h4_norm,
            current_layer=1,
        )
        attentional_hidden_states1 = self.layer_norm1(attentional_hidden_states1)
        attentional_hidden_states2 = self.layer_norm2(attentional_hidden_states2)
        attentional_hidden_states3 = self.layer_norm3(attentional_hidden_states3)
        outputs1 = self.mlp1(attentional_hidden_states1)
        outputs2 = self.mlp2(attentional_hidden_states2)
        outputs3 = self.mlp3(attentional_hidden_states3)
        expert_router_input = torch.cat((attentional_hidden_states1, attentional_hidden_states2, attentional_hidden_states3,), dim=-1)
        if not self.notinitial :
            memory = self.rm.init_memory(expert_router_input.size(0)).to(expert_router_input)
            self.notinitial = True
            self.memory = memory
        self.memory = self.rm(expert_router_input, self.memory).clone().detach()
        expert_weights = self.expert_router(self.memory)
        expert_weights = torch.softmax(expert_weights, dim=-1)
        outputs1= torch.stack((outputs1,), dim=2)
        outputs2= torch.stack((outputs2,), dim=2)
        outputs3= torch.stack((outputs3,), dim=2)
        input_hidden_states = torch.cat((
                                         outputs1, outputs2, outputs3,), dim=-2)
        experts_out = torch.einsum('bsno,bsn->bsno', input_hidden_states, expert_weights)
        moe_fusion = torch.sum(experts_out, dim=2)
        moe_fusion = moe_fusion.view(full_shape[0], full_shape[1], full_shape[2],
    self.rnn.hidden_size,)
        h = moe_fusion.reshape((-1, moe_fusion.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0],-1])
        log_prob = log_prob.mean(dim=1)
        return log_prob
    def get_graph(self):
        return self.graph
    # def locate(self, x, ):
    #     full_shape = x.shape
    #     graph, _ = self.attention(x)
    #     self.graph = graph
    #     x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    #     h,_ = self.rnn(x)
    #     h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
    #     h = self.gcn(h, graph)
    #     h = h.reshape((-1,h.shape[3]))
    #     x = x.reshape((-1,full_shape[3]))
    #     a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
    #     log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
    #     return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))

'''
class test(nn.Module):
    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF", batch_norm=True):
        super(test, self).__init__()
        if model=="MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,activation='tanh', mode='zero')
        self.attention = ScaleDotProductAttention(window_size*input_size)
    def forward(self, x, ):
        return self.test(x, ).mean()
    def test(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((full_shape[0]*full_shape[1], full_shape[2], full_shape[3]))
        x = x.reshape((-1,full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape([full_shape[0],full_shape[1],-1])
        log_prob = log_prob.mean(dim=1)
        return log_prob
    def locate(self, x, ):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        x = x.reshape((-1,full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2])
        log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
        return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))
'''