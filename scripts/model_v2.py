"""
model_v2.py implements the transformer model using torch, not dgl in a graph view, which can be more efficient.
"""
import torch as th
import numpy as np
from torch.nn import LayerNorm
import torch.nn.init as INIT
import copy
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl
import math
from torch.nn import Linear
from transformer_torch import *

torch.set_default_tensor_type('torch.FloatTensor')


# class Encoder(nn.Module):
#     def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout):
#         '''
#         :param d_model: Embedding Size
#         :param d_ff: FeedForward dimension
#         :param d_k: dimension of K(=Q)
#         :param d_v: dimension of  V
#         :param n_heads:  number of heads in Multi-Head Attention
#         :param n_layers: number of Encoder  Layer
#         :param dropout: dropout ratio
#         '''
#         super(Encoder, self).__init__()
#         self.src_emb = nn.Embedding(6, d_model)
#         self.pos_emb = PositionalEncoding(d_model)
#         self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])
#
#     def forward(self, enc_inputs):
#         '''
#         enc_inputs: [batch_size, src_len]
#         '''
#         enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
#         enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
#         enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
#         enc_self_attns = []
#         for layer in self.layers:
#             # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
#             enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
#             enc_self_attns.append(enc_self_attn)
#         return enc_outputs, enc_self_attns
#
#
# # test the Encoder class
# enc_inputs = torch.tensor([[1, 2, 3, 4, 3, 0], [1, 2, 3, 5, 4, 0]]).cuda()
# d_model = 512  # Embedding Size
# d_ff = 2048  # FeedForward dimension
# d_k = d_v = 128  # dimension of K(=Q), V
# n_layers = 6  # number of Encoder of Decoder Layer
# n_heads = 8  # number of heads in Multi-Head Attention
# dropout = 0.2
# model = Encoder(d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout).cuda()
# enc_outputs, enc_self_attns = model(enc_inputs)


# # test the Encoder_ class on dgl graph objects
# g1 = dgl.DGLGraph()
# g1.add_nodes(50)
# g1.ndata.update({'x': torch.rand(g1.number_of_nodes(), 75)})
# g1.ndata.update({'pad': torch.ones(g1.number_of_nodes())})
# # add padding nodes
# g1.add_nodes(50)
#
# g2 = dgl.DGLGraph()
# g2.add_nodes(80)
# g2.ndata.update({'x': torch.rand(g2.number_of_nodes(), 75)})
# g2.ndata.update({'pad': torch.ones(g2.number_of_nodes())})
# # add padding nodes
# g2.add_nodes(20)
#
# bg = dgl.batch([g1, g2])
# device = torch.device('cuda')
# bg.to(device)
# in_feat_gp = 75
# model = Encoder_(in_feat_gp, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout).cuda()
# enc_outputs, enc_self_attns = model(bg)  # [batch_size, src_len, d_model]
# enc_outputs = enc_outputs.view(-1, d_model)
# # ndata_x = bg.ndata['x'].view(2, -1, 75)
# # ndata_pad = bg.ndata['pad'].view(2, 100)


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h


class DTIConvGraph12Layer(nn.Module):
    def __init__(self, dim, vertex_update='gru', bias=False, drop_out=0.1):
        super(DTIConvGraph12Layer, self).__init__()
        self.dim = dim
        self.vertex_update = vertex_update
        assert self.vertex_update in ['inner', 'gru', 'mlp'], "only ['inner', 'gru', 'mlp'] were supported for " \
                                                              "vertex_update parameter"
        self.bias = bias
        self.drop_out = drop_out
        self.w1 = nn.Sequential(
            Linear(self.dim * 3, self.dim, self.bias),
            nn.LeakyReLU())
        self.w2 = nn.Sequential(
            Linear(self.dim, 1, self.bias),
            nn.LeakyReLU())
        self.w3 = nn.Sequential(
            Linear(self.dim, self.dim, self.bias),
            nn.LeakyReLU())
        if self.vertex_update == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(2 * self.dim, self.dim, self.bias),
                nn.LeakyReLU(),
                nn.Linear(self.dim, self.dim, self.bias),
                nn.LeakyReLU())
        elif self.vertex_update == 'gru':
            self.gru = nn.GRUCell(self.dim, self.dim)
        self.leaky_relu = nn.LeakyReLU()

    def edge_update(self, edges):
        e = (self.w1(torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=-1)))
        return {'e': self.leaky_relu(e)}

    def node_update(self, nodes):
        if self.vertex_update == 'inner':
            # return {'h': self.leaky_relu(nodes.data['m'] * nodes.data['h'])}
            return {'h': F.relu(nodes.data['m'] * nodes.data['h'])}
        elif self.vertex_update == 'gru':
            # return {'h': self.leaky_relu(self.gru(nodes.data['m'], nodes.data['h']))}
            return {'h': F.relu(self.gru(nodes.data['m'], nodes.data['h']))}
        else:
            # return {'h': self.leaky_relu(self.mlp(torch.cat([nodes.data['m'], nodes.data['h']], dim=-1)))}
            return {'h': F.relu(self.mlp(torch.cat([nodes.data['m'], nodes.data['h']], dim=-1)))}

    def message_func(self, edges):
        return {'m': edges.data['a'] * self.w3(edges.data['e'])}

    def forward(self, bg, nfeats, efeats):
        with bg.local_scope():
            bg.ndata['h'] = nfeats
            bg.edata['e'] = efeats
            # 更新边状态
            bg.apply_edges(self.edge_update)
            # 求解注意力系数
            logits = self.leaky_relu(self.w2(bg.edata['e']))
            bg.edata['a'] = edge_softmax(bg, logits)
            # 消息传递
            bg.update_all(self.message_func, fn.sum('m', 'm'))
            # bg.ndata['m'] = self.leaky_relu(bg.ndata['m'])
            bg.ndata['m'] = F.elu(bg.ndata['m'])
            # 节点状态更新
            bg.apply_nodes(self.node_update)
            return bg.ndata['h'], bg.edata['e']


class DTIGraph12PoolLayer(nn.Module):
    def __init__(self, dim, drop_out, bias, vertex_update):
        self.dim = dim
        self.drop_out = drop_out
        self.bias = bias
        self.vertex_update = vertex_update

        super(DTIGraph12PoolLayer, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * self.dim, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU()
        )
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.dim, self.dim, self.bias),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim, self.dim, self.bias),
            nn.LeakyReLU())

    def forward(self, g, node_feats, g_feats, get_node_weight=True):
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.leaky_relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            # message aggregation
            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.leaky_relu(g_repr)

            if get_node_weight:
                if self.vertex_update == 'inner':
                    return F.leaky_relu(context * g_feats), g.ndata['a']
                elif self.vertex_update == 'gru':
                    return F.leaky_relu(self.gru(context, g_feats)), g.ndata['a']
                else:
                    return F.leaky_relu(self.mlp(torch.cat([context, g_feats], dim=-1))), g.ndata['a']

            else:
                if self.vertex_update == 'inner':
                    return F.leaky_relu(context * g_feats)
                elif self.vertex_update == 'gru':
                    return F.leaky_relu(self.gru(context, g_feats))
                else:
                    return F.leaky_relu(self.mlp(torch.cat([context, g_feats], dim=-1)))


class DTIGraph12Readout(nn.Module):
    def __init__(self, dim, pooling_layers=2, drop_out=0.1, vertex_update='gru', bias=False):
        super(DTIGraph12Readout, self).__init__()
        self.dim = dim
        self.pooling_layers = pooling_layers
        self.drop_out = drop_out
        self.final_g_feats = 0
        self.vertex_update = vertex_update
        self.bias = bias
        self.graph_unit_transform = nn.Sequential(nn.Dropout(self.drop_out), nn.Linear(2 * self.dim, dim))
        self.readouts = nn.ModuleList()

        for _ in range(self.pooling_layers):
            self.readouts.append(DTIGraph12PoolLayer(self.dim, self.drop_out, self.bias, self.vertex_update))

    def forward(self, bg, node_feats, get_node_weight=True):
        self.final_g_feats = 0
        with bg.local_scope():
            bg.ndata['hv'] = node_feats
            g_feats = torch.cat([dgl.sum_nodes(bg, 'hv'), dgl.max_nodes(bg, 'hv')], dim=-1)
            g_feats = self.graph_unit_transform(g_feats)

            if get_node_weight:
                node_weights = []
                for readout in self.readouts:
                    g_feats, node_weights_t = readout(bg, node_feats, g_feats, get_node_weight)
                    self.final_g_feats = self.final_g_feats + g_feats
                    node_weights.append(node_weights_t)
                return self.final_g_feats, node_weights
            else:
                for readout in self.readouts:
                    g_feats = readout(bg, node_feats, g_feats, get_node_weight)
                    self.final_g_feats = self.final_g_feats + g_feats
                return self.final_g_feats


class DTIConvGraph12_NL(nn.Module):
    def __init__(self, node_dim, edge_dim, dim, vertex_update='gru', drop_out=0.1, num_layers=2, bias=False):
        super(DTIConvGraph12_NL, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dim = dim
        self.vertex_update = vertex_update
        self.drop_out = drop_out
        self.num_layers = num_layers
        self.bias = bias

        self.node_project = nn.Sequential(Linear(self.node_dim, self.dim),
                                          nn.LeakyReLU())
        self.edge_project = nn.Sequential(Linear(self.edge_dim, self.dim),
                                          nn.LeakyReLU())
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers):
            self.gnn_layers.append(DTIConvGraph12Layer(dim=self.dim, vertex_update=self.vertex_update, bias=self.bias,
                                                       drop_out=self.drop_out))

    def forward(self, bg, node_feats, edge_feats):
        self.sum_node_feats = 0
        node_feats = self.node_project(node_feats)
        edge_feats = self.edge_project(edge_feats)
        for gnn in self.gnn_layers:
            node_feats, edge_feats = gnn(bg, node_feats, edge_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class DTIGraph3EdgePoolLayer(nn.Module):
    def __init__(self, dim, drop_out, bias, vertex_update):
        self.dim = dim
        self.drop_out = drop_out
        self.bias = bias
        self.vertex_update = vertex_update

        super(DTIGraph3EdgePoolLayer, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * self.dim, 1),
            nn.LeakyReLU()
        )
        self.project_edges = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU()
        )
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.dim, self.dim, self.bias),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim, self.dim, self.bias),
            nn.LeakyReLU())

    def forward(self, g, edge_feats, g_feats, get_edge_weight=True):
        with g.local_scope():
            g.edata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_edges(g, F.leaky_relu(g_feats)), edge_feats], dim=1))
            g.edata['a'] = dgl.softmax_edges(g, 'z')
            g.edata['hv'] = self.project_edges(edge_feats)

            # message aggregation
            g_repr = dgl.sum_edges(g, 'hv', 'a')
            context = F.leaky_relu(g_repr)

            if get_edge_weight:
                if self.vertex_update == 'inner':
                    return F.leaky_relu(context * g_feats), g.data['a']
                elif self.vertex_update == 'gru':
                    return F.leaky_relu(self.gru(context, g_feats)), g.edata['a']
                else:
                    return F.leaky_relu(self.mlp(torch.cat([context, g_feats], dim=-1))), g.edata['a']

            else:
                if self.vertex_update == 'inner':
                    return F.leaky_relu(context * g_feats)
                elif self.vertex_update == 'gru':
                    return F.leaky_relu(self.gru(context, g_feats))
                else:
                    return F.leaky_relu(self.mlp(torch.cat([context, g_feats], dim=-1)))


class DTIGraph3EdgeReadout(nn.Module):
    def __init__(self, dim, pooling_layers=2, drop_out=0.1, vertex_update='gru', bias=False):
        super(DTIGraph3EdgeReadout, self).__init__()
        self.dim = dim
        self.pooling_layers = pooling_layers
        self.drop_out = drop_out
        self.final_g_feats = 0
        self.vertex_update = vertex_update
        self.bias = bias
        self.graph_unit_transform = nn.Sequential(nn.Dropout(self.drop_out), nn.Linear(2 * self.dim, dim))
        self.readouts = nn.ModuleList()

        for _ in range(self.pooling_layers):
            self.readouts.append(DTIGraph3EdgePoolLayer(self.dim, self.drop_out, self.bias, self.vertex_update))

    def forward(self, bg, edge_feats, get_edge_weight=True):
        self.final_g_feats = 0
        with bg.local_scope():
            bg.edata['hv'] = edge_feats
            g_feats = torch.cat([dgl.sum_edges(bg, 'hv'), dgl.max_edges(bg, 'hv')], dim=-1)
            g_feats = self.graph_unit_transform(g_feats)

            if get_edge_weight:
                node_weights = []
                for readout in self.readouts:
                    g_feats, node_weights_t = readout(bg, edge_feats, g_feats, get_edge_weight)
                    self.final_g_feats = self.final_g_feats + g_feats
                    node_weights.append(node_weights_t)
                return self.final_g_feats, node_weights
            else:
                for readout in self.readouts:
                    g_feats = readout(bg, edge_feats, g_feats, get_edge_weight)
                    self.final_g_feats = self.final_g_feats + g_feats
                return self.final_g_feats


# # test the DTIGraph3EdgeReadout class
# import torch as th
# g1 = dgl.DGLGraph()
# g1.add_nodes(3)
# g1.add_edges([0, 1, 2], [2, 1, 0])
# g1.ndata['h'] = th.rand(3, 4)
# g1.edata['e'] = th.rand(3, 10)
#
# g2 = dgl.DGLGraph()
# g2.add_nodes(5)
# g2.add_edges([0, 1, 2, 3], [3, 2, 1, 0])
# g2.ndata['h'] = th.rand(5, 4)
# g2.edata['e'] = th.rand(4, 10)
# bg = dgl.batch([g1, g2])
#
# readout = DTIGraph3EdgeReadout(dim=10, pooling_layers=2, drop_out=0.1, vertex_update='gru', bias=False)
# edge_feats = bg.edata['e']
# results = readout(bg, edge_feats, get_edge_weight=False)
# results, _ = readout(bg, edge_feats, get_edge_weight=True)


class AttentiveGRU1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPGNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()

        self.gnn = ModifiedAttentiveFPGNNV2(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return sum_node_feats


# the the model
# model = make_model()
# bg1 = dgl.batch([dataset.gp[0], dataset.gp[1]])
# res = model(bg1)
#
# attp = ModifiedAttentiveFPPredictorV2(94, 20)
# bg = dgl.batch([dataset.gl[0], dataset.gl[1]])
# res1 = attp(bg, bg.ndata['x'], bg.edata['e'])


class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3_Shenyu(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3_Shenyu, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            # bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class DTIConvGraph3Layer_Shenyu(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer_Shenyu, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3_Shenyu(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))



class DTIConvGraph3_V2(nn.Module):
    """
    Do not consider the distance information in graph3
    """

    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3_V2, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e_m': self.mpl(edges.data['m'])}

    def forward(self, bg, atom_feats):
        bg.ndata['x'] = atom_feats
        # bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('x', 'x', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e_m']


class DTIConvGraph3Layer_V2(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(DTIConvGraph3Layer_V2, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3_V2(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    # def forward(self, bg, atom_feats, bond_feats):
    def forward(self, bg, atom_feats):
        new_feats = self.grah_conv(bg, atom_feats)
        return self.bn_layer(self.dropout(new_feats))


class DistAwareConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DistAwareConv, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())
        # the threshold in the cutoff function and initial as 12.0
        self.R = nn.Parameter(torch.tensor([12.0]).float())
        self.lambd = nn.Parameter(torch.tensor([1.0]).float())

    def weights(self, dis_features):
        weights = torch.exp(-torch.pow(dis_features - self.R, 2) * self.lambd)
        return weights

    def forward(self, bg, atom_feats, dis_features):
        bg.ndata['x'] = atom_feats
        # bg.edata['dis'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('x', 'x', 'm'))
            outputs = self.mpl(bg.edata['m']) * self.weights(dis_features)
        return outputs


class DistAwareConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(DistAwareConvLayer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DistAwareConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, dis_feats):
        new_feats = self.grah_conv(bg, atom_feats, dis_feats)
        return self.bn_layer(self.dropout(new_feats))


# test the DistAwareConv and DistAwareConvLayer
# import dgl
# g = dgl.DGLGraph()
# g.add_nodes(100)
# g.add_edges(range(0, 50), range(50, 100))
# g.ndata.update({'x': torch.rand((g.number_of_nodes(), 75))})
# g.edata.update({'dis': 10+torch.rand((g.number_of_edges(), 1))})
#
# bg = dgl.batch([g, g, g])
#
# layer1 = DistAwareConv(75, 128)
# outputs1 = layer1(bg, bg.ndata['x'], bg.edata['dis'])
#
# layer2 = DistAwareConv(75, 128)
# outputs2 = layer2(bg, bg.ndata['x'], bg.edata['dis'])


class EdgeWeightAndSum_V2(nn.Module):
    """
    change the nn.Tanh() function to nn.Sigmoid()
    """

    def __init__(self, in_feats):
        super(EdgeWeightAndSum_V2, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, edge_feats):
        g.edata['e'] = edge_feats
        g.edata['w'] = self.atom_weighting(g.edata['e'])
        weights = g.edata['w']
        h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum, weights


class NodeWeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(NodeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, node_feats):
        g.ndata['n_f'] = node_feats
        g.ndata['w'] = self.atom_weighting(g.ndata['n_f'])
        weights = g.ndata['w']
        h_g_sum = dgl.sum_nodes(g, 'n_f', 'w')
        return h_g_sum, weights



class IGN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, task_type):
        super(IGN, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size,
                                                        dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size + 1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightAndSum_V2(outdim_g3)
        # self.readout = DTIGraph3EdgeReadout(dim=outdim_g3, pooling_layers=1, drop_out=dropout, vertex_update='gru',
        #                                     bias=False)

        self.task_type = task_type

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts, weights = self.readout(bg3, bond_feats3)
        if self.task_type == 'regression':
            return self.FC(readouts), weights
        # classification
        else:
            return torch.sigmoid(self.FC(readouts)), weights


class IGN_Shenyu(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, task_type):
        super(IGN_Shenyu, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size,
                                                        dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer_Shenyu(graph_feat_size*2 + 1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightAndSum_V2(outdim_g3)
        # self.readout = DTIGraph3EdgeReadout(dim=outdim_g3, pooling_layers=1, drop_out=dropout, vertex_update='gru',
        #                                     bias=False)

        self.task_type = task_type

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts, weights = self.readout(bg3, bond_feats3)
        if self.task_type == 'regression':
            return self.FC(readouts), weights
        # classification
        else:
            return torch.sigmoid(self.FC(readouts)), weights



class IGN_DDG(nn.Module):
    """
    This module was used as predicting the binding affinity change upon the mutation of
    protein in ligand-protein complexes
    """

    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(IGN_DDG, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size,
                                                        dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size + 1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3 + graph_feat_size, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightAndSum_V2(outdim_g3)

        self.readout2 = NodeWeightAndSum(graph_feat_size)

    def forward(self, bg_w, bg_m, bg3_w, bg3_m, bg_env_w, bg_env_m):
        # wild type forward (interaction pattern)
        atom_feats_w = bg_w.ndata.pop('h')
        bond_feats_w = bg_w.edata.pop('e')

        atom_feats_w = self.cov_graph(bg_w, atom_feats_w, bond_feats_w)

        bond_feats3_w = bg3_w.edata.pop('e')
        bond_feats3_w = self.noncov_graph(bg3_w, atom_feats_w, bond_feats3_w)
        readouts_w, _ = self.readout(bg3_w, bond_feats3_w)

        # wild type forward (mutant environment)
        atom_feats2_w = bg_env_w.ndata.pop('h')
        bond_feats2_w = bg_env_w.edata.pop('e')
        atom_feats2_w = self.cov_graph(bg_env_w, atom_feats2_w, bond_feats2_w)
        readouts2_w, _ = self.readout2(bg_env_w, atom_feats2_w)

        # mutation type forward (interaction pattern)
        atom_feats_m = bg_m.ndata.pop('h')
        bond_feats_m = bg_m.edata.pop('e')

        atom_feats_m = self.cov_graph(bg_m, atom_feats_m, bond_feats_m)

        bond_feats3_m = bg3_m.edata.pop('e')
        bond_feats3_m = self.noncov_graph(bg3_m, atom_feats_m, bond_feats3_m)
        readouts_m, _ = self.readout(bg3_m, bond_feats3_m)

        # mutation type forward (mutant environment)
        atom_feats2_m = bg_env_m.ndata.pop('h')
        bond_feats2_m = bg_env_m.edata.pop('e')
        atom_feats2_m = self.cov_graph(bg_env_m, atom_feats2_m, bond_feats2_m)
        readouts2_m, _ = self.readout2(bg_env_m, atom_feats2_m)

        diff = torch.cat([readouts_m - readouts_w, readouts2_m - readouts2_w], dim=-1)
        return self.FC(diff)


class IGN_New(nn.Module):
    "The covalent graph convolution module was implemented by our own (DTIConvGraph12_NL) in the class"

    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, vertex_update='gru'):
        super(IGN_New, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = DTIConvGraph12_NL(node_dim=node_feat_size, edge_dim=edge_feat_size, dim=graph_feat_size,
                                           vertex_update=vertex_update, drop_out=dropout, num_layers=num_layers,
                                           bias=False)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size + 1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightAndSum_V2(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts, weights = self.readout(bg3, bond_feats3)
        return self.FC(readouts), weights
