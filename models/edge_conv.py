from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing, EdgeConv
from .embedding_layer import EmbeddingLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple



class Edge_Conv(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 in_features,
                 out_features,
                 embedding_out_features,
                 embedding_num_classes,
                 dropout=0,
                 model_epoch=2,
                 aggr="mean",
                 device="cpu"):
        super(Edge_Conv, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.dropout = dropout
        self.model_epoch = model_epoch
        # 先对值进行embedding
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)
        '''
        self.MessagePassingNN = nn.ModuleList([
            FiLMConv(in_channels=embedding_out_features, out_channels=out_features, num_relations=self.num_edge_types)
            for _ in range(self.model_epoch)
        ])
        '''
        self.MessagePassingNN = nn.ModuleList([
            nn.ModuleList([
                EdgeConv(
                   nn.Linear(out_features*2, out_features)) for _ in range(self.num_edge_types)
            ]) for j in range(self.model_epoch)
        ])



    def forward(self, x, edge_list: List[torch.tensor], **kwargs):
        # 要把edge_list进行拼接，然后还要制作edge_type,
        # edge_type : size(E*edge_list_nums) 序号从0开始。
        edge_type_list = []
        for e_i in range(len(edge_list)):
            edge = edge_list[e_i]
            edge_type_list.append(torch.ones(edge.shape[1]) * e_i)

        #edge_type = torch.cat(edge_type_list, dim=0)
        #mask = edge_type == 0
        #edge = torch.cat(edge_list, dim=1)

        x_embedding = self.value_embeddingLayer(x)

        last_node_states = x_embedding
        for m_epoch in range(self.model_epoch):
            out_list = []
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0:
                    last_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
                    out_list.append(self.MessagePassingNN[m_epoch][i](x=last_node_states, edge_index=edge))
            cur_node_states = torch.div(sum(out_list), len(edge_list))
            last_node_states = last_node_states + cur_node_states

        out = last_node_states
        return out