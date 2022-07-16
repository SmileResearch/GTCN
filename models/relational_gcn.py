from torch_geometric.nn import GATConv, SAGPooling, GCNConv, RGCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple

# 本地库
from .embedding_layer import EmbeddingLayer  # from .*** import表示从当前包中导入
from .mlp_layer import MLPLayer


class Relational_GCN(MessagePassing):

    def __init__(
        self,
        num_edge_types,
        in_features,
        out_features,
        embedding_out_features,
        embedding_num_classes,
        dropout=0,
        add_self_loops=False,
        bias=True,
        aggr="mean",
        device="cpu",
    ):
        super(Relational_GCN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.dropout = dropout
        # 先对值进行embedding
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)
        self.MessagePassingNN = nn.ModuleList([
            RGCNConv(in_channels=embedding_out_features, out_channels=out_features, num_relations=self.num_edge_types)
            for _ in range(8)
        ])

    def forward(self, x, edge_list: List[torch.tensor], batch_map: torch.Tensor, **kwargs):

        x_embedding = self.value_embeddingLayer(x)

        edge_relation = []
        for index, edge in enumerate(edge_list):
            temp_relation_edge = torch.ones(edge.shape[1], device=self.device, dtype=torch.long) * index
            # print(edge.shape)
            # print(temp_relation_edge.shape)
            # print(temp_relation_edge)
            edge_relation.append(temp_relation_edge)
        edge_relation = torch.concat(edge_relation, dim=0)

        edge = torch.concat(edge_list, dim=1)

        last_node_states = x_embedding
        last_node_states = F.dropout(last_node_states, self.dropout, training=self.training)

        for i in range(8):
            cur_node_states = self.MessagePassingNN[i](x=last_node_states, edge_index=edge, edge_type=edge_relation)
            cur_node_states = F.relu(cur_node_states)
            last_node_states = torch.mean(torch.stack([cur_node_states, last_node_states], dim=0), dim=0)

        out = last_node_states
        return out