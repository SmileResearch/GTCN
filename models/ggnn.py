from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple

# 本地库
from .embedding_layer import EmbeddingLayer # from .*** import表示从当前包中导入
from .mlp_layer import MLPLayer


class GGNN(MessagePassing):
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
        super(GGNN, self).__init__(aggr=aggr)
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
            MLPLayer(in_features=embedding_out_features, out_features=out_features, device=device)
            for _ in range(self.num_edge_types)
        ])

        self.gru_cell = torch.nn.GRUCell(input_size=embedding_out_features, hidden_size=out_features)


    def forward(self,
                x,
                edge_list: List[torch.tensor],
                batch_map: torch.Tensor,
                **kwargs):
        x_embedding = self.value_embeddingLayer(x)
        last_node_states = x_embedding
        for _ in range(8):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0:
                    # 该种类型的边存在边
                    out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        out = last_node_states
        return out