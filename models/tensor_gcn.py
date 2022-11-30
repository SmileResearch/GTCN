from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from .embedding_layer import EmbeddingLayer
from .mlp_layer import MLPLayer
from .inter_conv import InterConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple

class Tensor_GCN(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 in_features,
                 out_features,
                 embedding_out_features,
                 embedding_num_classes,
                 dropout=0,
                 max_node_per_graph=50,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device="cpu"):
        super(Tensor_GCN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.dropout = dropout
        self.max_node_per_graph=max_node_per_graph
        # value embedding layer for variable name.
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)

        self.MessagePassingNN = nn.ModuleList(
            [
                 MLPLayer(in_features=embedding_out_features,
                    out_features=out_features,device=device) for _ in range(self.num_edge_types)
            ]
        )

        self.gru_cell = torch.nn.GRUCell(input_size=embedding_out_features, hidden_size=out_features)
        

        self.lin = nn.Linear(out_features, out_features)
        self.conv1 = GATConv(out_features, out_features//8, add_self_loops=True, heads=8, concat=True)

    
    def forward(self, x, edge_list: List[torch.tensor], **kwargs):
        
        x_embedding = self.value_embeddingLayer(x)
        # Tensor GGNN 
        last_node_states = x_embedding
        for _ in range(8):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0 :
                    out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        ggnn_out = last_node_states # shape: V, D
        # tensor GCN:
        cur_x = torch.cat([ggnn_out for _ in range(self.num_edge_types)], dim=0)  # 4V, D
        loop_edge_list = self.matrix_loop_new(edge_list) # 4V, 4V
        out = self.conv1(cur_x, loop_edge_list) # 4V, D
        out = out.view(self.num_edge_types, x_embedding.shape[0], out.shape[-1])  # 4, V, D
        out = torch.sum(out, dim=0)  # V, D
        out = F.relu(out)
        out = self.lin(out)
    

        return out


    def matrix_transfer(self, edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0]+=i
        edge_new[1]+=j
        return edge_new

    def matrix_loop(self, edge_list):
        # do tensor dot
        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list
        n = self.max_node_per_graph
        loop_edge_list = []
        loop_edge_list.append(A1)
        loop_edge_list.append(self.matrix_transfer(A2, n, 0))
        loop_edge_list.append(self.matrix_transfer(A3, 2*n, 0))
        loop_edge_list.append(self.matrix_transfer(A4, 3*n, 0))

        loop_edge_list.append(self.matrix_transfer(A4, 0, n))
        loop_edge_list.append(self.matrix_transfer(A1, n, n))
        loop_edge_list.append(self.matrix_transfer(A2, 2*n, n))
        loop_edge_list.append(self.matrix_transfer(A3, 3*n, n))

        loop_edge_list.append(self.matrix_transfer(A3, 0, 2*n))
        loop_edge_list.append(self.matrix_transfer(A4, n, 2*n))
        loop_edge_list.append(self.matrix_transfer(A1, 2*n, 2*n))
        loop_edge_list.append(self.matrix_transfer(A2, 3*n, 2*n))

        loop_edge_list.append(self.matrix_transfer(A2, 0, 3*n))
        loop_edge_list.append(self.matrix_transfer(A3, n, 3*n))
        loop_edge_list.append(self.matrix_transfer(A4, 2*n, 3*n))
        loop_edge_list.append(self.matrix_transfer(A1, 3*n, 3*n))

        return torch.cat(loop_edge_list, dim=1)

    def matrix_loop_new(self, edge_list):
        n = self.max_node_per_graph
        edge_nums = len(edge_list)
        loop_edge_list = []
        edge_start = 0
        for j in range(edge_nums):
            cur_edge_start = edge_start    
            for i in range(edge_nums):
                cur_edge_start %= edge_nums
                loop_edge_list.append(self.matrix_transfer(edge_list[cur_edge_start], i*n, j*n))
                cur_edge_start += 1
            edge_start-=1
            edge_start
        return torch.cat(loop_edge_list, dim=1)
