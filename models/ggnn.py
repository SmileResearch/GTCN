from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from layers import EmbeddingLayer, MPLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple


class GGNN(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 in_features,
                 out_features,
                 embedding_out_features,
                 classifier_features,
                 embedding_num_classes,
                 dropout=0,
                 max_variable_candidates=5,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device="cpu",
                 output_model="learning"):
        super(GGNN, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.output_model = output_model.lower()
        self.dropout = dropout
        self.max_variable_candidates = max_variable_candidates
        # 先对值进行embedding
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)

        self.MessagePassingNN = nn.ModuleList(
            [
                 MPLayer(in_features=embedding_out_features,
                    out_features=out_features,device=device) for _ in range(self.num_edge_types)
            ]
        )

        self.gru_cell = torch.nn.GRUCell(input_size=embedding_out_features, hidden_size=out_features)
        
        self.varmisuse_output_layer = nn.Linear(out_features* 2 + 1, 1)
        self.varnaming_output_layer = nn.Linear(out_features, classifier_features)

    def forward(self, x, edge_list: List[torch.tensor], slot_id, candidate_ids,
                candidate_masks, batch_map:torch.Tensor):
        x_embedding = self.value_embeddingLayer(x)

        last_node_states = x_embedding
        for _ in range(8):
            out_list = []
            cur_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0 :
                    # 该种类型的边存在边
                    out_list.append(self.MessagePassingNN[i](cur_node_states, edge))
            cur_node_states = sum(out_list)
            new_node_states = self.gru_cell(cur_node_states, last_node_states)  # input:states, hidden
            last_node_states = new_node_states

        out = last_node_states

        # varnaming
        #return self.varnaming_learning_output(out, slot_id)
        
        # varmisuse
        return self.varmisuse_learning_output(out, slot_id, candidate_ids, candidate_masks)
        

    def varmisuse_learning_output(self, out, slot_id, candidate_ids, candidate_masks):
        candidate_embedding = out[candidate_ids]  # shape: g*c, d
        slot_embedding = out[slot_id]  # shape: g, d

        candidate_embedding_reshape = candidate_embedding.view(-1, self.max_variable_candidates,
                                                               out.shape[-1])  # shape: g, c, d
        slot_inner_product = torch.einsum("cd,cvd->cv", slot_embedding, candidate_embedding_reshape)  #shape g, c

        slot_embedding_unsqueeze = torch.unsqueeze(slot_embedding, dim=1)  # shape: g,1,d
        slot_embedding_repeat = slot_embedding_unsqueeze.repeat(1, self.max_variable_candidates, 1)  # shape: g, c, d

        slot_cand_comb = torch.cat(
            [candidate_embedding_reshape, slot_embedding_repeat,
             torch.unsqueeze(slot_inner_product, dim=-1)], dim=2)  #shape: g, c, d*2+1
        logits = self.varmisuse_output_layer(slot_cand_comb)  # shape: g, c, 1
        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7

        return logits

    def varnaming_learning_output(self, out, slot_id):
        slot_embedding = out[slot_id]  # shape: g, d

        return self.varnaming_output_layer(slot_embedding)  # g, d*2 => g, classifier