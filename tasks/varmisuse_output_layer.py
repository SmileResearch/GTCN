from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple

from utils.model_metrics import cal_metrics

# 本地库


class VarmisuseOutputLayer(nn.Module):
    def __init__(
        self,
        out_features,
        max_variable_candidates=5,
        criterion=nn.CrossEntropyLoss(),
        metrics=cal_metrics,
        device="cpu",
    ):
        super(VarmisuseOutputLayer, self).__init__()
        
        self.max_variable_candidates = max_variable_candidates
        
        self.varmisuse_linear = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.Linear(out_features, out_features),
        )
        self.varmisuse_layer = nn.Linear(out_features * 2 + 1, 1)
        self.criterion=criterion
        self.metrics = metrics

    def forward(self,
                output,
                slot_id,
                candidate_ids,
                candidate_masks,
                label=None,
                output_label=False,
                metric_functions=None,
                **kwargs):
        output = self.varmisuse_linear(output)
        
        
        candidate_embedding = output[candidate_ids]  # shape: g*c, d
        slot_embedding = output[slot_id]  # shape: g, d

        candidate_embedding_reshape = candidate_embedding.view(-1, self.max_variable_candidates,
                                                               output.shape[-1])  # shape: g, c, d
        slot_inner_product = torch.einsum("cd,cvd->cv", slot_embedding, candidate_embedding_reshape)  #shape g, c

        slot_embedding_unsqueeze = torch.unsqueeze(slot_embedding, dim=1)  # shape: g,1,d
        slot_embedding_repeat = slot_embedding_unsqueeze.repeat(1, self.max_variable_candidates, 1)  # shape: g, c, d

        slot_cand_comb = torch.cat(
            [candidate_embedding_reshape, slot_embedding_repeat,
             torch.unsqueeze(slot_inner_product, dim=-1)], dim=2)  #shape: g, c, d*2+1
        logits = self.varmisuse_layer(slot_cand_comb)  # shape: g, c, 1
        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7
        logits=F.softmax(logits, dim=-1)

        result = [logits]

        if label is not None:
            loss = self.criterion(logits, label)
            result.append(loss)
            if output_label:
                result.append(label)
        
        # 如果metrics传入进来了，则计算每一个metrics的值。并返回。否则使用self.metrics
        if metric_functions is not None:
            if type(metric_functions) is list:
                metrics = dict()
                for m_function in metric_functions:
                    temp_metrics = m_function(logits, label, self.max_variable_candidates)
                    metrics.update(temp_metrics)
            else:
                metrics = metric_functions(logits, label, self.max_variable_candidates)
            result.append(metrics)
        elif self.metrics:
            metrics = self.metrics(logits, label, self.max_variable_candidates)
            result.append(metrics)
            
            
        return result