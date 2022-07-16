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


class VarnamingOutputLayer(nn.Module):
    def __init__(
        self,
        out_features,
        classifier_nums,
        criterion=nn.CrossEntropyLoss(),
        metrics=cal_metrics,
        device="cpu",
    ):
        super(VarnamingOutputLayer, self).__init__()
        
        self.varnaming_linear = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.varmisuse_layer = nn.Linear(out_features, classifier_nums)
        self.criterion=criterion
        self.metrics = metrics
        self.classifier_nums = classifier_nums

    def forward(self,
                output,
                slot_id,
                value_label=None, 
                output_label=False, 
                metric_functions=None,
                **kwargs):
        output = self.varnaming_linear(output)
        
        slot_embedding = output[slot_id]  # shape: g, d
        
        logits = self.varmisuse_layer(slot_embedding)
        logits = F.softmax(logits, dim=-1)

        result = [logits]

        if value_label is not None:
            loss = self.criterion(logits, value_label)
            result.append(loss)
            if output_label:
                result.append(value_label)
        
        
        # 如果metrics传入进来了，则计算每一个metrics的值。并返回。否则使用self.metrics
        if metric_functions is not None:
            if type(metric_functions) is list:
                metrics = dict()
                for m_function in metric_functions:
                    temp_metrics = m_function(logits, value_label, self.classifier_nums)
                    metrics.update(temp_metrics)
            else:
                metrics = metric_functions(logits, value_label, self.classifier_nums)
        elif self.metrics:
            metrics = self.metrics(logits, value_label, self.classifier_nums)
            result.append(metrics)
            
        return result