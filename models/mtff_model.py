from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple
from .embedding_layer import EmbeddingLayer # from .*** import表示从当前包中导入


class alignModule(nn.Module):
    def __init__(self, out_features, heads, max_node_per_graph):
        super(alignModule, self).__init__()
        self.max_node_per_graph = max_node_per_graph

        self.align_1_mmh = torch.nn.MultiheadAttention(out_features, heads)
        #self.align_1_mmh = torch.nn.MultiheadAttention(out_features, heads)

        self.FFN = nn.Sequential(nn.Linear(out_features * 2, out_features), nn.ReLU(),
                                 nn.Linear(out_features, out_features))

    def forward(self, x1, x2):
        x1_v = x1.view(self.max_node_per_graph, -1, x1.shape[-1])
        x2_v = x2.view(self.max_node_per_graph, -1, x2.shape[-1])

        out, _ = self.align_1_mmh(x1_v, x2_v, x2_v)
        return self.FFN(torch.cat([x1, out.view(-1, out.shape[-1])], dim=1))


class MTFF_Co_Attention(MessagePassing):
    # 多任务特征融合模型的部分。使用互注意力层提取不同任务之间的信息
    def __init__(
        self,
        feature_models,
        in_features,
        out_features,
        embedding_out_features,
        embedding_num_classes,
        heads=8,
        max_node_per_graph=50,
        device="cpu"
        
    ):
        super(MTFF_Co_Attention, self).__init__()
        self.max_node_per_graph = max_node_per_graph
        self.feature_models = feature_models
        self.model_nums = len(self.feature_models)
        
        self.linear = nn.Linear(out_features*self.model_nums, out_features)
        
        self.align = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)

        self.integrate = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)

        self.task_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                            in_features,
                                            embedding_out_features,
                                            device=device)




    def forward(
        self,
        x,
        **kwargs,
    ):
        
        x_list = [model(x, **kwargs) for model in self.feature_models]

        x = self.value_embeddingLayer(x)

        x_a = self.linear(torch.cat(x_list,dim=1))
        x_align = self.align(x, x_a)

        x_integrating = self.integrate(x_align, x_align)


        x_integrating_view = x_integrating.view(-1, x_integrating.shape[-1])

        x_mapping = self.task_mapping(x_integrating_view)
        
        return x_mapping
