from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.nn.functional as F
import torch


class InterConv(MessagePassing):
    def __init__(self,  in_features, out_features, num_edge_types, device="cpu"):
        super(InterConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.num_edge_types = num_edge_types
        self.linear = nn.Linear(in_features, out_features)
        self.edge_index, weights = dense_to_sparse(torch.ones(num_edge_types) - torch.eye(num_edge_types))
        self.edge_index = self.edge_index.to("cuda")
        self.device = device

        
    
    def forward(self, x):
        # x: 4, VB, D
        D = x.shape[-1]
        x = x.view(-1, D) # x: 4VB, D
        x = self.linear(x)
        x = F.relu(x)
        x = x.view(self.num_edge_types, -1) # 4, VBD
        x = self.propagate(self.edge_index, x=x)
        x = x.view(self.num_edge_types, -1, D)
        return x


if __name__ == '__main__':
    import torch
    a = torch.rand(size=(4, 4*5, 16))
    ic = InterConv(16, 16, 4)
    d = ic(a)
    print("")