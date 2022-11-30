from torch import FloatTensor
from torch import squeeze
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(MessagePassing):
    def __init__(self,  in_features, out_features, device="cpu"):
        """MPLayer  norm Message Passing layer for gnn.

        Args:
            in_feature ([type]): in_feature
            out_features ([type]): out_features
            device ([type]): device
        """
        super(MLPLayer, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.linear = nn.Linear(in_features, out_features)
        self.device = device

        
    
    def forward(self, x, edge_index):
        x = self.linear(x)
        return self.propagate(edge_index, x=x)


if __name__ == '__main__':
    import torch
    a = torch.rand(size=(10, 16))
    print(a)
    model = MLPLayer( in_feature=16, out_features=25, device="cpu")
    b = model(a, torch.tensor([[1,2,3],[1,2,3]]))
    print(b)
    print("")