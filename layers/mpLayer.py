from torch import FloatTensor
from torch import squeeze
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F


class MPLayer(MessagePassing):
    def __init__(self,  in_features, out_features, device="cpu"):
        """MPLayer  普通的MessagePassinglayer

        Args:
            in_feature ([type]): 传入的features
            out_features ([type]): 传出的features
            device ([type]): device
        """
        super(MPLayer, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.linear = nn.Linear(in_features, out_features)
        self.device = device

        
    
    def forward(self, x, edge_index):
        x = self.linear(x)
        return self.propagate(edge_index, x=x)


if __name__ == '__main__':
    import torch
    a = torch.rand(size=(10, 16))
    print(a)
    model = MPLayer( in_feature=16, out_features=25, device="cpu")
    b = model(a, torch.tensor([[1,2,3],[1,2,3]]))
    print(b)
    print("")