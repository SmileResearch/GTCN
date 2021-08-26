from torch_geometric.utils import subgraph
import torch
import json
from dataProcessing import CGraphDataset, PYGraphDataset, statistic_torch_graph_node_nums_and_label
from torch_geometric.data import DataLoader
from models import DGAP
from code_completion import CodeCompletion

if __name__ == '__main__':

    a = torch.rand(8, 10)
    print(a)
    print(a.view(-1, 2, 2))
    print(a)

    def matrix_transfer(edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0]+=i
        edge_new[1]+=j
        return edge_new

    def matrix_loop(edge_list):
        # 4个邻接矩阵并列。
        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list
        n = 50
        loop_edge_list = []
        loop_edge_list.append(A1)
        loop_edge_list.append(matrix_transfer(A2, n, 0))
        loop_edge_list.append(matrix_transfer(A3, 2*n, 0))
        loop_edge_list.append(matrix_transfer(A4, 3*n, 0))

        loop_edge_list.append(matrix_transfer(A4, 0, n))
        loop_edge_list.append(matrix_transfer(A1, n, n))
        loop_edge_list.append(matrix_transfer(A2, 2*n, n))
        loop_edge_list.append(matrix_transfer(A3, 3*n, n))

        loop_edge_list.append(matrix_transfer(A3, 0, 2*n))
        loop_edge_list.append(matrix_transfer(A4, n, 2*n))
        loop_edge_list.append(matrix_transfer(A1, 2*n, 2*n))
        loop_edge_list.append(matrix_transfer(A2, 3*n, 2*n))

        loop_edge_list.append(matrix_transfer(A2, 0, 3*n))
        loop_edge_list.append(matrix_transfer(A3, n, 3*n))
        loop_edge_list.append(matrix_transfer(A4, 2*n, 3*n))
        loop_edge_list.append(matrix_transfer(A1, 3*n, 3*n))

        return torch.stack(loop_edge_list, dim=1)

    a =  torch.tensor([[0,0,0],[1,2,3]]).type(torch.IntTensor)
    b =  torch.tensor([[0,0,0],[4,5,6]]).type(torch.IntTensor)
    c =  torch.tensor([[0,0,0],[7,8,9]]).type(torch.IntTensor)
    d =  torch.tensor([[0,0,0],[9,2,3]]).type(torch.IntTensor)
    print(matrix_loop([a,b,c,d]))

    print("")