import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, subgraph
from os import listdir, mkdir
import os.path as osp
import json
import numpy as np
from utils import get_neighbors

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2
                 for (idx, char) in enumerate(ALPHABET)
                 }  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1


class CGraphData(Data):
    def __init__(self):
        super(CGraphData, self).__init__()
    """
    def __init__(self, edge_index, edge_index_three_hop, edge_index_five_hop, label, x):
        super(GraphData, self).__init__()
        self.edge_index = edge_index
        self.edge_index_three_hop = edge_index_three_hop
        self.edge_index_five_hop = edge_index_five_hop
        self.label = label
        self.x = x
    """
    def load_attr(self, edge_index,edge_index_ncs, edge_index_cfg, edge_index_dfg, label,x,right_most,candidate_id, candidate_id_mask):
        self.edge_index = edge_index
        #elf.edge_index_two_hop = edge_index_two_hop
        #self.edge_index_three_hop = edge_index_three_hop
        #self.edge_index_five_hop = edge_index_five_hop
        self.edge_index_cfg = edge_index_cfg
        self.edge_index_ncs = edge_index_ncs
        self.edge_index_dfg = edge_index_dfg
        self.label = label
        self.x = x
        self.right_most = right_most
        self.candidate_id = candidate_id
        self.candidate_masks = candidate_id_mask

    def __inc__(self, key, value):
        if key == "right_most":
            return self.num_nodes
        if key == "candidate_id":
            return self.num_nodes
        if key == "candidate_masks":
            return 0
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value): 
        return super().__cat_dim__(key, value)

def CMake_task_input(batch_data, get_num_edge_types=False):
    # 当get_num_edge_types设置为true的时候返回边的个数。
    if get_num_edge_types:
        return 4
    return {
        "x": batch_data.x,
        "edge_list":[
            batch_data.edge_index,
            batch_data.edge_index_ncs,
            batch_data.edge_index_dfg,
            batch_data.edge_index_cfg,
        ],
        "slot_id":batch_data.right_most,
        "candidate_ids":batch_data.candidate_id,
        "candidate_masks":batch_data.candidate_masks,
        "batch_map":batch_data.batch
    }

class CGraphDataset(Dataset):
    def __init__(self, root, value_dict, type_dict, graph_node_max_num_chars=19, max_node_per_graph=50,max_graph=20000, max_variable_candidates=5, transform=None, pre_transform=None):
        self.graph_node_max_chars=graph_node_max_num_chars
        self.value_dict = value_dict
        self.max_variable_candidates = max_variable_candidates
        self.max_node_per_graph = max_node_per_graph
        self.max_graph = max_graph
        super(CGraphDataset, self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        if not osp.exists(self.processed_dir):
            mkdir(self.processed_dir)
        return [
            file for file in listdir(self.processed_dir)
            if file.startswith("data")
        ]
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   'data_{}.pt'.format(idx)))
        return data
    '''
    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...'''

    def trans_list_to_edge_tensor(self, edge_index, is_add_self_loop=True, is_to_undirected=True, self_loop_num_nodes:int=None,make_sub_graph=True):
        """将[src,tgt]转为输入到data对象中的edge_index，即tensor([src...],[tgt...])

        Args:
            edge_index ([type]): [src,tgt]
            add_self_loop (bool, optional): 是否加自连接边. Defaults to True.
            to_undirected (bool, optional): 是否转为无向图. Defaults to True.
        """ 
        if edge_index is not None and len(edge_index) != 0:  # 当edge_index存在且有边时。
            t = torch.tensor(edge_index).t().contiguous()
        else:
            t = torch.tensor([]).type(torch.LongTensor).t().contiguous()
        

        if is_add_self_loop:
            if self_loop_num_nodes:
                t, weight = add_self_loops(t, num_nodes=self_loop_num_nodes)
            else:
                t, weight = add_self_loops(t, num_nodes=self.max_node_per_graph)  # 这里强制加了0-150的节点。方便做subgraph不会报错。
        if is_to_undirected:
            t = to_undirected(t)

        if make_sub_graph:
            t, attr = subgraph(torch.arange(start=0, end=self.max_node_per_graph), t)

        return t

    def process(self):
        # Read data into huge `Data` list.
        data_i = 0
        for raw_path in self.raw_paths:
            with open(raw_path, "r") as f:
                raw_sample = json.load(f)
                nodeDictRaw = raw_sample['nodeList']
                nodeDict = {node:value for node,value in nodeDictRaw.items() if int(node) < self.max_node_per_graph}
                num_nodes = len(nodeDict)

                astEdges = raw_sample['astEdges']
                cfgEdges = raw_sample['cfgEdges']
                dfgEdges = raw_sample["dfgEdges"]
                ncsEdges = raw_sample["ncsEdges"]
                candidateNodeListRaw = raw_sample["candidateNodeList"]
                candidateNodeList = [node for node in candidateNodeListRaw if node["candidateNode"] < self.max_node_per_graph]
                #three_neighbors_adj = get_neighbors(astEdges, 3)
                #five_neighbors_adj = get_neighbors(astEdges, 5)

                if not astEdges or not cfgEdges or not dfgEdges:
                    continue

                ast_adj = self.trans_list_to_edge_tensor(astEdges, self_loop_num_nodes=self.max_node_per_graph)  # 保留0-max_node_per_graph的边
                cfg_adj = self.trans_list_to_edge_tensor(cfgEdges, self_loop_num_nodes=self.max_node_per_graph)  
                ncs_adj = self.trans_list_to_edge_tensor(ncsEdges, self_loop_num_nodes=self.max_node_per_graph)  
                dfg_adj = self.trans_list_to_edge_tensor(dfgEdges, self_loop_num_nodes=self.max_node_per_graph)  
                #one_nei_adj = self.trans_list_to_edge_tensor(raw_sample['astEdges'])
                #three_nei_adj = self.trans_list_to_edge_tensor(three_neighbors_adj)
                #five_nei_adj = self.trans_list_to_edge_tensor(five_neighbors_adj)

                all_node_value_name = [node_name for node, node_name in nodeDict.items()]
                all_node_value_label = [
                    self.value_dict.get(node_name, len(self.value_dict))
                    for node_name in all_node_value_name
                ]

                candidate_name = [node["candidateCode"] for node in candidateNodeList]
                candidate_id = [node["candidateNode"] for node in candidateNodeList]
                candidate_type = [node["candidateType"] for node in candidateNodeList]
                for n in range(3, len(candidateNodeList)):
                    other_id = [candidate_id[idj] for idj in range(len(candidate_name)) if candidate_name[idj]!=candidate_name[n]]
                    correct_id_list = [candidate_id[idj] for idj in range(len(candidate_name)) if candidate_name[idj]==candidate_name[n] and idj!=n]
                    if len(other_id)>0 and len(correct_id_list)>0:
                        slot_node_id = candidate_id[n]
                        cand_id = other_id
                        correct_id = correct_id_list[0]
                        nodeDict[str(candidate_id[n])] = "<slot>"
                        candidate_id_list = [correct_id] + cand_id[:self.max_variable_candidates-1]
                        
                        mask_num = self.max_variable_candidates - len(candidate_id_list)
                        mask = [True] *len(candidate_id_list) + [False]*mask_num
                        candidate_id_list += [0] * mask_num

                        # 根据节点名称，获取初始embedding。
                        node_value_chars = np.zeros(shape=(self.max_node_per_graph, self.graph_node_max_chars), dtype=np.uint8)
                        for node, label in nodeDict.items():
                            node_value = label
                            for (char_idx, value_char) in enumerate(
                                    node_value[:self.graph_node_max_chars].lower()):
                                node_value_chars[int(node), char_idx] = ALPHABET_DICT.get(
                                    value_char, 1)


                        data = CGraphData()
                        data.load_attr(
                            edge_index=ast_adj,
                            edge_index_ncs=ncs_adj,
                            edge_index_cfg=cfg_adj,
                            edge_index_dfg=dfg_adj,
                            label=torch.tensor([0]).type(torch.LongTensor),
                            x=torch.tensor(node_value_chars).type(torch.LongTensor),
                            right_most=slot_node_id,
                            candidate_id=torch.tensor(candidate_id_list).type(torch.LongTensor),
                            candidate_id_mask=torch.tensor(mask).type(torch.FloatTensor),
                        )
                        torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(data_i)))
                        data_i += 1

                        if data_i > self.max_graph:
                            return

                        break

        '''
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        '''
    
if __name__ == '__main__':
    from torch_geometric.data import DataLoader
    D = 16
    V = 5
    edge = torch.tensor([[0, 1], [1, 2], [2, 4], [3, 4]])
    edge = to_undirected(edge.t())
    edge, weight = add_self_loops(edge)
    adj1 = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (V,V))
    adj2 = torch.sparse.mm(adj1, adj1)
    adj3 = torch.sparse.mm(adj2, adj1)
    adj4 = torch.sparse.mm(adj3, adj1)
    adj5 = torch.sparse.mm(adj4, adj1)

    adj1 = adj1.coalesce()
    adj2 = adj2.coalesce()
    adj3 = adj3.coalesce()
    adj4 = adj4.coalesce()
    adj5 = adj5.coalesce()
    # 获取1， 3， 5阶邻居只能通过遍历来。通过矩阵乘法不靠谱啊。
    # 这里先弄一个数据出来。
    
    adj1_index = adj1.indices()
    adj3_index = adj3.indices()
    adj5_index = adj5.indices()
    node_init_str = torch.ones(size=(V, D))
    #data = GraphData(adj1_index, adj3_index, adj5_index, 1, node_init_str)
    data = CGraphData()
    data.edge_index = adj1_index
    data.edge_index_three_hop = adj3_index
    data.edge_index_five_hop = adj5_index
    data.label = 1
    data.x = node_init_str
    print(data)
    with open(r"W:\Scripts\Python\Vscode\YangJia-GGNN-GCN-Pytorch\dataProcessing\left_sequence_linux_terminal_dict_1k.json", "r") as f:
        vocab_dict = json.load(f)
    gd = CGraphDataset("data/train_data", value_dict=vocab_dict)
    gdl = DataLoader(gd, batch_size=2)
    for g in gdl:
        print(g)
        print("")

    print("")
    
    d = Data(edge_index=torch.tensor([[1,2,3,1],[1,2,3,1]]))
    print(d.edge_index)