import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, subgraph, is_undirected
from os import listdir, mkdir
import os.path as osp
import json
import numpy as np
#from utils import get_neighbors
import json
from random import shuffle
import gzip
import re


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2
                 for (idx, char) in enumerate(ALPHABET)
                 }  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1


class LRStaticGraphData(Data):
    def __init__(self):
        super(LRStaticGraphData, self).__init__()

    """
    def __init__(self, edge_index, edge_index_three_hop, edge_index_five_hop, label, x):
        super(GraphData, self).__init__()
        self.edge_index = edge_index
        self.edge_index_three_hop = edge_index_three_hop
        self.edge_index_five_hop = edge_index_five_hop
        self.label = label
        self.x = x
    """

    def load_attr(self, Child, NextToken, LastUse, LastWrite, LastLexicalUse,ComputedFrom, GuardedByNegation, GuardedBy,
                        FormalArgName, ReturnsTo, x, slot_id, candidate_ids, candidate_masks, label, value_label, type_label):
        """由于dataset的限制，不能初始化的时候传入参数，所以设立一个函数传入参数。

        Args:
            edge_index ([type]): ast one hop edge index
            edge_index_three_hop ([type]): ast three hop edge index
            edge_index_five_hop ([type]): ast five hop edge index
            value_label ([type]): the label of node value
            type_label ([type]): the label of node type
            x ([type]): the initial embedding of node value
            x_type ([type]): the initial embedding of node type
            right_most ([type]): the right most node id of a graph
        """
        self.Child_index = Child
        self.NextToken_index  = NextToken
        self.LastUse_index  = LastUse
        self.LastWrite_index  = LastWrite
        self.LastLexicalUse_index  = LastLexicalUse
        self.ComputedFrom_index  = ComputedFrom
        self.GuardedByNegation_index  = GuardedByNegation
        self.GuardedBy_index  = GuardedBy
        self.FormalArgName_index  = FormalArgName
        self.ReturnsTo_index  = ReturnsTo
        #self.SelfLoop_index = SelfLoop
        #self.two_nei_index = two_nei_adj
        #self.three_nei_index = three_nei_adj
        #self.five_nei_index = five_nei_adj
        self.x = x
        self.slot_id = slot_id
        self.candidate_ids = candidate_ids
        self.candidate_masks = candidate_masks
        self.label = label
        self.value_label = value_label
        self.type_label = type_label

    def __inc__(self, key, value):
        if key == "slot_id":
            return self.num_nodes
        if key == "candidate_ids":
            return self.num_nodes
        if key == "candidate_masks":
            return 0
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value):
        return super().__cat_dim__(key, value)

def LRStaticMake_task_input(batch_data, get_num_edge_types=False):
    # 当get_num_edge_types设置为true的时候返回边的个数。
    if get_num_edge_types:
        return 4
    return {
        "x": batch_data.x,
        "edge_list":[
            batch_data.Child_index,
            batch_data.NextToken_index,
            batch_data.LastUse_index,
            batch_data.LastWrite_index,
            #batch_data.LastLexicalUse_index,
            #batch_data.ComputedFrom_index,
            #batch_data.GuardedByNegation_index,
            #batch_data.GuardedBy_index,
            #batch_data.FormalArgName_index,
            #batch_data.ReturnsTo_index,
            #batch_data.SelfLoop_index,
        ],
        "slot_id":batch_data.slot_id,
        "candidate_ids":batch_data.candidate_ids,
        "candidate_masks":batch_data.candidate_masks,
        "batch_map":batch_data.batch,
    }

class LRStaticGraphDataset(Dataset):
    """静态的learning graph dataset。做出以下改进：
        1.将图的大小限制到0-150
        2.每个图都有self loop。因为是针对于attention的。所以加上self loop后，更容易计算邻居对于本节点的权重。
            当没有邻居节点时，该节点值也会自我更新。不会因为没有邻居节点，节点值就为空。

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
                 root,
                 value_dict,
                 type_dict,
                 graph_node_max_num_chars=19,
                 max_node_per_graph=50,
                 max_graph=20000,
                 max_variable_candidates=5,
                 transform=None,
                 pre_transform=None):
        self.graph_node_max_chars = graph_node_max_num_chars
        self.max_node_per_graph = max_node_per_graph
        self.value_dict = value_dict
        self.type_dict = type_dict
        self.max_graph = max_graph
        self.max_variable_candidates = max_variable_candidates
        super(LRStaticGraphDataset, self).__init__(root, transform, pre_transform)
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

    def trans_list_to_edge_tensor(self,
                                  edge_index,
                                  is_add_self_loop=True,
                                  is_to_undirected=True, self_loop_num_nodes:int=None, make_sub_graph=True, do_node_trans=False):
        """将[src,tgt]转为输入到data对象中的edge_index，即tensor([src...],[tgt...])
            加上自连接边、加上双向边。

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
        if is_to_undirected and t.shape[0]!=0:
            t = to_undirected(t)
        
        if make_sub_graph:
            t, attr = subgraph(torch.arange(start=0, end=self.max_node_per_graph), t)

        if do_node_trans:
            t_new = torch.tensor([[self.node_trans_dict[i.item()] for i in row]  for row in t]).type(torch.LongTensor)
            t = t_new
        return t

    def process(self):
        # Read data into huge `Data` list.
        data_i = 0

        for raw_path in self.raw_paths:
            with gzip.open(raw_path, "r") as f:  # josnl.gz文件先通过gz读取，再按行读取。
                raw_data = f.read()
                data_utf8 = raw_data.decode("utf-8")
                for raw_line in data_utf8.split("\n"):
                    if len(raw_line) == 0:  # 过滤最后一行的空行。
                        continue

                    graph_data = LRStaticGraphData()
                    raw_dict = json.loads(raw_line)
                    filename = raw_dict['filename']
                    ContextGraph = raw_dict['ContextGraph']

                    # node label
                    NodeLabels = ContextGraph["NodeLabels"]
                    NodeTypes = ContextGraph["NodeTypes"]
                    slotTokenIdx = raw_dict["slotTokenIdx"]
                    SlotDummyNode = raw_dict["SlotDummyNode"]
                    SymbolCandidates = raw_dict["SymbolCandidates"]
                    
                    num_nodes = len(NodeLabels)
                    if num_nodes < self.max_node_per_graph:
                        # 由于要将图固定大小，然后做图归一化，所以小于这个数的图需要被抛弃。
                        continue

                    # 制作一个转换dict，将0-self.max_node_per_graph节点进行扰乱。
                    node_id_list = list(range(self.max_node_per_graph))
                    shuffle(node_id_list)
                    #self.node_trans_dict = {i:node_id_list[i] for i in node_id_list}
                    #self.node_trans_dict_reverse = {node_id_list[i]:i for i in node_id_list}

                    # edge_index
                    Edges = ContextGraph["Edges"]
                    Child = self.trans_list_to_edge_tensor(
                        Edges.get("Child", None))
                    NextToken = self.trans_list_to_edge_tensor(
                        Edges.get("NextToken", None))
                    LastUse = self.trans_list_to_edge_tensor(
                        Edges.get("LastUse", None))
                    LastWrite = self.trans_list_to_edge_tensor(
                        Edges.get("LastWrite", None))
                    LastLexicalUse = self.trans_list_to_edge_tensor(
                        Edges.get("LastLexicalUse", None))
                    ComputedFrom = self.trans_list_to_edge_tensor(
                        Edges.get("ComputedFrom", None))
                    GuardedByNegation = self.trans_list_to_edge_tensor(
                        Edges.get("GuardedByNegation", None))
                    GuardedBy = self.trans_list_to_edge_tensor(
                        Edges.get("GuardedBy", None))
                    FormalArgName = self.trans_list_to_edge_tensor(
                        Edges.get("FormalArgName", None))
                    ReturnsTo = self.trans_list_to_edge_tensor(
                        Edges.get("ReturnsTo", None))
                    #SelfLoop = add_self_loops(torch.tensor([]), num_nodes=num_nodes)[0].type(torch.LongTensor)
                    #SelfLoopEdge

                    #two_neighbors_adj = get_neighbors(Edges.get("Child", None), 2)
                    #three_neighbors_adj = get_neighbors(Edges.get("Child", None), 3)
                    #five_neighbors_adj = get_neighbors(Edges.get("Child", None), 5)

                    #two_nei_adj = self.trans_list_to_edge_tensor(two_neighbors_adj)
                    #three_nei_adj = self.trans_list_to_edge_tensor(three_neighbors_adj)
                    #five_nei_adj = self.trans_list_to_edge_tensor(five_neighbors_adj)

                    correct_candidate_id = None
                    distractor_candidate_ids = []  # type: List[int]
                    for candidate in SymbolCandidates:
                        if candidate["IsCorrect"]:
                            #correct_candidate_id = self.node_trans_dict[candidate['SymbolDummyNode']]
                            correct_candidate_id = candidate['SymbolDummyNode']
                        else:
                            #distractor_candidate_ids.append(self.node_trans_dict[candidate['SymbolDummyNode']])
                            distractor_candidate_ids.append(candidate['SymbolDummyNode'])
                    if correct_candidate_id is None:
                        continue

                    candidate_node_ids = [
                        correct_candidate_id
                    ] + distractor_candidate_ids[:self.max_variable_candidates -
                                                 1]

                    num_scope_padding = self.max_variable_candidates - len(
                        candidate_node_ids)
                    candidate_node_ids_mask = [True] * len(
                        candidate_node_ids) + [False] * num_scope_padding

                    candidate_node_ids = candidate_node_ids + [
                        0
                    ] * num_scope_padding  # 在导入数据阶段，转化为正负。
                    #shuffle(candidate_node_ids)
                    #cur_label = candidate_node_ids.index(correct_candidate_id)
                    #candidate_node_ids_mask = [True]*self.max_variable_candidates
                    #for j in range(self.max_variable_candidates):
                    #    if candidate_node_ids[j] == self.max_node_per_graph-1:
                    #        candidate_node_ids_mask[j] = False
                    # 根据节点名称，获取初始embedding。
                    node_value_chars = np.zeros(
                        shape=(self.max_node_per_graph, self.graph_node_max_chars),
                        dtype=np.uint8)
                    node_counter = 0
                    for node, label in NodeLabels.items():
                        node_counter += 1
                        if node_counter > self.max_node_per_graph:
                            break
                        #node = self.node_trans_dict[int(node)]
                        node = int(node)
                        for (char_idx, value_char) in enumerate(
                                label[:self.graph_node_max_chars].lower()):
                            node_value_chars[node,
                                             char_idx] = ALPHABET_DICT.get(
                                                 value_char, 1)
                    #value_label = int(self.value_dict.get(NodeLabels[str(self.node_trans_dict_reverse[correct_candidate_id])], len(self.value_dict)))
                    #type_label = int(self.type_dict.get(NodeTypes[str(self.node_trans_dict_reverse[correct_candidate_id])], len(self.type_dict)))
                    value_label = int(self.value_dict.get(NodeLabels[str(correct_candidate_id)], len(self.value_dict)))
                    type_label = int(self.type_dict.get(NodeTypes[str(correct_candidate_id)], len(self.type_dict)))
                    #SlotDummyNode_trans = self.node_trans_dict[SlotDummyNode]
                    graph_data.load_attr(
                        Child=Child, NextToken=NextToken, LastUse=LastUse, LastWrite=LastWrite, LastLexicalUse=LastLexicalUse,
                        ComputedFrom=ComputedFrom, GuardedByNegation=GuardedByNegation, GuardedBy=GuardedBy,
                        FormalArgName=FormalArgName, ReturnsTo=ReturnsTo,#SelfLoop,
                        x=torch.tensor(node_value_chars).type(torch.LongTensor),
                        slot_id=torch.tensor(SlotDummyNode).type(torch.LongTensor),
                        candidate_ids=torch.tensor(candidate_node_ids).type(torch.LongTensor),
                        candidate_masks=torch.tensor(candidate_node_ids_mask).type(torch.FloatTensor),
                        label=torch.tensor([0]).type(torch.LongTensor),
                        value_label=value_label,
                        type_label=type_label,)
                    torch.save(graph_data, osp.join(self.processed_dir, 'data_{}.pt'.format(data_i)))
                    data_i += 1
                    #two_nei_adj, three_nei_adj, five_nei_adj,
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

    with open(r"vocab_dict\learning_terminal_dict_1k_value.json", "r") as f:
        value_dict = json.load(f)
    with open(r"vocab_dict\learning_terminal_dict_1k_type.json", "r") as f:
        type_dict = json.load(f)
    gd = LRStaticGraphDataset("data/lrtemp/",
                        value_dict=value_dict,
                        type_dict=type_dict)
    batch_iterator = DataLoader(gd, batch_size=3)
    for b in batch_iterator:
        print(b)
