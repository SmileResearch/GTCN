import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, subgraph, remove_self_loops
from os import listdir, mkdir
import os.path as osp
import json

from torch_geometric.utils.undirected import is_undirected
import numpy as np
from utils import get_neighbors
import json
from random import shuffle
from treelib import Node, Tree

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2
                 for (idx, char) in enumerate(ALPHABET)
                 }  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1

BIG_NUMBER = 1e7
class PYGraphData(Data):
    def __init__(self):
        super(PYGraphData, self).__init__()
    """
    def __init__(self, edge_index, edge_index_three_hop, edge_index_five_hop, label, x):
        super(GraphData, self).__init__()
        self.edge_index = edge_index
        self.edge_index_three_hop = edge_index_three_hop
        self.edge_index_five_hop = edge_index_five_hop
        self.label = label
        self.x = x
    """
    def load_attr(self, edge_index, edge_index_ncs, edge_index_dfg, edge_index_last_write, edge_index_last_use, edge_index_self_loop, value_label, type_label, label, x, x_type, right_most, candidate_id, candidate_id_mask):
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
        self.edge_index = edge_index
        #elf.edge_index_two_hop = edge_index_two_hop
        #self.edge_index_three_hop = edge_index_three_hop
        #self.edge_index_five_hop = edge_index_five_hop
        self.edge_index_ncs = edge_index_ncs
        self.edge_index_dfg = edge_index_dfg
        self.edge_index_last_write = edge_index_last_write
        self.edge_index_last_use = edge_index_last_use
        self.edge_index_self_loop = edge_index_self_loop
        self.value_label = value_label
        self.type_label = type_label
        self.label = label
        self.x = x
        self.x_type = x_type
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


def PYMake_task_input(batch_data, get_num_edge_types=False):
    # 当get_num_edge_types设置为true的时候返回边的个数。
    if get_num_edge_types:
        return 4
    return {
        "x": batch_data.x,
        "edge_list":[
            batch_data.edge_index,
            batch_data.edge_index_ncs,
            batch_data.edge_index_last_write,
            batch_data.edge_index_last_use,
        ],
        "slot_id":batch_data.right_most,
        "candidate_ids":batch_data.candidate_id,
        "candidate_masks":batch_data.candidate_masks,
        "batch_map":batch_data.batch
    }


class PYGraphDataset(Dataset):
    def __init__(self, root, value_dict, type_dict, graph_node_max_num_chars=19, max_node_per_graph=50,max_graph=20000, max_variable_candidates=5, transform=None, pre_transform=None):
        self.max_node_per_graph = max_node_per_graph
        self.graph_node_max_chars = graph_node_max_num_chars
        self.value_dict = value_dict
        self.type_dict = type_dict
        self.max_graph = max_graph
        self.max_variable_candidates = max_variable_candidates
        super(PYGraphDataset, self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        if not osp.exists(self.processed_dir):
            mkdir(self.processed_dir)
        return [file for file in listdir(self.processed_dir) if file.startswith("data")]

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    '''
    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...'''

    def trans_list_to_edge_tensor(self, edge_index, is_add_self_loop=True, is_to_undirected=True, self_loop_num_nodes:int=None):
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
                t, weight = add_self_loops(t)
        if is_to_undirected and is_undirected(t):
            t = to_undirected(t)
        return t

    def process(self):
        # Read data into huge `Data` list.
        data_i = 0

        def is_var(node):
            return node.is_leaf() and node.data["value"]!="EMPTY"

        for raw_path in self.raw_paths:
            with open(raw_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    # one line means one graph
                    # list[Dict["type":typename, "children":list[index], "value":valuename]]
                    # 检查有value的是不是都没children 不是。有些节点有children也有value
                    # 是不是没有children的都有value.不是。有些节点即没有children也没有value。
                    # 任务：
                    #   1.要建立ast edge_index,   done
                    #   2.要建立nodeDict nodeDict["nodeid":{"type", "value"}]  done
                    #   3.建立叶子节点下标：没有children，有value的。
                    #   4.建立ncs。 建立一个图  done
                    ast_edge_index = []
                    ncs_edge_index = []
                    dfg_edge_index = []
                    ncs_pre_node = None
                    var_id_list = []
                    nodeDict = {}
                    ast_node_list = json.loads(line)

                    if len(ast_node_list)<20 or len(ast_node_list)>200:  # 这个过滤值得商量，因为现在图的大小限制到self.max_node_per_graph了。
                        # 过滤一下，这里只保留大于10，小于300个ast的function。
                        continue

                    # node_parent record the parent node of give node
                    # for_id record the forstatement node in the ast
                    # if_id record the ifstatement and orelse node in the ast
                    # if_index work for if_id
                    # this four args is for the last_use and last_write edge index.
                    node_parent = {}
                    for_id = []
                    if_id = []
                    assign_left_child = []
                    if_index = -1
                    var_name_dict = {}  # 用于对同名变量串联的dict。
                    for node_id in range(len(ast_node_list)):
                        ast_node = ast_node_list[node_id]
                        node_type = ast_node["type"]
                        node_children = ast_node.get("children", None)
                        node_value = ast_node.get("value", "EMPTY")  # node_value 默认值为EMPTY

                        if node_type == "For":
                            for_id.append(node_id)
                        if node_type == "If":
                            if_id.append([node_id])
                            if_index += 1
                        if node_type == "orelse" and if_index!=-1:
                            if_id[if_index].append(node_id)
                        if node_type == "Assign":
                            assign_left_child.append(node_id+1)  # Assign node left child. 被赋值的那个节点。

                        if node_children is not None:
                            for node in node_children:
                                node_parent[node] = node_id

                        if node_children is not None:
                            ast_edge_index.extend([[node_id, tgt_id] for tgt_id in node_children])

                        nodeDict[node_id] = {"type":node_type, "value":node_value}
                        
                        if node_children is None:
                            # 叶子节点 并不一定是一个变量。
                            pass
                            

                        if node_children is None and node_value != "EMPTY":
                            # 该节点是一个变量  没有孩子，且有node value node_value不为EMPTY。
                            var_id_list.append(node_id)

                            # 添加ncs
                            if ncs_pre_node is not None:
                                # 添加了ncs边
                                ncs_edge_index.append([ncs_pre_node, node_id])
                            ncs_pre_node = node_id

                            # 添加同名变量之间的dataflow
                            if node_value in var_name_dict.keys():
                                dfg_edge_index.append([var_name_dict[node_value], node_id])
                            var_name_dict[node_value] = node_id

                    #get last use last write edge.
                    tree = Tree()
                    for node in range(len(ast_node_list)):
                        tree.create_node(node, node, parent=node_parent.get(node, None), data={"type":ast_node_list[node]["type"], "value":ast_node_list[node].get("value", "EMPTY")})
                    #tree.show()
                    # 所有排列都是先序深度优先遍历的。
                    # get all vars
                    all_var = [node.identifier for node in tree.all_nodes() if is_var(node)]
                    # 获取所有的被赋值的变量。
                    all_assign_left_var = [node.identifier for alc in assign_left_child for node in tree.leaves(alc) if is_var(node) ]
                    # get all f min max var node.
                    # 将for下面的变量乘以2.
                    for f_id in for_id:
                        f_leaves = [node.identifier for node in tree.leaves(f_id) if is_var(node)]
                        min_f_leaves = min(f_leaves)
                        max_f_leaves = max(f_leaves)
                        min_f_leaves_index = all_var.index(min_f_leaves)
                        max_f_leaves_index = all_var.index(max_f_leaves)
                        all_var = all_var[:min_f_leaves_index]+all_var[min_f_leaves_index:max_f_leaves_index+1]*2 +all_var[max_f_leaves_index+1:]
                    

                    # 重新设置这个if。
                    # 获取if下面的所有节点。，  设置一个参数segment id。
                    # 当为if segment时：
                    #   如果 list 长度为1，则与它构造边，并添加一个元素。list长度为2.
                    #   如果 list 长度大于1，检查segment id，如果和segment id相同，则与-1建立边，并替代。
                    #                                      如果segment id 不同。则与第1个建立边，并在最末尾添加一个元素。
                    # 当不为if segemnt的时候：
                    #   如果list长度为1，则构造边并替代。
                    #   如果list长度大于1.则与每个构造边，并整个替代。
                    if_node_2_segment_id = {}
                    segment_id = 0
                    for if_segment in if_id:
                        cur_if_segment_node_id_list = []
                        for i in range(len(if_segment)):
                            if_nodeId = if_segment[i]
                            cur_if_segment_node_id_list.append({node.identifier for node in tree.leaves(if_nodeId) if is_var(node)})
                            if i != 0:
                                cur_if_segment_node_id_list[0] -= cur_if_segment_node_id_list[i]

                        for i in range(len(if_segment)):
                            for node in cur_if_segment_node_id_list[i]:
                                if_node_2_segment_id[node] = segment_id
                            segment_id += 1
                
                    
                    last_use_edge_index = []
                    last_write_edge_index = []
                    last_use_dict = {}  # dict[valueName] = list(lastNode_id)
                    last_write_dict = {}

                    def create_lastUse_lastWrite_edge():
                        # 重新设置这个if。
                        # 获取if下面的所有节点。，  设置一个参数segment id。
                        # 当为if segment时：
                        #   如果 list 长度为1，则与它构造边，并添加一个元素。list长度为2.
                        #   如果 list 长度大于1，检查segment id，如果和segment id相同，则与-1建立边，并替代。
                        #                                      如果segment id 不同。则与第1个建立边，并在最末尾添加一个元素。
                        #   如果segment id和之前相同，则与最后一个建立边并代替。
                        #   如果segment id不同，则与第一个建立边，并在末尾添加一个元素。并替换segment id
                        #   
                        # 当不为if segemnt的时候：
                        #   如果list长度为1，则构造边并替代。
                        #   如果list长度大于1.则与每个构造边，并整个替代。
                        segment_id = -1
                        i = 0
                        while True:
                            if i == len(all_var):
                                break  # 遍历至最后一个节点。了。
                            cur_var_id = all_var[i]
                            cur_var_value = tree.get_node(cur_var_id).data["value"]

                            last_use_node_list = last_use_dict.get(cur_var_value, None)
                            if last_use_node_list is None:
                                last_use_dict[cur_var_value] = [cur_var_id]  # 设置该节点为上一个节点
                            else:
                                # 上一个值至少存在一个。
                                cur_node_segment_id = if_node_2_segment_id.get(cur_var_id, None)

                                if cur_node_segment_id is not None:
                                    # 当前为 if segment中的变量。
                                    if cur_node_segment_id == segment_id:
                                        # 当前segment id和之前相同。
                                        last_use_edge_index.append([cur_var_id, last_use_node_list[-1]])  # 创建该节点到上一个节点的边。
                                        last_use_dict[cur_var_value][-1] = cur_var_id  # 设置该节点为上一个节点
                                    else:
                                        last_use_edge_index.append([cur_var_id, last_use_node_list[0]])  # 创建该节点到上一个节点的边。
                                        last_use_dict[cur_var_value].append(cur_var_id)
                                        segment_id = cur_node_segment_id

                                else:
                                    # 不为if segment. 与其中的每个节点构造边并替代。
                                    for last_use_node_id in last_use_node_list:
                                        last_use_edge_index.append([cur_var_id, last_use_node_id])  # 创建该节点到上一个节点的边。
                                    last_use_dict[cur_var_value] = [cur_var_id]  # 设置该节点为上一个节点
                                        
                            # last write 在加入边的时候检查其是否是左值节点。
                            last_write_node_list = last_write_dict.get(cur_var_value, None)
                            if last_write_node_list is None:
                                if cur_var_id in all_assign_left_var:
                                    last_write_dict[cur_var_value] = [cur_var_id]  # 设置该节点为上一个节点
                            else:
                                # 上一个值至少存在一个。
                                cur_node_segment_id = if_node_2_segment_id.get(cur_var_id, None)

                                if cur_node_segment_id is not None:
                                    # 当前为 if segment中的变量。
                                    if cur_node_segment_id == segment_id:
                                        # 当前segment id和之前相同。
                                        last_write_edge_index.append([cur_var_id, last_write_node_list[-1]])  # 创建该节点到上一个节点的边。
                                        if cur_var_id in all_assign_left_var:
                                            last_write_dict[cur_var_value][-1] = cur_var_id  # 设置该节点为上一个节点
                                    else:
                                        last_write_edge_index.append([cur_var_id, last_write_node_list[0]])  # 创建该节点到上一个节点的边。
                                        if cur_var_id in all_assign_left_var:
                                            last_write_dict[cur_var_value].append(cur_var_id)
                                        segment_id = cur_node_segment_id

                                else:
                                    # 不为if segment. 与其中的每个节点构造边并替代。
                                    for last_write_node_id in last_write_node_list:
                                        last_write_edge_index.append([cur_var_id, last_write_node_id])  # 创建该节点到上一个节点的边。
                                    if cur_var_id in all_assign_left_var:
                                        last_write_dict[cur_var_value] = [cur_var_id]  # 设置该节点为上一个节点

                            i += 1



                    create_lastUse_lastWrite_edge()

                    graph = {
                        "nodeDict":nodeDict,
                        "ast_edge_index":ast_edge_index,
                        "ncs_edge_index":ncs_edge_index,
                        "last_use_edge_index":last_use_edge_index,
                        "last_write_edge_index":last_write_edge_index,
                        "var_id_list":var_id_list,
                        "data_flow":dfg_edge_index,

                    }


                    num_nodes = len(nodeDict)

                    astEdges = ast_edge_index
                    #astEdges.extend(ncs_edge_index)
                    #astEdges.extend(dfg_edge_index)
                    #two_neighbors_adj = get_neighbors(astEdges, 2)
                    #three_neighbors_adj = get_neighbors(astEdges, 3)
                    #five_neighbors_adj = get_neighbors(astEdges, 5)

                    if not astEdges or not dfg_edge_index or not ncs_edge_index:
                        continue

                    one_nei_adj = self.trans_list_to_edge_tensor(astEdges)
                    #two_nei_adj = self.trans_list_to_edge_tensor(two_neighbors_adj)
                    #three_nei_adj = self.trans_list_to_edge_tensor(three_neighbors_adj)
                    #five_nei_adj = self.trans_list_to_edge_tensor(five_neighbors_adj)
                    ncs_adj = self.trans_list_to_edge_tensor(ncs_edge_index, self_loop_num_nodes=num_nodes)  # 后面会删除self_loop边。
                    dfg_adj = self.trans_list_to_edge_tensor(dfg_edge_index, self_loop_num_nodes=num_nodes)
                    last_use_adj = self.trans_list_to_edge_tensor(last_use_edge_index, self_loop_num_nodes=num_nodes)
                    last_write_adj = self.trans_list_to_edge_tensor(last_write_edge_index, self_loop_num_nodes=num_nodes)
                    self_loop_adj = self.trans_list_to_edge_tensor(None, is_to_undirected=False, is_add_self_loop=True, self_loop_num_nodes=self.max_node_per_graph)  # self_loop 大小固定为self.max_node_per_graph, 转为无向边会报错。不知道为什么。

                    all_node_value_name = [node_attr_dict['value'] for node, node_attr_dict in nodeDict.items()]
                    all_node_value_label = [
                        self.value_dict.get(node_name, len(self.value_dict))
                        for node_name in all_node_value_name
                    ]

                    all_node_type_name = [node_attr_dict['type'] for node, node_attr_dict in nodeDict.items()]
                    all_node_type_label = [
                        self.type_dict.get(node_name, len(self.type_dict))
                        for node_name in all_node_type_name
                    ]

                    # 根据节点名称，获取初始embedding。
                    node_value_chars = np.zeros(shape=(num_nodes, self.graph_node_max_chars), dtype=np.uint8)
                    for node, label in nodeDict.items():
                        node_value = label['value']
                        for (char_idx, value_char) in enumerate(
                                node_value[:self.graph_node_max_chars].lower()):
                            node_value_chars[int(node), char_idx] = ALPHABET_DICT.get(
                                value_char, 1)

                    node_type_chars = np.zeros(shape=(num_nodes, self.graph_node_max_chars), dtype=np.uint8)
                    for node, label in nodeDict.items():
                        node_type = label['type']
                        for (char_idx, value_char) in enumerate(
                                node_type[:self.graph_node_max_chars].lower()):
                            node_type_chars[int(node), char_idx] = ALPHABET_DICT.get(
                                value_char, 1)

                    # 对每一个变量。获取在这个变量前的所有节点。即left_sequence
                    #candidateNodeList = {node:node_attr_dict for node,node_attr_dict in nodeDict.items() if node_attr_dict["value"] != "EMPTY"}
                    #for j in range(1, len(var_id_list)):
                    for j in var_id_list:
                        # 不预测第一个变量，避免麻烦。
                        node_id = j
                        #node_id = int(candidate)
                        if node_id < 51:
                            # 过滤掉一些变量节点太小的图，小于10个节点图就过滤掉了。
                            continue
                        
                        # 获取one_hop three_hop five_hop子图 
                        cur_node_value = node_value_chars[node_id-self.max_node_per_graph:node_id, :]
                        cur_node_type = node_type_chars[node_id-self.max_node_per_graph:node_id, :]
                        try:
                            cur_sub_graph_one, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), one_nei_adj)
                            cur_sub_graph_one, attr = remove_self_loops(cur_sub_graph_one)

                            '''cur_sub_graph_two, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), two_nei_adj)
                            cur_sub_graph_two, attr = remove_self_loops(cur_sub_graph_two)

                            cur_sub_graph_three, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), three_nei_adj)
                            cur_sub_graph_three, attr = remove_self_loops(cur_sub_graph_three)

                            cur_sub_graph_five, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), five_nei_adj)
                            cur_sub_graph_five, attr = remove_self_loops(cur_sub_graph_five)'''
                            
                            cur_ncs_adj, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), ncs_adj)
                            cur_ncs_adj, attr = remove_self_loops(cur_ncs_adj)
                            
                            cur_dfg_adj, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), dfg_adj)
                            cur_dfg_adj, attr = remove_self_loops(cur_dfg_adj)

                            cur_last_write_adj, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), last_write_adj)
                            cur_last_write_adj, attr = remove_self_loops(cur_last_write_adj)

                            cur_last_use_adj, attr = subgraph(torch.arange(start=node_id-self.max_node_per_graph, end=node_id), last_use_adj)
                            cur_last_use_adj, attr = remove_self_loops(cur_last_use_adj)

                            # 出现一种情况时会报错，即66节点不存在边。但是你尝试通过subgraph获取0-66节点的边时就报错。告诉你66节点不存在边。这种报错ast中不存在，


                            # 由于只选取了self.max_node_per_graph个节点，做运算时，对节点选择进行运算，所以图也要相应的缩小。
                            cur_sub_graph_one -= (node_id-self.max_node_per_graph)
                            #cur_sub_graph_two -= (node_id-self.max_node_per_graph)
                            #cur_sub_graph_three -= (node_id-self.max_node_per_graph)
                            #cur_sub_graph_five -= (node_id-self.max_node_per_graph)
                            cur_ncs_adj -= (node_id-self.max_node_per_graph)
                            cur_dfg_adj -= (node_id-self.max_node_per_graph)
                            cur_last_write_adj -= (node_id-self.max_node_per_graph)
                            cur_last_use_adj -= (node_id-self.max_node_per_graph)
                            cur_self_loop_adj = self_loop_adj
                        except Exception as e:
                            # 出现一些问题，为避免故障，直接跳过。
                            continue
                        cur_node_value_label = all_node_value_label[node_id]
                        if cur_node_value_label == len(self.value_dict):
                            # 当label为 unk 时过滤掉。
                            continue
                        cur_node_type_label = all_node_type_label[node_id]
                        cur_right_most_node_id = self.max_node_per_graph-1  # 最后一个值。0-49 共50


                        # 对于next变量。我们先试试选择所有变量进行对比效果如何吧？
                        # 不太行，我们试试选择10个节点。
                        cur_node_value_name = all_node_value_name[node_id]
                        cur_node_type_name = all_node_type_name[node_id]
                        var_id_less_50 = [nid for nid in var_id_list if nid<node_id and nid>50]
                        type_ane_value_correct_id_list = [nid for nid in var_id_less_50 if all_node_value_name[nid] == cur_node_value_name and all_node_type_name[nid] == cur_node_type_name]
                        
                        other_id_list = []
                        # 变量类型和cur_node相同。
                        # 变量value_name 和 cur_node不同
                        # 变量只出现一次。
                        # node_id - 50
                        candidate_value_name_set = {cur_node_value_name}
                        for id in var_id_less_50:
                            candidate_value_name = all_node_value_name[id]
                            candidate_type_name = all_node_type_name[id]
                            if candidate_value_name != cur_node_value_name and candidate_type_name == cur_node_type_name and candidate_value_name not in candidate_value_name_set:
                                other_id_list.append(id)
                                candidate_value_name_set.add(candidate_value_name)
                        #other_id_list = [nid for nid in var_id_less_50 if all_node_value_label[nid] != cur_node_value_label and all_node_type_label[nid] == cur_node_type_label]
                        
                        
                        if len(type_ane_value_correct_id_list) == 0 or len(type_ane_value_correct_id_list) == 0:
                            # 没有50个词以内的同名节点。
                            continue
                        shuffle(other_id_list)
                        shuffle(type_ane_value_correct_id_list)
                        correct_candidate_id = type_ane_value_correct_id_list[0]
                        candidate_node_ids = [
                            correct_candidate_id
                        ] + other_id_list[:self.max_variable_candidates -
                                                    1]

                        num_scope_padding = self.max_variable_candidates - len(
                        candidate_node_ids)
                        candidate_node_ids_mask = [True] * len(
                        candidate_node_ids) + [False] * num_scope_padding

                        candidate_node_ids = candidate_node_ids + [
                            0
                        ] * num_scope_padding  # 在导入数据阶段，转化为正负。
                        cur_var_location_label = 0


                        """
                        cur_var_location_label = 0
                        """

                        """
                        # 对于next变量。我们先试试选择所有变量进行对比效果如何吧？
                        # 不太行，我们试试选择10个节点。
                        type_ane_value_correct_id_set = {nid for nid in range(node_id-50, node_id) if all_node_type_label[nid] == cur_node_type_label and all_node_value_label[nid] == cur_node_value_label}
                        other_id_list = list(set(range(node_id-50, node_id)) - type_ane_value_correct_id_set)
                        type_ane_value_correct_id_list = list(type_ane_value_correct_id_set)
                        if len(type_ane_value_correct_id_set) == 0:
                            # 没有50个词以内的同名节点。
                            continue

                        shuffle(other_id_list)
                        shuffle(type_ane_value_correct_id_list)
                        cur_var_location_label = 0
                        cur_candidate = [type_ane_value_correct_id_list[0]] + other_id_list[:self.max_variable_candidates-1] 
                        candidate_node_ids_mask = [True] * self.max_variable_candidates
                        """
                        """
                        cur_var_location_label = -1
                        for k in range(node_id-1, node_id-1-50, -1):
                            if all_node_type_label[k] == cur_node_type_label and all_node_value_label[k] == cur_node_value_label:
                                cur_var_location_label = k - (node_id-50)
                                break
                        if cur_var_location_label == -1:
                            # 即没有找到相同的变量则跳过。
                            continue
                        """


                        data = PYGraphData()
                        data.load_attr(
                            edge_index=cur_sub_graph_one,
                            #edge_index_two_hop=cur_sub_graph_two,
                            #edge_index_three_hop=cur_sub_graph_three,
                            #edge_index_five_hop=cur_sub_graph_five,
                            edge_index_ncs=cur_ncs_adj,
                            edge_index_dfg=cur_dfg_adj,
                            edge_index_last_write=cur_last_write_adj,
                            edge_index_last_use=cur_last_use_adj,
                            edge_index_self_loop=cur_self_loop_adj,
                            value_label=torch.tensor([cur_node_value_label]).type(torch.LongTensor),
                            type_label=torch.tensor([cur_node_type_label]).type(torch.LongTensor),
                            label=torch.tensor([cur_var_location_label]),
                            x=torch.tensor(cur_node_value).type(torch.LongTensor),
                            x_type=torch.tensor(cur_node_type).type(torch.LongTensor),
                            right_most=cur_right_most_node_id,
                            candidate_id=torch.tensor(candidate_node_ids).type(torch.LongTensor)-(node_id-self.max_node_per_graph),
                            candidate_id_mask=torch.tensor(candidate_node_ids_mask).type(torch.FloatTensor)
                        )
                        torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(data_i)))
                        data_i += 1
                        if data_i > self.max_graph:
                            return
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
    
    with open(r"vocab_dict\python_terminal_dict_1k_value.json", "r") as f:
        value_dict = json.load(f)
    with open(r"vocab_dict\python_terminal_dict_1k_type.json", "r") as f:
        type_dict = json.load(f)
    gd = PYGraphDataset("data/py150/validate_data", value_dict=value_dict, type_dict=type_dict)
    