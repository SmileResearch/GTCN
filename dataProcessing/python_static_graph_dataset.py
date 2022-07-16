from math import trunc
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.utils import to_undirected, add_self_loops, subgraph, remove_self_loops
from os import listdir, mkdir
import os.path as osp
import json

from torch_geometric.loader import DataLoader
from torch_geometric.utils.undirected import is_undirected
import numpy as np
import json
from random import shuffle
import random
import os
import gzip
from tqdm import tqdm
from multiprocessing import cpu_count

from collections import defaultdict

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1

BIG_NUMBER = 1e7

from tree_parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from tree_parser import (remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, tree_to_variable_index,
                    tree_variable_index_to_ast_index, tree_to_node_list)
from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

#load parsers
LANGUAGE = Language('tree_parser/my-languages.so', "python")
parser = Parser()
parser.set_language(LANGUAGE)
parser = [parser, dfg_function["python"]]


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

    def load_attr(self, ast, ncs, comesFrom, computedFrom, ast_reverse, ncs_reverse, comesFrom_reverse, computedFrom_reverse, selfLoop, x, slot_id, candidate_ids, candidate_masks, label, value_label):
        """由于dataset的限制，不能初始化的时候传入参数，所以设立一个函数传入参数。
            # 所有的边的名称必须以index结尾
        Args:
            edge_index ([type]): ast one hop edge index
            value_label ([type]): the label of node value
            type_label ([type]): the label of node type
            x ([type]): the initial embedding of node value
            x_type ([type]): the initial embedding of node type
            right_most ([type]): the right most node id of a graph
        """
        self.ast_index = ast
        self.ncs_index = ncs
        self.comesFrom_index = comesFrom
        self.computedFrom_index = computedFrom
        self.ast_reverse_index = ast_reverse
        self.ncs_reverse_index = ncs_reverse
        self.comesFrom_reverse_index = comesFrom_reverse
        self.computedFrom_reverse_index = computedFrom_reverse
        self.selfLoop_index = selfLoop
        self.x = x
        self.slot_id = slot_id
        self.candidate_ids = candidate_ids
        self.candidate_masks = candidate_masks
        self.label = label
        self.value_label=value_label


def extract_edges(ast_node_list, tree, node_parent):
    """
    > This function takes a list of AST nodes and returns a list of edges
    return edge list: ast, ddg, ncs.
    """

    def is_var(node):
        return node.is_leaf() and node.data["value"] != "EMPTY"

    ast_edge_index = []
    ncs_edge_index = []
    dfg_edge_index = []
    nodeDict = {}
    # 变量的id
    var_id_list = []

    for_id = []
    if_id = []
    assign_left_child = []
    if_index = -1
    var_name_dict = {}  # 用于对同名变量串联的dict。

    ncs_pre_node = None
    # 提取ddg
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
        if node_type == "orelse" and if_index != -1:
            if_id[if_index].append(node_id)
        if node_type == "Assign":
            assign_left_child.append(node_id + 1)  # Assign node left child. 被赋值的那个节点。

        if node_children is not None:
            ast_edge_index.extend([[node_id, tgt_id] for tgt_id in node_children])

        nodeDict[node_id] = {"type": node_type, "value": node_value}

        if node_children is None:
            # 叶子节点 并不一定是一个变量。
            pass

        # 添加NCS边
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
    #tree.show()
    # 所有排列都是先序深度优先遍历的。
    # get all vars
    all_var = [node.identifier for node in tree.all_nodes() if is_var(node)]
    # 获取所有的被赋值的变量。
    all_assign_left_var = [node.identifier for alc in assign_left_child for node in tree.leaves(alc) if is_var(node)]
    # get all f min max var node.
    # 将for下面的变量乘以2.
    for f_id in for_id:
        f_leaves = [node.identifier for node in tree.leaves(f_id) if is_var(node)]
        min_f_leaves = min(f_leaves)
        max_f_leaves = max(f_leaves)
        min_f_leaves_index = all_var.index(min_f_leaves)
        max_f_leaves_index = all_var.index(max_f_leaves)
        all_var = all_var[:min_f_leaves_index] + all_var[min_f_leaves_index:max_f_leaves_index +
                                                         1] * 2 + all_var[max_f_leaves_index + 1:]

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

    # 提取DDG
    last_use_edge_index = []
    last_write_edge_index = []
    last_use_dict = {}  # dict[valueName] = list(lastNode_id)
    last_write_dict = {}

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

    return ast_edge_index, ncs_edge_index, dfg_edge_index, nodeDict, var_id_list, last_use_edge_index, last_write_edge_index


def get_varmisuse_nodes(root, max_node_per_graph):
    # 通过ast tree 获取所有候选词和slot_nodes
    # 返回候选词节点。
    # 第一次遍历得到的词进入候选词（变量的申明）。当出现过一次的词进入slot_nodes

    # 遍历。
    # 传入的参数进入候选词
    # 深度优先遍历第一次出现的词进入候选词

    traversal_stack = [root]
    seen = set()
    slot_nodes = []  # 槽节点
    candidate_nodes = []  # 候选节点。 第一次出现、函数参数作为候选节点
    node_count = 0
    while traversal_stack and node_count < max_node_per_graph:
        cur_node = traversal_stack.pop()
        node_count += 1
        if cur_node.type == "comment":
            # 注释节点
            continue
        elif cur_node.type == "function_definition":
            # 先入栈其他孩子。后入栈参数。
            for node in cur_node.children:
                if node.type != "parameters":
                    traversal_stack.append(node)

            # 函数定义模块。将他们首先入栈。
            # parameters只是一个节点
            parameters = cur_node.child_by_field_name("parameters")

            traversal_stack.append(parameters)
        else:
            if (len(cur_node.children) == 0 or cur_node.type == 'string'):
                # 叶子节点
                if cur_node.type == "identifier":
                    # 包括了函数名节点和变量节点。
                    if cur_node.text in seen:
                        # 见过这个节点
                        slot_nodes.append(cur_node)
                    else:
                        # 没见过
                        seen.add(cur_node.text)
                        candidate_nodes.append(cur_node)
            else:
                # 非叶子节点
                # 倒序入栈。保证是先序深度优先遍历。
                traversal_stack.extend(cur_node.children[::-1])

    return slot_nodes, candidate_nodes


def extract_data_flow(code, parser, root_node, lang="python"):

    code = code.split('\n')
    tokens_index = tree_to_token_index(root_node)  # [(start, end) for leaves nodes] 只获取了叶子节点的。
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]  # [string code of (start, end)]

    index_to_code = {}  # 一个字典，将tokens index（start, end) 映射为 idx和代码。
    for idx, (index, code_temp) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code_temp)  # index_to_code = dict((start,end):(叶子节点的index, 代码))
    try:
        DFG, _ = parser[1](root_node, index_to_code, {})
    except:
        DFG = []
    DFG = sorted(DFG, key=lambda x: x[1])
    # DFG: (code, idx, "edgeType", [codes],[idxs])

    indexs = set()
    for d in DFG:
        if len(d[-1]) != 0:
            indexs.add(d[1])
        for x in d[-1]:
            indexs.add(x)
    new_DFG = []
    for d in DFG:
        if d[1] in indexs:
            new_DFG.append(d)
    dfg = new_DFG

    # 上部分代码提取出了dfg，但是dfg的下标是变量的顺序，
    # 下面的代码将变量下标映射为AST中节点的下标。

    dfg_edge_dict = dict()  # 存储了边类型到 边的映射。
    for d in dfg:
        if d[2] not in dfg_edge_dict:
            dfg_edge_dict[d[2]] = list()
        for from_idx in d[4]:
            # 将d[4]中的idx和d[1]组合成边
            dfg_edge_dict[d[2]].append([from_idx, d[1]])

    # 将叶子节点的下标转为AST节点的下标。

    # 按照先序优先遍历root，获取
    index_to_ast_index = tree_variable_index_to_ast_index(root_node)
    # 打印节点的code，然后通过ast_index找到nodeList 找到code

    dfg_trans_edge = {}
    for key, value in dfg_edge_dict.items():
        # 将dfg的下标转为ast node 的下标。
        dfg_trans_edge[key] = [[index_to_ast_index[d1], index_to_ast_index[d2]] for d1, d2 in value]

    return dfg_trans_edge


def extract_ast_flow(func, root_node, lang):
    # 获取AST边
    # 获取所有的AST NOde，获取ASTnode到下标的map。
    # 然后添加节点到子节点的边
    ast_edge = []

    # 获取父节点的下标，然后添加父节点到子节点的边的连接。
    # 使用一个父节点的stack

    traversal_stack = [root_node]
    traversal_index_stack = [None]
    count = -1
    while traversal_stack:
        count += 1
        cur_node = traversal_stack.pop()
        cur_node_parent_index = traversal_index_stack.pop()
        if cur_node_parent_index != None:
            ast_edge.append([cur_node_parent_index, count])

        if (len(cur_node.children) == 0 or cur_node.type == 'string') and cur_node.type != 'comment':
            # 叶子节点，没有子节点了。
            continue
        else:
            traversal_index_stack.extend([count] * len(cur_node.children))
            traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
    return ast_edge


def extract_ncs_flow(func, root_node, lang):
    # 获取ncs边
    # 获取所有的AST Node，按照先序深度优先遍历，然后连接各个节点
    # 然后添加节点到子节点的边
    node_list = tree_to_node_list(root_node)
    ncs_edge = []
    # 先序深度优先遍历感觉像。
    # 所以NCS就是各个节点顺序连接。
    pre_node_range = range(len(node_list) - 1)
    next_node_range = range(1, len(node_list))
    ncs_edge = list(zip(pre_node_range, next_node_range))

    return ncs_edge


def choose_varmisuse_candidate(root_node, num_candidates, max_node_per_graph):
    # 提取所有的slot节点和候选词节点
    all_slot_node_list, all_candidate_node_list = get_varmisuse_nodes(root_node, max_node_per_graph)

    # 候选词数量不足则退出。这里图方便就退出了。
    if len(all_candidate_node_list) < num_candidates:
        raise Exception("候选词数量不足")

    # 选择slot词
    choose_slot_node_index = random.randint(0, len(all_slot_node_list) - 1)
    slot_node = all_slot_node_list[choose_slot_node_index]

    # 选择候选词
    true_candidate_node = [node for node in all_candidate_node_list if node.text == slot_node.text][0]
    choose_candidate_nodes = random.sample(all_candidate_node_list, num_candidates)
    choose_candidate_nodes = [node for node in choose_candidate_nodes if node.text != slot_node.text]
    # 最终确定的候选词。 候选词的数量为 args.num_candidates
    candidate_nodes = [true_candidate_node] + choose_candidate_nodes[:num_candidates - 1]

    # 转换为index
    all_node_list = tree_to_node_list(root_node)
    slot_index = all_node_list.index(slot_node)
    candidate_index = [all_node_list.index(c) for c in candidate_nodes]

    return slot_index, candidate_index


def get_all_node_exper(root_node, code):
    """
    Given a root node, return a dictionary that maps the index of each node to its value and type
    
    Args:
      root_node: the root node of the AST
      code: the code to be parsed
    
    Returns:
      index2exper: index to exper
        index2value: index to value
        index2type: index to type
    """

    # 非叶子节点获取节点type
    # 叶子节点获取节点的value
    code = code.split("\n")
    index2exper = {}  # index 2 exper
    index2value = {}
    index2type = {}
    traversal_stack = [root_node]
    count = -1
    while traversal_stack:
        cur_node = traversal_stack.pop()
        count += 1
        value = index_to_code_token((cur_node.start_point, cur_node.end_point), code)
        type = cur_node.type
        # TODO: 这里有问题
        index2value[count] = value
        index2type[count] = type
        if (len(cur_node.children) == 0 or cur_node.type == 'string') and cur_node.type != 'comment':
            # 叶子节点 节点值为value
            index2exper[count] = value
        else:
            # 非叶子节点 节点值为type
            index2exper[count] = type
            traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
        # print(f"count {count},  value:{index2value[count]}, type:{index2type[count]}, exper:{index2exper[count]}")

    return index2exper, index2value, index2type


def trans_string_to_embedding(expr_dict, graph_node_max_chars, num_nodes=None):
    """
    It takes a dictionary of nodes and their string, and returns a matrix of the same size, where each
    row is the embedding of the corresponding node
    
    Args:
      expr_dict: a dictionary of the expression, where the keys are the node IDs and the values are the
    node labels.
      graph_node_max_chars: The maximum number of characters in a node's string.
      num_nodes: The number of nodes in the graph.
    Return:
      node_value_chars: The embedding numpy array of node string.
    """

    # 将表达转为embedding
    if not num_nodes:
        num_nodes = max(expr_dict.keys()) + 1
    node_value_chars = np.zeros(shape=(num_nodes, graph_node_max_chars), dtype=np.uint8)
    for node, node_value in expr_dict.items():
        if node >= num_nodes:
            # 超出了范围就截断
            continue
        try:
            for (char_idx, value_char) in enumerate(node_value[:graph_node_max_chars].lower()):
                node_value_chars[int(node), char_idx] = ALPHABET_DICT.get(value_char, 1)
        except Exception as e:
            print("char idx", char_idx)
            print("value_char", value_char)
            print("node_value_chars shape", node_value_chars.shape)
            print("ALPHABET_DICT.get(value_char, 1)", ALPHABET_DICT.get(value_char, 1))
            print(e)
            raise e
    return node_value_chars

def reverse_edge(edge_index):
    if edge_index:
        return [[e[1],e[0]] for e in edge_index]
    else:
        return None

def convert_examples_to_features(item,
                                 num_candidates,
                                 max_node_per_graph,
                                 slot_singal,
                                 graph_node_max_chars,
                                 value_dict,
                                 lang="python"):
    func, sha, cache = item

    # TODO: 做截断和补全。
    # 1. 不在每个图添加自连接边
    # 2. 添加一个单独的自连接边
    # 3. 每个边添加一个反向边
    # 4. 做截断。截断目标节点或源节点大于的边

    # 尝试移除注释
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass

    tree = parser[0].parse(bytes(func, 'utf8'))
    root_node = tree.root_node

    #extract data flow
    dfg_edge = extract_data_flow(func, parser, root_node, lang)

    #   有关截断和补全
    #   trans_list_to_edge_tensor 会做补全, trans_string_to_embedding也会做补全
    #   trans_list_to_edge_tensor做图的截断。 choose_varmisuse_candidate做节点截断，trans_string_to_embedding做节点表达式的截断

    ast_edge = extract_ast_flow(func, root_node, lang)
    ncs_edge = extract_ncs_flow(func, root_node, lang)
    
    # 转换slot词、候选词在句子中的位置。
    try:
        slot_index, candidate_index = choose_varmisuse_candidate(root_node, num_candidates, max_node_per_graph)
    except Exception as e:
        print(f"extract slot, candidate Exception:", e)
        return None

    # 获取节点的features，如果是叶子节点，这获取节点的代码，否则获取节点的type
    nodeindex2exper, nodeindex2value, nodeindex2type = get_all_node_exper(root_node, func)
    # 设置slot节点的expr 为 slot_singal
    #print("slot node index", nodeindex2exper[slot_index])
    #print("candidate node index", [nodeindex2exper[index] for index in candidate_index])
    
    # slot节点的变量值。
    slot_node_value = nodeindex2exper[slot_index]
    
    nodeindex2exper[slot_index] = slot_singal
    try:
        node_embedding = trans_string_to_embedding(nodeindex2exper, graph_node_max_chars, num_nodes=max_node_per_graph)
    except Exception as e:
        return None

    # 检查是否有不合格的边, 设置为None
    if not ast_edge:
        return None
    comesFrom = dfg_edge.get("comesFrom", None)
        
    computedFrom = dfg_edge.get("computedFrom", None)
    edges = {}
    for name, edge in zip(["ast", "ncs", "comesFrom", "computedFrom"], [ast_edge, ncs_edge, comesFrom, computedFrom]):
        # 进行一些预处理，
        # 转为tensor, 处理None Yes
        # 截断 Yes
        # 添加自连接边 No
        # 转为无向图 No
        edges[name] = trans_list_to_edge_tensor(edge, max_node_per_graph=max_node_per_graph, truncate=True, is_add_self_loop=True, is_to_undirected=True)
        edges[name+"_reverse"] = trans_list_to_edge_tensor(reverse_edge(edge), max_node_per_graph=max_node_per_graph, truncate=True, is_add_self_loop=True, is_to_undirected=True)

    edges["selfLoop"] = trans_list_to_edge_tensor(None, is_add_self_loop=True, max_node_per_graph=max_node_per_graph, is_to_undirected=False, truncate=False)

    value_label = value_dict.get(slot_node_value, len(value_dict))

    data = PYGraphData()
    data.load_attr(
        **edges,
        x=torch.tensor(node_embedding).type(torch.LongTensor),
        slot_id=torch.tensor(slot_index).type(torch.LongTensor),
        candidate_ids=torch.tensor(candidate_index).type(torch.LongTensor),
        candidate_masks=torch.tensor([True] * len(candidate_index)).type(torch.FloatTensor),
        label=torch.tensor([0]).type(torch.LongTensor),
        value_label = torch.tensor([value_label]).type(torch.LongTensor)
    )

    return data


def trans_list_to_edge_tensor(edge_index,
                              is_add_self_loop=True,
                              max_node_per_graph=None,
                              is_to_undirected=True,
                              truncate=True):
    """将[src,tgt]转为输入到data对象中的edge_index，即tensor([src...],[tgt...])
        加上自连接边、加上双向边。
        默认情况下，添加自连接边，转为无向图，图做截断。

    Args:
        edge_index ([type]): [src,tgt]
        max_node_per_graph: 一个图最多的节点树，用于做截断、添加自连接边、转为无向图。
        add_self_loop (bool, optional): 是否加自连接边. Defaults to True.
        to_undirected (bool, optional): 是否转为无向图. Defaults to True.
    Return:
        edge: 完成处理后的edge
    """
    # 做子图截断
    if truncate and edge_index is not None:
        # 对图截断
        edge_index = [e for e in edge_index if e[0]<max_node_per_graph and e[1]<max_node_per_graph]
        # t, attr = subgraph(torch.arange(start=0, end=max_node_per_graph), t)
        
        
    # 检查是否为空
    if edge_index is not None and len(edge_index) != 0:  # 当edge_index存在且有边时。
        t = torch.tensor(edge_index).t().contiguous()
    else:
        t = torch.tensor([]).type(torch.LongTensor).t().contiguous()

    # 添加自连接边
    if is_add_self_loop:
        t, weight = add_self_loops(t, num_nodes=max_node_per_graph)  # 这里强制加了0-150的节点。方便做subgraph不会报错。

    # 转为无向图
    if is_to_undirected and t.shape[0] != 0:
        # 转为无向图
        t = to_undirected(t)


    return t


class PYGraphDataset(Dataset):

    def __init__(self,
                 root,
                 value_dict,
                 type_dict,
                 max_graph,
                 graph_node_max_num_chars=19,
                 max_node_per_graph=512,
                 slot_singal="<SLOT>",
                 max_variable_candidates=5,
                 transform=None,
                 pre_transform=None):
        self.graph_node_max_chars = graph_node_max_num_chars
        self.max_node_per_graph = max_node_per_graph
        self.value_dict = value_dict
        self.type_dict = type_dict
        self.slot_singal = slot_singal
        self.max_variable_candidates = max_variable_candidates
        self.max_graph = max_graph
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

    def process(self):
        # Read data into huge `Data` list.
        # TODO:
        # 1. self loop 边 由模型添加。数据不添加
        # 2. 图进行截断。截断长度和graph code bert 相同。长度不足的，补齐相应的。
        # 3. 修改csharp代码，同样进行截断。长度不足的，补齐相应的。
        # 4. 根据graphcodebert逻辑选择候选词。
        # 5. 精简代码。

        data = []
        cache = {}
        for data_file in os.listdir(self.raw_dir):
            data_file = os.path.join(self.raw_dir, data_file)
            if data_file.endswith(".jsonl.gz"):
                with gzip.open(data_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        # original_string 是原始代码 sha是哈希码
                        data.append((js["original_string"], js["sha"], cache))

        #only use 10% valid data to keep best model
        if 'valid' in self.raw_dir:
            data = random.sample(data, int(len(data) * 0.1))

        examples = []

        idx = 0
        for x in tqdm(data[:self.max_graph], total=min(len(data), self.max_graph)):
            d = convert_examples_to_features(x,
                                         num_candidates=self.max_variable_candidates,
                                         max_node_per_graph=self.max_node_per_graph,
                                         slot_singal=self.slot_singal,
                                         graph_node_max_chars=self.graph_node_max_chars,
                                         value_dict=self.value_dict)
            if d is not None:
                # 将数据落盘
                torch.save(d, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
                idx+=1
        print("数据生成完毕!")

        if 'train' in self.raw_dir:
            # 展示前三个example
            for idx, example in enumerate(examples[:3]):
                print("*** Example ***")
                print("idx: {}".format(idx))
                print("label: {}".format(example.label))
                print("x: {}".format(example.x))
                print("slot_id: {}".format(example.slot_id))
                print("candidate_ids: {}".format(example.candidate_ids))
                print("candidate_masks: {}".format(example.candidate_masks))
                print("ast: {}".format(example.ast_index))
                print("ncs: {}".format(example.ncs_index))
                print("comesFrom: {}".format(example.comesFrom_index))
                print("computedFrom: {}".format(example.computedFrom_index))



# 转换为一个类，然后从其中获取特征拼接成batch_data
class PythonStaticGraphDatasetGenerator:
    def __init__(
            self,
            root,
            value_dict,
            type_dict,
            num_edge_types,
            batch_size,
            shuffle=True,
            graph_node_max_num_chars=19,
            max_node_per_graph=50,
            max_graph=20000,
            max_variable_candidates=5,
            num_workers=int(cpu_count() / 2),  # 并行数。目前官方库好像是用大于0有问题，待验证。如果有问题，设置为0
            slice_edge_type=None,
            slot_singal="<SLOT>",
            device="cpu",
            **kwargs,
    ):
        self.device = device
        self.num_edge_types = num_edge_types
        self.slice_edge_type=slice_edge_type
        self._dataset = PYGraphDataset(root, value_dict, type_dict, max_graph,graph_node_max_num_chars, max_node_per_graph, slot_singal,
                                             max_variable_candidates)
        self._dataset = self._dataset[:max_graph]
        self.data_nums = len(self._dataset)
        self.batch_iterator = DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def batch_generator(self):
        # 解释下为什么需要这个类来做一层接口：
        #   数据当中，边的种类数量会影响隐藏层权重的大小。所以各个数据集边的种类数量必须统一。
        #   但测试过程中，可能会需要测试不同种类的边排列组合的效果。所以使用统一的参数num_edge_types来确定边的数量。
        #   并规定edge_list的边种类数必须等于这个值。然后用edge_list统一的在模型中完成训练。
            
        for batch_data in self.batch_iterator:
            batch_data = batch_data.to(self.device)
            
            edge_list = [
                    batch_data.ast_index,
                    batch_data.ncs_index,
                    batch_data.comesFrom_index,
                    batch_data.computedFrom_index,
                    batch_data.selfLoop_index,
                    batch_data.ast_reverse_index,
                    batch_data.ncs_reverse_index,
                    batch_data.comesFrom_reverse_index,
                    batch_data.computedFrom_reverse_index,
                ]
            
            if self.slice_edge_type is not None:
                assert len(self.slice_edge_type)==self.num_edge_types, "数据集中边的数量不等于规定数量"
                edge_list = [edge_list[index] for index in self.slice_edge_type]
            else:
                edge_list = edge_list[:self.num_edge_types]
                
                
            data = {
                "x":
                batch_data.x,
                "edge_list": edge_list,
                "slot_id":
                batch_data.slot_id,
                "candidate_ids":
                batch_data.candidate_ids,
                "candidate_masks":
                batch_data.candidate_masks,
                "batch_map":
                batch_data.batch,
                "num_graphs":
                batch_data.num_graphs,
                "num_nodes":
                batch_data.num_nodes,
                "label":
                batch_data.label,
                "value_label":
                batch_data.value_label,
                "batch_data":
                batch_data,
            }
            yield data

if __name__ == '__main__':
    from torch_geometric.data import DataLoader

    # with open(r"vocab_dict\python_terminal_dict_1k_value.json", "r") as f:
    #     value_dict = json.load(f)
    # with open(r"vocab_dict\python_terminal_dict_1k_type.json", "r") as f:
    #     type_dict = json.load(f)
    gd = PYGraphDataset("data/pytemp", value_dict=None, type_dict=None, max_node_per_graph=10000)
