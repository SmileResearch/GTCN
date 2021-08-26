# encoding:utf-8
import os
import json
from collections import Counter
import torch
import gzip

def statistic_raw_graph_node_nums(path: str,
                             suffix: str = ".json"):
    # 统计path下面的所有json文件。

    def count_single_json_terminal_dict(raw: str, graph_node_count):
        cur_graph_node_nums = len(raw)
        temp = cur_graph_node_nums//10*10
        graph_node_count[temp] = graph_node_count[temp] + 1
        #candidateNodeList = json_dict['candidateNodeList']  # list
        '''
        for node in raw:
            node_value = node.get("value", "EMPTY")
            node_type = node.get("type")

            value_terminal_dict[node_value] = value_terminal_dict[node_value] + 1
            type_terminal_dict[node_type] = type_terminal_dict[node_type] + 1
        '''

    graph_node_count = Counter()
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(suffix):
                continue
            absolute_file_path = os.path.join(root, file)
            with open(absolute_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    raw = json.loads(line)
                    count_single_json_terminal_dict(raw, graph_node_count)

    graph_node_count_sorted = graph_node_count.most_common()
    return graph_node_count_sorted



def statistic_torch_graph_node_nums_and_label(path: str,
                             suffix: str = ".pt"):
    # 统计path下面的所有json文件。

    def count_single_json_terminal_dict(data: str, graph_node_count):
        label=1000
        graph_node = data.x.shape[0] // 10 * 10

        graph_node_count[graph_node] = graph_node_count[graph_node] + 1
        if data.value_label == 0:  #"EMPTY"==0 , "UNK"==1000
            return 1
        else:
            return 0
        #candidateNodeList = json_dict['candidateNodeList']  # list
        '''
        for node in raw:
            node_value = node.get("value", "EMPTY")
            node_type = node.get("type")

            value_terminal_dict[node_value] = value_terminal_dict[node_value] + 1
            type_terminal_dict[node_type] = type_terminal_dict[node_type] + 1
        '''

    graph_node_count = Counter()
    graph_label_not_in_dict_count = 0
    graph_nums = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(suffix) or not file.startswith("data"):
                continue
            absolute_file_path = os.path.join(root, file)
            data = torch.load(absolute_file_path)
            graph_label_not_in_dict_count += count_single_json_terminal_dict(data, graph_node_count)
            graph_nums += 1

    graph_node_count_sorted = graph_node_count.most_common()
    return graph_node_count_sorted, graph_label_not_in_dict_count, graph_nums

def statistic_learning_graph_node_nums_and_label(path: str,
                             suffix: str = ".jsonl.gz"):
    # 统计path下面的所有json文件。

    def count_single_json_terminal_dict(raw: str, graph_node_count):
        ContextGraph = raw["ContextGraph"]
        SymbolCandidates = raw["SymbolCandidates"]  # 候选词个数
        graph_node = len(ContextGraph["NodeLabels"]) // 150 * 150

        # 现在得到的观察是150以下2674， 以上大约5000左右，由于候选词都在1-10之间。我们将图固定到150应该问题不大。
        graph_node_count[graph_node] = graph_node_count[graph_node]+1
        #graph_node_count[]
        return 5 if len(SymbolCandidates)>5 else len(SymbolCandidates)
        #candidateNodeList = json_dict['candidateNodeList']  # list
        '''
        for node in raw:
            node_value = node.get("value", "EMPTY")
            node_type = node.get("type")

            value_terminal_dict[node_value] = value_terminal_dict[node_value] + 1
            type_terminal_dict[node_type] = type_terminal_dict[node_type] + 1
        '''

    graph_node_count = Counter()
    graph_candidate_count = 0
    graph_nums = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(suffix):
                continue
            absolute_file_path = os.path.join(root, file)
            with gzip.open(absolute_file_path, "r") as f:
                raw_data = f.read()
                data_utf8 = raw_data.decode("utf-8")
                for raw_line in data_utf8.split("\n"):
                    if len(raw_line) == 0:
                        continue
                    raw = json.loads(raw_line)
                    graph_candidate_count += count_single_json_terminal_dict(raw, graph_node_count)
                    graph_nums += 1

    graph_node_count_sorted = graph_node_count.most_common()
    return graph_node_count_sorted, graph_candidate_count, graph_nums
    """ 37575
        12055
        占比0.32.说明有百分之30的准确率选出正确的。
    """


data_dir = r"W:\Scripts\Python\Vscode\Federated_DGAP\data\learning"

if __name__ == '__main__':

    graph_node_count_sorted, graph_label_not_in_dict_count, graph_nums = statistic_learning_graph_node_nums_and_label(data_dir)
    print(graph_node_count_sorted)
    print(graph_label_not_in_dict_count)
    print(graph_nums)