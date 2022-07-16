from treelib import Node, Tree
import os
import json

data_dir = "data/pytemp/raw"

for roots, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(roots, file)
        print(path)
        with open(path, "r") as f:
            for index, line in enumerate(f.readlines()):
                if index>20:
                    break
                data = json.loads(line)
                ast_node_list = data
                tree = Tree()
                node_parent = dict()
                for node_id, ast_node in enumerate(ast_node_list):
                    node_children = ast_node.get("children", None)
                    if node_children is not None:
                        # 添加子节点到父节点的映射
                        for node in node_children:
                            node_parent[node] = node_id
                for node in range(len(ast_node_list)):
                    tree.create_node(ast_node_list[node]["type"]+ast_node_list[node].get("value", "None"), node, parent=node_parent.get(node, None), data={"type":ast_node_list[node]["type"], "value":ast_node_list[node].get("value", "EMPTY")})
                tree.to_graphviz(f"data/graphviz_show/{index}.dot")

                