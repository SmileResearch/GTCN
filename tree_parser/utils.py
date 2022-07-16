import re
from io import StringIO
import  tokenize
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(root_node):
    # 遍历整颗树，获取每个叶子节点（非注释节点）的（起始位置、终止位置）。
    # code tokens 存储的是代码 中 每个token对应的起始位置和终止位置的集和。
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens

def tree_variable_index_to_ast_index(root_node):
    """
    It traverses the tree in pre-order, and for each node, it checks if it's a leaf node (i.e. a
    variable) or not. If it is, it adds the node's index to the dictionary. If it isn't, it adds the
    node's children to the stack
    
    Args:
      root_node: the root node of the AST tree
    
    Returns:
      A dictionary mapping the index of a variable in the tree to its index in the AST.
    """
    # 获取树的变量在AST树中的下标
    # 通过先序遍历获取
    variable_index_to_ast_index = dict()
    traversal_stack = [root_node]
    node_count = -1 # 节点编号从0开始。
    leave_count = -1
    while traversal_stack:
        cur_node = traversal_stack.pop()
        node_count+=1
        if (len(cur_node.children)==0 or cur_node.type=='string') and cur_node.type!='comment':
            leave_count += 1
            variable_index_to_ast_index[leave_count] = node_count
        else:
            traversal_stack.extend(cur_node.children[::-1]) # 倒序入栈，保证先序遍历。
    return variable_index_to_ast_index
    
    
def tree_to_leave_list(root_node):
    # 遍历整颗树，获取每个叶子节点（非注释节点）的（起始位置、终止位置）。
    # code tokens 存储的是代码 中 每个token对应的起始位置和终止位置的集和。
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [root_node]
    else:
        node_list=[]
        for child in root_node.children:
            node_list+=tree_to_leave_list(child)
        return node_list
    
def tree_to_node_list(root_node):
    # code tokens 存储的是代码 中 每个token对应的起始位置和终止位置的集和。
    traversal_stack = [root_node]
    node_list = list()
    while traversal_stack:
        cur_node = traversal_stack.pop()
        node_list.append(cur_node)
        if (len(cur_node.children)==0 or cur_node.type=='string') and cur_node.type!='comment':
            continue
        else:
            traversal_stack.extend(cur_node.children[::-1]) # 倒序入栈，保证先序遍历。
        
    return node_list
    
def tree_to_variable_index(root_node,index_to_code):
    # 给一个节点，在index_to_code中找到它的index
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        index=(root_node.start_point,root_node.end_point)
        _,code=index_to_code[index]
        if root_node.type!=code:
            return [(root_node.start_point,root_node.end_point)]
        else:
            return []
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_variable_index(child,index_to_code)
        return code_tokens    

def index_to_code_token(index,code):
    """
    Given a start and end index, return the corresponding code token
    
    Args:
      index: a tuple of two tuples, each of which is a tuple of two integers.
      code: the code to be tokenized
    
    Returns:
      A string of code.
    """
    # 根据code，将index映射为code。
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
   