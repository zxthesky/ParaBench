import json

class Node:
    def __init__(self, information: str = None) -> None:
        self.information: str = information
        self.children: list = []
        self.parent: list = []
        self.layer: int = None

def read_file(filename):
    if filename.endswith(".json"):
        with open(filename, 'r') as f:
            all_data = json.load(f)
        return all_data
    elif filename.endswith(".jsonl"):
        all_data = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
        return all_data
    else:
        print("your filename is wrong")

def write_file(all_data, filename):
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=2)


def judge_parent_node(now_node: Node, parent_node: Node):
    """判断parent_node是否在now_node节点之上
    """
    if len(now_node.parent) == 0:
        return False
    elif now_node == parent_node:
        return True
    else:
        for parent in now_node.parent:
            judge = judge_parent_node(parent, parent_node)
            if judge:
                return True
        return False

            
            
####################################
####################################
#################################### 构造树结构，方便我们管理数据
####################################

def check_one_data_whether_circle(node: Node)->bool:
    """判断环是否是当前节点产生
    Args:
        node(Node): now_node
    Returns(bool): True means produce a circle
    """
    for children in node.children:
        if judge_parent_node(node, children):
            return True
    return False

def check_whether_contain_circle(has_node: dict)->bool:
    """
    Args:
        has_node(dict): {"information": node, ......} 所有的节点，需要对每个节点去判别
    Returns(bool): True means has circle
    """
    for information in has_node:
        now_node = has_node[information]
        if check_one_data_whether_circle(now_node):
            return True
        
    return False




def add_root_node(has_nodes: dict):
    """
    Args:
        has_node(dict): {information: node, ............}
    """
    all_candidate_node = []
    for information in has_nodes:
        now_node = has_nodes[information]
        if len(now_node.parent) == 0:
            all_candidate_node.append(now_node)
    if len(all_candidate_node) == 0:
        return -1
    root_node = Node()
    for candidate_node in all_candidate_node:
        root_node.children.append(candidate_node)
        candidate_node.parent.append(root_node)
    return root_node

def create_tree(all_edges, subtasks, check_data=False):
    """构成树
    Args:
        all_edges: [{"subtask1":..., "subtask2": ...}, ....]
        subtasks: ["subtask1", ..."subtaski", ...]
    """
    has_nodes = {} ######3 information: node, 字典，information对应其node
    for information in subtasks:
        has_nodes[information] = Node(information)

    for edge in all_edges:
        substep1 = edge["subtask1"]
        substep2 = edge["subtask2"]
        assert has_nodes.get(substep1, -1) != -1
        assert has_nodes.get(substep2, -1) != -1
        parent_node = has_nodes[substep1]
        children_node = has_nodes[substep2]
        
        parent_node.children.append(children_node)
        children_node.parent.append(parent_node)
    # if check_data:
    #     print(final_node.information)
    root_node = add_root_node(has_nodes)

    whether_a_tree = check_whether_contain_circle(has_nodes) # True表示有环，不是一个树
    if whether_a_tree:
        return False
    else:
        return root_node

####################################
####################################
####################################
####################################


def dfs(root_node: Node, ifo_2_node:dict):
    """深度优先遍历，遍历整棵树，获得dict [{information: node} ...] information表示当前节点的信息，node代表当前节点
    Args:
        root_node:当前节点
        ifo_2_node: [{information: node} ...]  text :  Node
    """
    if root_node.information != None and ifo_2_node.get(root_node.information, -1) == -1:
        ifo_2_node[root_node.information] = root_node
    for children_node in root_node.children:
        dfs(children_node, ifo_2_node)



def get_tree_deepth(root_node, information_2_layer):
    max_deepth = root_node.layer
    for node in root_node.children:

        node.layer = root_node.layer + 1

        information_2_layer[node.information] = root_node.layer
        max_deepth = max(max_deepth, get_tree_deepth(node, information_2_layer))

    return max_deepth

def get_tree_width(root_node, layer_2_node: dict = {}, now_layer: int = 0):

    if layer_2_node.get(now_layer, -1) == -1:
        layer_2_node[now_layer] = []
    if root_node not in layer_2_node[now_layer]:
        layer_2_node[now_layer].append(root_node)

    for node in root_node.children:
        get_tree_width(node, layer_2_node, now_layer+1)

def get_tree_deepth_and_width(root_node: Node):

    root_node.layer = 0
    layer_id_num = {}
    information_2_layer = {}
    deepth = get_tree_deepth(root_node, information_2_layer)
    get_tree_width(root_node, layer_id_num)
    width = max([len(layer_id_num[i]) for i in layer_id_num])
    return deepth, width, information_2_layer


    
################  用来获得所有node的直接上级依赖
def visited_node(now_node: Node, node2_dependency: dict):
    if (len(now_node.parent)) == 1 and (now_node.parent[0].information == None): ### 上面的是空的根节点
        for node in now_node.children:
            visited_node(node, node2_dependency)
    else:
        for parent_node in now_node.parent:
            if node2_dependency.get(now_node.information, -1) == -1:
                node2_dependency[now_node.information] = [parent_node.information]
            else:
                node2_dependency[now_node.information].append(parent_node.information)
        for node in now_node.children:
            visited_node(node, node2_dependency)


def get_order_dependency(root_node: Node) -> dict:
    """
    用来获得子任务依赖， 有些节点必须在某些节点之后完成。
    Used to obtain subtask dependencies. Some nodes must be completed after a certain node.

    Args:
        root_node: now node we need
    Returns:
        dict: {information1:[information2, information3]} 表示information2, information3需要在information1的上面

    """
    node2_dependency = {} ## 当前node需要在某个node之后 如information1:[information2, information3]表示information2, information3需要在information1的上面
    
    for node in root_node.children:
        visited_node(node, node2_dependency)
    return node2_dependency
##############################################
##############################################

def judge_one_whether_parent_node(test_node: Node, parent_node: Node):
    """判断当前node是否符合要求，parent_node在树里面应该在test_node父节点
    """
    judge = judge_parent_node(test_node, parent_node)
    return judge


def judge_nodes_whether_valid(test_node:Node, parent_nodes:list) -> bool:
    """判断当前node是否符合要求，gold_answer中当前节点的
    """
    for parent_node in parent_nodes:
        judge = judge_one_whether_parent_node(test_node, parent_node)
        if judge == False:
            return False
    return True

##############################################
##############################################


def compute_metric_one_data(gold_answer_root_node: Node, predict_answer_root_node: Node):
    """用来计算 一条数据的指标

    Args:
        gold_answer_root_node: the root node of gold answer's tree
        predict_answer_root_node: the root node of predict answer's tree
    """
    pass_or_not = True

    node2_dependency = get_order_dependency(gold_answer_root_node)
    gold_depth, gold_width, gold_information_2_layer = get_tree_deepth_and_width(gold_answer_root_node)
    predict_depth, predict_width, information_2_layer = get_tree_deepth_and_width(predict_answer_root_node)
    
    ifo_2_node = {}
    dfs(predict_answer_root_node, ifo_2_node)
    for now_information in node2_dependency:
        now_node = ifo_2_node[now_information]
        parent_nodes = [ifo_2_node[parent_information] for parent_information in node2_dependency[now_information]]
        pass_or_not = judge_nodes_whether_valid(now_node, parent_nodes)
        if pass_or_not == False:
            break

    return {
        "pass_or_not": pass_or_not,
        "predict_depth": predict_depth,
        "predict_width": predict_width,
        "gold_depth": gold_depth,
        "gold_width": gold_width
    }
        
def change_all_str_to_lower(all_data)->list:
    """将所有的边都转化为小写，方便后面的比较
    Args:
        all_data: list ,每个数据是一个dict, dict:{"task": ..., "subtasks": [...], "gold_dependencies": [....], "predict_dependencies": [...]}
                        gold_dependencies代表金标答案的边, predict_dependencies代表预测的边
    Returns(dicr): the metric of the dataset
    """
    final_data = []
    for data in all_data:
        try:
            temp_data = {}
            temp_data["task"] = data["task"].lower()
            temp_data["subtasks"] = [i.lower() for i in data["subtasks"]]
            gold_dependencies = []
            predict_dependencies = []
            for gold_edge in data["gold_dependencies"]:
                temp_gold_edge = {"subtask1": gold_edge["subtask1"].lower(), "subtask2": gold_edge["subtask2"].lower()}
                gold_dependencies.append(temp_gold_edge)
            
            for predict_edge in data["predict_dependencies"]:
                temp_predict_edge = {"subtask1": predict_edge["subtask1"].lower(), "subtask2": predict_edge["subtask2"].lower()}
                predict_dependencies.append(temp_predict_edge)
            temp_data["gold_dependencies"] = gold_dependencies
            temp_data["predict_dependencies"] = predict_dependencies

            final_data.append(temp_data)
        except:
            continue

    return final_data

def whether_generate_wrong_subtask(subtasks, dependencies):
    """看dependencies中是否都是subtask的边
    """
    for edge in dependencies:
        if (edge["subtask1"] not in subtasks) or (edge["subtask2"] not in subtasks):
            return True
    return False

def eval_all_data(all_data: list, wrong_number: int=0)->dict:
    """用来计算所有数据的指标
    Args:
        all_data(list): ,每个数据是一个dict, dict:{"task": ..., "subtasks": [...], "gold_dependencies": [....], "predict_dependencies": [...]}
                        gold_dependencies代表金标答案的边, predict_dependencies代表预测的边
        wrong_number(int): 初始化错误数据，模型生成结果有问题等等
    Returns(dicr): the metric of the dataset
    """
    all_data = change_all_str_to_lower(all_data)
    pass_data_number = 0
    all_length = 0

    predict_all_depth = 0
    predict_all_width = 0

    gold_all_depth = 0
    gold_all_width = 0

    #############################下面这个指标不管是否pass都计算depth，只要predict生成了树
    all_all_length = 0

    all_predict_all_depth = 0
    all_predict_all_width = 0

    all_gold_all_depth = 0
    all_gold_all_width = 0
    ##############################
    unable_create_tree = 0
    hallucination_number = 0
    for data in all_data:
        whether_generate_wrong_subtask_judge = whether_generate_wrong_subtask(data["subtasks"], data["predict_dependencies"])
        if whether_generate_wrong_subtask_judge == False:
            gold_answer_root_node = create_tree(data["gold_dependencies"], data["subtasks"])
            predict_answer_root_node = create_tree(data["predict_dependencies"], data["subtasks"])
            if gold_answer_root_node == False:
                continue
            if predict_answer_root_node != False: #### 确保生成了树
                raw_metric = compute_metric_one_data(gold_answer_root_node, predict_answer_root_node)
                pass_or_not = raw_metric["pass_or_not"]
                predict_depth = raw_metric["predict_depth"]
                predict_width = raw_metric["predict_width"]
                gold_depth = raw_metric["gold_depth"]
                gold_width = raw_metric["gold_width"]
                if pass_or_not:
            
                    pass_data_number += 1
                    predict_all_depth += predict_depth
                    predict_all_width += predict_width
                    all_length += len(data["subtasks"])
                    gold_all_depth += gold_depth
                    gold_all_width += gold_width
                    ############
                    all_predict_all_depth += predict_depth
                    all_predict_all_width += predict_width
                    all_all_length += len(data["subtasks"])
                    all_gold_all_depth += gold_depth
                    all_gold_all_width += gold_width

                else:
                    all_predict_all_depth += predict_depth
                    all_predict_all_width += predict_width
                    all_all_length += len(data["subtasks"])
                    all_gold_all_depth += gold_depth
                    all_gold_all_width += gold_width

            else:
                unable_create_tree += 1
                pass_or_not = False
        else:
            hallucination_number += 1

    if len(all_data) != 0:
        pass_rate = pass_data_number/(len(all_data) + wrong_number)
        hallucination_rate = hallucination_number/(len(all_data) + wrong_number)
        uncreate_tree_rate = unable_create_tree/(len(all_data) + wrong_number)
    else:
        pass_rate = 0
        hallucination_rate = 0
        uncreate_tree_rate = 0
    if all_length != 0:
        predict_compression_rate = predict_all_depth/all_length
        gold_compression_rate = gold_all_depth/all_length
    else:
        predict_compression_rate = 0
        gold_compression_rate = 0

    if all_all_length != 0:
        all_predict_compression_rate = all_predict_all_depth/all_all_length
        all_gold_compression_rate = all_gold_all_depth/all_all_length
    else:
        all_predict_compression_rate = 0
        all_gold_compression_rate = 0
    

    return {
        "pass_rate": pass_rate,
        "predict_compression_rate": predict_compression_rate,
        "gold_compression_rate": gold_compression_rate,
        "hallucination_rate": hallucination_rate,
        "uncreate_tree_rate": uncreate_tree_rate,
        "all_predict_compression_rate": all_predict_compression_rate,
        "all_gold_compression_rate": all_gold_compression_rate,
    }

        
        




