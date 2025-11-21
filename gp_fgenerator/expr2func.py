
import os
from deap.gp import Primitive
from deap import creator
from .expr_inject import Node
from .expr_clean import cleaning1, cleaning2
from .expr_tree2func import tree2func
from .utils import write_file


#%%
def expr_id(pset):
    # primitive
    id_pset = {}
    for i, item in enumerate(pset.primitives[pset.ret]):
        id_pset[item.name] = i
    # terminal
    id_tset = {}
    for i, item in enumerate(pset.terminals[pset.ret]):
        id_tset[item.value] = i
    
    id_polish = {'rand_num': 1,
                 'x': 2,
                 'x_': 2,
                 'first_dv': 3,
                 'trans_dv': 4,
                 'rot_mat': 5,
                 'index_vec': 6,
                 'rand_mat': 7,
                 'add': 11,
                 'sub': 12,
                 'mul': 13,
                 'div': 14,
                 'neg': 21,
                 'reciprocal': 22,
                 'mul10': 23,
                 'square': 24,
                 'sqrt': 25,
                 'abs_': 26,
                 'roundoff': 27,
                 'sin': 28,
                 'cos': 29,
                 'ln': 30,
                 'exp': 31,
                 'sum_vec': 32,
                 'mean_vec': 33,
                 'cumsum_vec': 34,
                 'prod_vec': 35,
                 'amax_vec': 36}
    dict_id = {'pset': id_pset, 'tset': id_tset, 'polish': id_polish}
    return  dict_id
# END DEF

#%%
# cleaning unary and binary operators
def expr2func(expr, pset, x):
    dict_id = expr_id(pset)
    tree, dict_ephemeral = expr2tree(creator.Individual(expr), dict_id)
    cleaning1(tree)
    cleaning2(tree)
    cleaning1(tree)
    cleaning2(tree)
    func_, strf_ = tree2func(tree, dict_ephemeral, x)
    return func_, strf_
# END DEF

#%%
# translate expression to tree
def expr2tree(expr, dict_id):
    edges = list()
    labels = dict()
    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, Primitive) else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()
    dict_node = {}
    dict_EphemeralConstant = {}
    for i in labels.keys():
        label_temp = labels[i]
        if (isinstance(label_temp, str)):
            dict_node[i] = Node(dict_id['polish'][label_temp])
        else:
            type_ = str(expr[i]).split(" ")[0].replace('<deap.gp.', '')
            dict_node[i] = Node(dict_id['polish'][type_])
            if (type_ not in dict_EphemeralConstant.keys()):
                dict_EphemeralConstant[type_] = [labels[i]]
            else:
                dict_EphemeralConstant[type_].append(labels[i])
    for edge in reversed(edges):
        a = edge[0]
        b = edge[1]
        if not (dict_node[a].left):
            dict_node[a].left = dict_node[b]
        else:
            dict_node[a].right = dict_node[b]
    return dict_node[0], dict_EphemeralConstant
# END DEF