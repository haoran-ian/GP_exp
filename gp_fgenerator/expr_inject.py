
import numpy as np


# %%
def injection(tree):
    r = np.random.random()
    if r < 0.05:
        # Noisy landscape
        tree = Node(13, tree, Node(7))
    elif r < 0.1:
        # Flat landscape
        tree = Node(27, tree)
    elif r < 0.2:
        # Multimodal landscape
        tree = Node(11, tree, Node(28, tree))
    elif r < 0.25:
        # Highly multimodal landscape
        tree = Node(11, tree, Node(23, Node(28, tree)))
    elif r < 0.3:
        # Linkages between all the variables and the first variable
        tree = injection2(tree, 1)
    elif r < 0.35:
        # Linkages between each two contiguous variables
        tree = injection2(tree, 2)
    elif r < 0.4:
        # Complex linkages between all the variables
        tree = injection2(tree, 3)
    elif r < 0.45:
        # Different optimal values to all the variables
        tree = injection2(tree, 4)
    return tree
# END DEF

# %%


def injection2(tree, tree_type):
    if (isinstance(tree, Node)):
        if tree.value == 2:
            if tree_type == 1:
                tree = Node(12, Node(2), Node(3))
            elif tree_type == 2:
                tree = Node(12, Node(2), Node(4))
            elif tree_type == 3:
                tree = Node(5)
            elif tree_type == 4:
                tree = Node(13, Node(6), Node(2))
        else:
            tree.left = injection2(tree.left, tree_type)
            tree.right = injection2(tree.right, tree_type)
    return tree
# END DEF

# %%


class Node(object):
    def __init__(self, *args):
        self.value = args[0]
        self.left = []
        self.right = []

        if (len(args) > 1):
            self.left = args[1]
            if (len(args) > 2):
                self.right = args[2]
    # %%

    def get_type(self):
        # Type of node: 0. operand 1. unary operator 2. binary operator
        if not (self.left):
            return 0
        elif not (self.right):
            return 1
        else:
            return 2
    # %%

    def get_iscons(self):
        # Whether the node is a constant
        if (self.value <= 1):
            return True
        else:
            return False
    # %%

    def get_isscalar(self):
        # Whether the node is a scalar
        if ((self.value in [1, 3, 7]) or (self.value <= 1)):
            return True
        else:
            return False
    # %%

    def get_isbinary(self):
        # Whether the node is a binary operator
        if (self.value in [11, 12, 13, 14]):
            return True
        else:
            return False
    # %%

    def get_isvector(self):
        # Whether the node is a vector-oriented operator
        if (self.value in [32, 33, 34, 35, 36]):
            return True
        else:
            return False
# END CLASS
