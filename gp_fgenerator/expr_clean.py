



#%%
# clean unary operators
def cleaning1(tree):
    if (tree.get_type() == 0):
        scalar = tree.get_isscalar()
        
    elif (tree.get_type() == 1):
        scalar = cleaning1(tree.left)
        
        if (scalar and tree.get_isvector()):
            # If the node is a vector-oriented operator and the child is a scalar, replace the node with its child
            tree.value = tree.left.value
            tree.right = tree.left.right
            tree.left  = tree.left.left
        else:
            if (tree.value==26 and (tree.left.value in [25,26,30])):
                # If the node is abs and the child is abs, log, or sqrt, replace the node with its child
                tree.value = tree.left.value
                tree.left  = tree.left.left
            
            elif ((tree.value in [25,26,30]) and tree.left.value==26):
                # If the node is abs, log, or sqrt and the child is abs, replace the child with its child
                tree.left = tree.left.left
            
            elif (tree.value==21 and tree.left.value==21 or tree.value==22 and tree.left.value==22):
                # If both the node and child are negative or reciprocal, replace the node with its child's child
                tree.value = tree.left.left.value
                tree.right = tree.left.left.right
                tree.left  = tree.left.left.left
            
            elif (tree.value==24 and tree.left.value==25 or tree.value==25 and tree.left.value==24 or 
                  tree.value==30 and tree.left.value==31 or tree.value==31 and tree.left.value==30):
                # If both the node and child are square and sqrt or log and exp, replace the node with its child's child
                tree.value = tree.left.left.value
                tree.right = tree.left.left.right
                tree.left  = tree.left.left.left
            
            elif (tree.value==21 and tree.left.value==12):
                # If the node is negative and the child is subtraction, replace the node with its child then exchange its children
                tree.value = 12
                tree.right = tree.left.left
                tree.left  = tree.left.right
            
            elif (tree.value==22 and tree.left.value==14):
                # If the node is reciprocal and the child is division, replace the node with its child then exchange its children
                tree.value = 14
                tree.right = tree.left.left
                tree.left  = tree.left.right
        scalar = scalar or tree.get_isvector() and tree.value!=34
        
    elif (tree.get_type() == 2):
        scalar1 = cleaning1(tree.left)
        scalar2 = cleaning1(tree.right)
        scalar = scalar1 and scalar2
    return scalar
# END DEF

#%%
# clean binary operators
def cleaning2(tree):
    if (tree.get_type() == 0):
        cons = tree.get_iscons()
        
    elif (tree.get_type() == 1):
        cons = cleaning2(tree.left)
        
    elif (tree.get_type() == 2):
        cons1 = cleaning2(tree.left)
        cons2 = cleaning2(tree.right)
        cons = cons1 and cons2
        
        if (cons):
            # If both the children are constants, change the node to a constant
            tree.value = 1
            tree.left  = []
            tree.right = []
        
        elif (tree.right.value == 21):
            # If the node is addition and the right child is negative, change the node to subtraction and replace the right child with its child
            if (tree.value == 11):
                tree.value = 12
                tree.right = tree.right.left
            # If the node is subtraction and the right child is negative, change the node to addition and replace the right child with its child
            elif (tree.value == 12):
                tree.value = 11
                tree.right = tree.right.left
        
        elif (tree.right.value == 22):
            # If the node is multiplication and the right child is reciprocal, change the node to division and replace the right child with its child
            if (tree.value == 13):
                tree.value = 14
                tree.right = tree.right.left
            # If the node is division and the right child is reciprocal, change the node to multiplication and replace the right child with its child
            elif tree.value == 14:
                tree.value = 13
                tree.right = tree.right.left
        
        elif (tree.left.value==21 and tree.value==11):
            # If the node is addition and the left child is negative, change the node to subtraction, replace the left child
            # with its child, then exchange the node's children
            tree.value = 12
            temp = tree.right
            tree.right = tree.left.left
            tree.left = temp
        
        elif (tree.left.value==22 and tree.value==13):
            # If the node is multiplication and the left child is reciprocal, change the node to division, replace the left
            # child with its child, then exchange the node's children
            tree.value = 14
            temp = tree.right
            tree.right = tree.left.left
            tree.left = temp
        
        elif (tree.left.get_isbinary() and cons2):
            if (all_ismember([tree.value,tree.left.value],[11,12]) or all_ismember([tree.value,tree.left.value],[13,14])):
                if (tree.left.left.get_iscons() and tree.left.right.get_iscons()):
                    # If the left child is a binary operator, at least one of the left child's children is a constant,
                    # and the right child is a constant, replace the node with its left child
                    tree.value = tree.left.value
                    tree.right = tree.left.right
                    tree.left = tree.left.left
        
        elif (cons1 and tree.right.get_isbinary()):
            if (all_ismember([tree.value,tree.right.value],[11,12]) or all_ismember([tree.value,tree.right.value],[13,14])):
                if (tree.right.right.get_iscons()):
                    # If the right child is a binary operator, the right child's right child is a constant, and the
                    # left child is a constant, replace the right child with its left child
                    tree.right = tree.right.left
                elif (tree.right.left.get_iscons()):
                    # If the right child is a binary operator, the right child's left child is a constant, and the
                    # left child is a constant, replace the right child with its right child then change the node's operator
                    if (tree.value == tree.right.value):
                        if (tree.value <= 12):
                            tree.value = 11
                        else:
                            tree.value = 13
                        # END IF
                    else:
                        if (tree.value <= 12):
                            tree.value = 12
                        else:
                            tree.value = 14
                    tree.right = tree.right.right
        
        elif (tree.left.get_isbinary() and tree.right.get_isbinary()):
            if (all_ismember([tree.left.value,tree.value,tree.right.value],[11,12]) or all_ismember([tree.left.value,tree.value,tree.right.value],[13,14])):
                if ((tree.left.left.get_iscons() or tree.left.right.get_iscons()) and tree.right.right.get_iscons()):
                    # If both the left and right children are binary operators, at least one of the left child's children is a constant, 
                    # and the right child's right child is a constant, replace the right child with its left child
                    tree.right = tree.right.left
                elif ((tree.left.left.get_iscons() or tree.left.right.get_iscons()) and tree.right.left.get_iscons()):
                    # If both the left and right children are binary operators, at least one of the left child's children is a constant, 
                    # and the right child's left child is a constant, replace the right child with its right child then change the node's operator
                    if (tree.value == tree.right.value):
                        if (tree.value <= 12):
                            tree.value = 11
                        else:
                            tree.value = 13
                    else:
                        if (tree.value <= 12):
                            tree.value = 12
                        else:
                            tree.value = 14
                    tree.right = tree.right.right
    return cons
# END DEF

#%%
def all_ismember(list_item, list_check):
    boolean = True
    for item in list_item:
        if (item not in list_check):
            boolean = False
            break
    return boolean
# END DEF