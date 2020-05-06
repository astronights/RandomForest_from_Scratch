from abc import ABC

import numpy as np
import pandas as pd

from rf_node import Node


class Tree:
    def __init__(self, node, max_depth, min_samples_split, min_samples_leaf, num_splits):
        self.root = node
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_splits = num_splits

    def build_tree(self):
        while(self.depth(self.root) < self.max_depth):
            node_to_expand = self.get_expandable_node(self.root)
            # node_to_expand.print_node()
            if(node_to_expand is None or node_to_expand.leaf == False):
                break
            elif(node_to_expand.samples <= self.min_samples_split):
                node_to_expand.tried = True
            else:
                node_to_expand.split_node(self.min_samples_leaf, self.num_splits)
            # self.print_tree()
            # print(self.depth(self.root))

    def print_tree(self):
        print("~~~~~")
        stack = [(0,self.root)]
        while(stack):
            i, node = stack.pop(0)
            if(node):
                # print("Level "+str(i)+" : ", end=' ')
                print((' '*2*i) + '|' + ('-'*i), end=' ')
                node.print_node()
                stack = [(i+1,node.left), (i+1,node.right)] + stack

    def get_expandable_node(self, node):
        max_imp = 0
        max_node = node
        if(node is not None):
            # node.print_node()
            left = self.get_expandable_node(node.left)
            right = self.get_expandable_node(node.right)
            if(left and left.leaf and left.tried == False and left.impurity > max_imp):
                max_imp = left.impurity
                max_node = left
            if(right and right.leaf and right.tried == False and right.impurity > max_imp):
                max_imp = right.impurity
                max_node = right
        return(max_node)

    def predict(self, x):
        node = self.root
        while(node.leaf == False):
            if(node.is_cat):
                if(x[node.criteria] == node.value):
                    node = node.left
                else:
                    node = node.right
            else:
                if(x[node.criteria] <= node.value):
                    node = node.left
                else:
                    node = node.right
        return(node.max_val)

    def depth(self, tree):
        if tree.left == None and tree.right == None:
            return(1)

        return(1+max(self.depth(tree.left), self.depth(tree.right)))
