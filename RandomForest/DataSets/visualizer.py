import argparse
import os
import random
os.environ["PATH"] += os.pathsep + './graphviz/bin/'
from graphviz import Digraph, nohtml


def get_spaced_colors(n):
    return ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(n)]

class TreeNode(object):
    def __init__(self, weight, featureIndex, threshold, splitMetricValue, classIndex,
                 objectCount, leftChildIndex, rightChildIndex):
        self.weight = weight
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.splitMetricValue = splitMetricValue
        self.classIndex = classIndex
        self.objectCount = objectCount
        self.leftChildIndex = leftChildIndex
        self.rightChildIndex = rightChildIndex
        self.color = None

    @staticmethod
    def from_list(list):
        return TreeNode(
            weight=list[0],
            featureIndex=int(list[1]),
            threshold=list[2],
            splitMetricValue=list[3],
            classIndex=int(list[4]),
            objectCount=int(list[5]),
            leftChildIndex=int(list[6]),
            rightChildIndex=int(list[7])
        )

    def class_to_str(self):
        return 'Class \\={}\\n'.format(str(class_map[self.classIndex])) if (
                self.leftChildIndex == -1 and self.rightChildIndex == -1) else ''

    def feat_idx_to_str(self):
        return 'feature[{}] \\>\\= {}\\n'.format(str(self.featureIndex), str(self.threshold)) if (
                self.leftChildIndex != -1 or self.rightChildIndex != -1) else ''

    def gini_to_str(self):
        return 'Gini \\= {}\\n'.format(str(self.splitMetricValue))

    def obj_cnt_to_str(self):
        return 'ObjectCount \\= {}\\n'.format(str(self.objectCount))

    def to_string(self):
        return self.class_to_str() \
            + self.feat_idx_to_str() \
            + self.gini_to_str() \
            + self.obj_cnt_to_str()

class Tree(object):
    def __init__(self, list):
        self.size = int(list[0])
        self.tree = []
        for i in range(self.size):
            self.tree.append(TreeNode.from_list(list[i*8 + 1:]))

    @staticmethod
    def from_list(list):
        tree = Tree(list)
        return tree, tree.size*8 + 1

    def get_color(self, cur_idx):
        cur_node = self.tree[cur_idx]
        if cur_node.color:
            return cur_node.color
        if cur_node.leftChildIndex == -1 and cur_node.rightChildIndex == -1:
            return colors[cur_node.classIndex]
        left_obj_cnt = None
        if cur_node.leftChildIndex != -1:
            color = self.get_color(cur_node.leftChildIndex)
            left_obj_cnt = self.tree[cur_node.leftChildIndex].objectCount
        if cur_node.rightChildIndex != -1:
            if left_obj_cnt is None or self.tree[cur_node.rightChildIndex].objectCount > left_obj_cnt:
                color = self.get_color(cur_node.rightChildIndex)
        return color

    def recursive_creation(self, g, cur_idx, limit, max_limit):
        cur_node = self.tree[cur_idx]
        g.node('node'+str(cur_idx), cur_node.to_string(),style='filled',fillcolor=self.get_color(cur_idx))
        if limit > max_limit:
            return
        if cur_node.leftChildIndex>0:
            self.recursive_creation(g,cur_node.leftChildIndex, limit+1, max_limit)
            g.edge('node'+str(cur_idx), 'node'+str(cur_node.leftChildIndex))
        if cur_node.rightChildIndex>0:
            self.recursive_creation(g,cur_node.rightChildIndex, limit+1, max_limit)
            g.edge('node'+str(cur_idx), 'node'+str(cur_node.rightChildIndex))


    def print(self, max_limit):
        g = Digraph('g', filename='btree.gv', node_attr={'shape': 'record', 'height': '.1'})
        self.recursive_creation(g,0, 0, max_limit)
        g.render('round-table.gv', view=True)


def read_class_translation(list):
    class_map = dict()
    size = int(list[0])
    for i in range(size):
        class_map[int(list[1 + i*2])] = int(list[2 + i*2])
    return list[size*2+1:], class_map


def read_model_as_list(f):
    return [float(s) for s in f.readline().split(' ')[:-1]]

p=argparse.ArgumentParser(
    description='tool for visualising decision trees',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
p.add_argument('--model_path', required=True)
p.add_argument('--limit', default=100, type=int)

args = p.parse_args()

with open(args.model_path,'r') as f:
    model_list, class_map = read_class_translation(read_model_as_list(f))
    colors = get_spaced_colors(len(class_map))
    tree_count = int(model_list[0])
    trees = []
    pos = 1
    for i in range(tree_count):
        tree, add_pos = Tree.from_list(model_list[pos:])
        pos += add_pos
        trees.append(tree)
    trees[0].print(args.limit)
    