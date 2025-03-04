import pyconll
import zss
from pqgrams.tree import Node
from pqgrams.PQGram import Profile
from xml.etree import ElementTree as ET

def conllTree_to_zssNode_upos(tree: pyconll.tree)->zss.Node:
    node = zss.Node(tree.data.upos)
    for child in tree:
        node.addkid(conllTree_to_zssNode_upos(child))
    return node


def conllTree_to_pqTree_depth(conll_tree: pyconll.tree, depth=0) -> Node:
    root_pq = Node(str(depth))
    for child in conll_tree:
        root_pq.addkid(conllTree_to_pqTree_depth(child,depth+1))
    return root_pq


def conllTree_to_pqTree_unlabeled(conll_tree: pyconll.tree) -> Node:
    root_pq = Node("_")
    for child in conll_tree:
        root_pq.addkid(conllTree_to_pqTree_unlabeled(child))
    return root_pq

def conllTree_to_pqTree_upos(conll_tree: pyconll.tree) -> Node:
    root_pq = Node(conll_tree.data.upos)
    for child in conll_tree:
        root_pq.addkid(conllTree_to_pqTree_upos(child))
    return root_pq


def conllTree_to_zssNode_unlabel(tree: pyconll.tree)->zss.Node:
    node = zss.Node("_")
    for child in tree:
        node.addkid(conllTree_to_zssNode_unlabel(child))
    return node


def pqTree_to_zssTree(tree1: Node) -> zss.Node:
    node = zss.Node(tree1.label)
    for child in tree1.children:
        node.addkid(pqTree_to_zssTree(child))
    return node


def xml_to_pqTree(root):
    node = Node(root.tag)
    for child in root:
        node.addkid(xml_to_pqTree(child))
    return node


def string_to_pqtree(string:str):
    """
    stringsデータセットに対して用いる
    """
    root = Node(string[0])
    current = root
    for c in string[1:]:
        current.addkid(Node(c))
        current = current.children[0]
    return root


def create_binary_tree(height, label="notset"):
    
    root = Node(label if label!="notset" else str(height))

    if height > 1:
        left_child = create_binary_tree(height-1, label)
        right_child = create_binary_tree(height-1, label)
        root.addkid(left_child)
        root.addkid(right_child)
    
    return root 


def string_to_pqtree(string: str):
    root = Node(str(0))

            
