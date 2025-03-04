from pqgrams.tree import Node
from pqgrams.PQGram import Profile
import zss

from .TreeKernel import tree, tree_kernels

import random


class MyTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def generate_strings(n: int):
    return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(n))


def string_to_pqgram_tree(string, p, q):
    root = Node(string[0])
    current = root
    for c in string[1:]:
        current.addkid(Node(c))
        current = current.children[0]
    return Profile(root, p=p, q=q)

def string_to_zss_tree(string):
    root = zss.Node(string[0])
    current = root
    for c in string[1:]:
        current.addkid(zss.Node(c))
        current = current.children[0]
    return root


def generate_binaries(n: int, lower: int, upper: int):

    binaries = []
    for _ in range(n):
        height = random.randint(lower, upper)
        string = generate_binarytree(height)
        binaries.append(string)
    
    return binaries



def generate_binarytree(height: int):
    text = ''
    for depth in range(height):
        text += str(depth)*(2**depth)
    print(text)
    return text


def tree_to_prolog_unlabel(node: MyTree):
    if node is None:
        return ""
    
    left = tree_to_prolog_unlabel(node.left)
    right = tree_to_prolog_unlabel(node.right)

    if left or right:
        return f"_({left},{right})".strip(",")
    else:
        return "_"
    

def tree_to_prolog_upos(node: MyTree):
    if node is None:
        return ""
    
    left = tree_to_prolog_upos(node.left)
    right = tree_to_prolog_upos(node.right)

    if left or right:
        return f"{node.value}({left},{right})".strip(",")
    else:
        return f"{node.value}"

    

def create_complete_tree_TK_upos(n: int, labels: list=[]) -> tree.Tree :
    
    if len(labels)==0:
        for _ in range(n):
            labels.append("_")

    if n <= 0 or len(labels)!=n:
        return None

    
    nodes = [MyTree(labels[i]) for i in range(n)]

    for i in range(n):
        left_index = 2*i + 1
        right_index = 2*i + 2

        if left_index < n:
            nodes[i].left = nodes[left_index]
        if right_index < n:
            nodes[i].right = nodes[right_index]
    
    prolog_style = tree_to_prolog_upos(nodes[0])
    root_tk = tree.TreeNode.fromPrologString(prolog_style)
    return tree.Tree(root_tk)
        


def create_complete_tree_TK_unlabel(n: int) -> tree.Tree :
    

    if n <= 0:
        return None

    
    nodes = [MyTree(value="_") for i in range(n)]

    for i in range(n):
        left_index = 2*i + 1
        right_index = 2*i + 2

        if left_index < n:
            nodes[i].left = nodes[left_index]
        if right_index < n:
            nodes[i].right = nodes[right_index]
    
    prolog_style = tree_to_prolog_unlabel(nodes[0])
    root_tk = tree.TreeNode.fromPrologString(prolog_style)
    return tree.Tree(root_tk)


def create_linear_tree_TK_upos(n, direction="left", labels: list=[]) -> tree.Tree:

    if n <= 0 or direction not in ["left", "right"]:
        return None


    root = MyTree(labels[0])
    current = root

    for i in range(1, n):
        new_node = MyTree(value=labels[i])
        if direction == "left":
            current.left = new_node
        elif direction == "right": 
            current.right = new_node
        current = new_node
    
    prolog_style = tree_to_prolog_upos(root)
    root_tk = tree.TreeNode.fromPrologString(prolog_style)
    return tree.Tree(root_tk)



def create_linear_tree_TK_unlabel(n, direction="left") -> tree.Tree:

    if n <= 0 or direction not in ["left", "right"]:
        return None


    root = MyTree(value="_")
    current = root

    for i in range(1, n):
        new_node = MyTree(value="_")
        if direction == "left":
            current.left = new_node
        elif direction == "right": 
            current.right = new_node
        current = new_node
    
    prolog_style = tree_to_prolog_unlabel(root)
    root_tk = tree.TreeNode.fromPrologString(prolog_style)
    return tree.Tree(root_tk)
    

    
def inorder_traversal(tree, index, result):
    """
    完全二分木（配列として表現）を中間順巡回してソートする。
    
    Args:
        tree (list): 元の配列
        index (int): 現在のノードのインデックス
        result (list): 中間順巡回で得た要素を格納するリスト
    """
    # インデックスが配列の範囲外なら終了
    if index >= len(tree):
        return

    # 左部分木の探索
    inorder_traversal(tree, 2 * index + 1, result)
    
    # 現在のノードを追加
    result.append(tree[index])
    
    # 右部分木の探索
    inorder_traversal(tree, 2 * index + 2, result)

def sort_by_inorder_traversal(arr):
    """
    配列を完全二分木と見立て、中間順巡回でソートする。
    
    Args:
        arr (list): ソートしたい配列
    Returns:
        list: 中間順巡回によるソート結果
    """
    result = []
    inorder_traversal(arr, 0, result)
    return result