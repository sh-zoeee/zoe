import zss 
from pqgrams.tree import Node
from collections import deque
from graphviz import Graph


def tree_visualize(root, FILENAME="tmp_tree"):
    """
    木構造の幅優先探索を利用して木を描画する
    """
    queue = deque([root])
    graph = Graph(filename=FILENAME, format="pdf")
    graph.attr("node", shape="circle")
    node_id = 1
    graph.node(str(node_id), root.label)
    node_id += 1
    parent_id = 0
    while queue:
        node = queue.popleft()
        #print(node.label)  # ノードのラベルを出力
        parent_id += 1

        for child in node.children:
            queue.append(child)
            graph.node(str(node_id), child.label)
            graph.edge(str(parent_id), str(node_id))
            node_id += 1
    graph.render()