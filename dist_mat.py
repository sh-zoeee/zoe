
from pqgrams.tree import Node
from pqgrams.PQGram import Profile
from collections import deque
import zss
from graphviz import Graph
import numpy as np
import torch
import time

def softplus(x):
   return np.log(1 + np.exp(x))

def zss_Node(root: Node):
  node = zss.Node(root.label)
  for child in root.children:
    node.addkid(zss_Node(child))
  return node


def visualize(root: zss.Node, depth=0):
  print("-"*depth, root.label, sep="")
  for child in root.children:
    visualize(child, depth+1)


def visualize_graphviz(root, FILENAME="tmp_tree"):
    """
    木構造の幅優先探索を行う関数
    """
    if not root:
        return
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

def pqgram_distance_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor, device="cpu"):
    dim = tensor1.size()[0]
    tensor_min = torch.minimum(tensor1, tensor2)
    tensor_diff = tensor1 + tensor2 - 2*tensor_min

    dist = torch.mm(torch.ones(dim, dtype=torch.float32).to(device=device).transpose(),tensor_diff)
    return dist.to("cpu").detach().numpy()


    


device = "cuda:0"

# 木構造1
"""
  a
 /|\
a b c
|\
e b 
"""
root1 = Node("a")
root1.addkid(Node("a"))
root1.addkid(Node("b"))
root1.addkid(Node("c"))
root1.children[0].addkid(Node("e"))
root1.children[0].addkid(Node("b"))


# 木構造2
"""
  a
 /|\
a b d
|\
e b
"""
root2 = Node("a")
root2.addkid(Node("a"))
root2.addkid(Node("b"))
root2.addkid(Node("d"))
root2.children[0].addkid(Node("e"))
root2.children[0].addkid(Node("b"))


# PQ-Gram プロファイルの作成
p1 = Profile(root1, p=2, q=3)
p2 = Profile(root2, p=2, q=3)


J = [pqgram for pqgram in p1]
for pqgram in p2:
   if pqgram not in J:
      J.append(pqgram)


dimension = len(J)

v1 = np.zeros(dimension, dtype=int)
v2 = np.zeros(dimension, dtype=int)

for pqgram in p1:
   if pqgram in J:
      for i, subtree in enumerate(J):
         if pqgram == subtree:
            v1[i] += 1

for pqgram in p2:
   if pqgram in J:
      for i, subtree in enumerate(J):
         if pqgram == subtree:
            v2[i] += 1

tensor1 = torch.from_numpy(v1.astype(np.float32)).to(device)
tensor2 = torch.from_numpy(v2.astype(np.float32)).to(device)

start = time.time()
distance = pqgram_distance_tensor(tensor1, tensor2, device=device)
end = time.time()
print(distance, end-start, "ns")

exit()
tensor_min = torch.minimum(tensor1, tensor2)

min12 = np.minimum(v1, v2)

tensor_diff = tensor1 + tensor2 - 2*tensor_min


print(((torch.ones(dimension, dtype=torch.float32).to(device=device)@tensor_diff)).to("cpu").detach().numpy())

diff = v1 + v2 - 2*min12


print("distance from vector",end="\t")
print(np.ones(dimension, dtype=int) @ diff) # 内積計算



