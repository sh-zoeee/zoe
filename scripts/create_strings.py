from pqgrams.tree import Node
from pqgrams.PQGram import Profile
import zss

import random


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