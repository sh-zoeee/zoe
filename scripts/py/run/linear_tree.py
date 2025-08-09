"""
    実装すること：
        大きさnの線形の木の距離を、pq-gram と 編集距離 で考えるとどのようになるか
        横軸：木の大きさn、縦軸：距離でグラフを描くとどのような曲線になるか
"""

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))

from function import create_strings, dist
from zss import simple_distance
import numpy as np

from tqdm import tqdm

def main():
    P, Q = 2, 2
    length_chars = 5
    count_strings = 10

    chars_pqgram = []
    chars_zss = []

    for _ in range(count_strings):
        s = create_strings.generate_strings(length_chars)
        pqgram = create_strings.string_to_pqgram_tree(string=s, p=P, q=Q)
        chars_pqgram.append(pqgram)
        zss_node = create_strings.string_to_zss_tree(string=s)
        chars_zss.append(zss_node)
    
    distances_pq = []
    distances_ted = []

    for i in range(count_strings):
        for j in range(i+1, count_strings):
            distances_pq.append(dist.pqgram_distance(chars_pqgram[i],chars_pqgram[j]))
            distances_ted.append(simple_distance(chars_zss[i], chars_zss[j]))
    
    print(np.mean(distances_pq))
    print(np.mean(distances_ted))



    return

if __name__=="__main__":
    main()