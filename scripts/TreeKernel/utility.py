import pyconll, pyconll.tree
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import time

from . import tree_kernels, tree

from tqdm import tqdm, trange



def to_prolog_unlabel(tree: pyconll.tree.tree.Tree) -> str:
    if tree._children:
        children_repr = ', '.join(to_prolog_unlabel(child) for child in tree._children)
        return f'_({children_repr})'
    else:
        return f'_'
    

def to_prolog_upos(tree: pyconll.tree.tree.Tree) -> str:
    if tree._children:
        children_repr = ', '.join(to_prolog_upos(child) for child in tree._children)
        return f'{tree.data.upos}({children_repr})'
    else:
        return f'{tree.data.upos}'
    
    


def calc_kernel_matrix(kernel: tree_kernels.Kernel, data1, data2=None):
    
    if data2==None:
        n = len(data1)
        matrix = np.zeros((n,n))
        start = time()
        for i in trange(n):
            if i%1000 == 0:
                print(f"Loop: {i}, \t{time()-start:.3f} sec")
            for j in range(i,n):
                matrix[i][j] = kernel.kernel(data1[i], data1[j])
        return matrix + matrix.T -np.diag(matrix.diagonal())

    n1, n2 = len(data1), len(data2)
    matrix = np.zeros((n1, n2))
    for i in trange(n1):
        for j in range(n2):
            matrix[i][j] = kernel.kernel(data1[i], data2[j])
    return matrix


def calc_kernel_matrix_parallel(kernel, data1, data2=None, use_threads=True):
    def compute_row(i, n, data, other_data=None):
        row = np.zeros(n)
        for j in range(n):
            if other_data is None:
                row[j] = kernel.kernel(data[i], data[j])
            else:
                row[j] = kernel.kernel(data[i], other_data[j])
        return i, row

    n1 = len(data1)
    n2 = len(data2) if data2 is not None else n1
    result = np.zeros((n1, n2))
    
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_cls() as executor:
        futures = []
        for i in range(n1):
            futures.append(executor.submit(compute_row, i, n2, data1, data2))

        for future in tqdm(futures):
            i, row = future.result()
            result[i, :] = row

    if data2 is None:
        return result + result.T - np.diag(result.diagonal())
    return result