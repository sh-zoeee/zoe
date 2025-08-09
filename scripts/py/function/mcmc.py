import random, copy
import zss


def generate_random_tree(n, random_state=0) -> dict[int, list[int]]:
    """
        辞書型のtreeで木構造を実現
        keyの子ノードをvalue（リスト）とする
    """
    random.seed(random_state) # 固定したいなら

    tree = {0: []}
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(i)
        tree[i] = []
    return tree


def get_descendants(tree, node) -> set:
    """
        treeにおけるnodeのすべての子孫を再帰的に取得
    """
    desc = set()
    for child in tree[node]:
        desc.add(child)
        desc |= get_descendants(tree, child) # 和集合を取得 
    return desc


def propose_subtree_move(tree) -> dict[int, list[int]]:
    """
        サブツリーを切り取って他の位置に接続する操作
    """
    tree = copy.deepcopy(tree)
    nodes = list(tree.keys())
    nodes.remove(0)
    v = random.choice(nodes)

    # 現在の親を探す
    parent_v = None
    for parent, children in tree.items():
        if v in children:
            parent_v = parent
            break

    # 切り離す
    tree[parent_v].remove(v)

    # 移動先の候補（子孫と自分自身を除く）
    descendants = get_descendants(tree, v)
    invalid = descendants | {v}
    candidates = [u for u in tree if u not in invalid]

    if not candidates:
        # 復元して元に戻す
        tree[parent_v].append(v)
        return tree

    new_parent = random.choice(candidates)
    tree[new_parent].append(v)

    return tree


def dict_to_zss(tree: dict[int, list[int]], labels: dict[int, str] = None) -> zss.Node:
    """
    dict形式の木構造（親→子）を zss.Node に変換する。
    
    Parameters:
        tree: 木構造（親 → 子リスト）
        labels: ノードID → ラベル（任意）。なければIDをそのまま文字列化

    Returns:
        zss.Node形式の根ノード
    """
    if labels is None:
        labels = {i: "_" for i in tree.keys()}

    def build_subtree(node_id: int) -> zss.Node:
        label = labels[node_id] if labels and node_id in labels else str(node_id)
        children = [build_subtree(child_id) for child_id in tree.get(node_id, [])]
        return zss.Node(label, children)

    return build_subtree(0)  # 0 を根として仮定


def zss_to_dict(zss_node: zss.Node) -> dict[int, list[int]]:
    """
    zss形式の木構造を dict形式（親 → 子リスト）に変換する。
    
    Parameters:
        zss_node: zss.Node形式の根ノード

    Returns:
        dict形式の木構造
    """
    tree = {}
    node_counter = [0]  # ノードIDを管理するためのリスト（ミュータブル）

    def build_tree(node: zss.Node, parent_id=None):
        node_id = node_counter[0]
        node_counter[0] += 1

        # 子ノードのラベルを取得
        children_ids = []
        for child in node.children:
            try:
                child_id = node_counter[0]
                children_ids.append(child_id)
                build_tree(child, parent_id=node_id)
            except Exception as e:
                print(f"Error processing child node: {e}")

        # 現在のノードをツリーに追加
        tree[node_id] = children_ids

    build_tree(zss_node)
    return tree


# サイズnの完全二分木を生成
def generate_full_binary_tree(n):
    """
    サイズ n の完全二分木を生成する関数
    """
    tree = {i: [] for i in range(n)}
    for i in range(n):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < n:
            tree[i].append(left_child)
        if right_child < n:
            tree[i].append(right_child)
    return tree

# スター木を生成
def generate_star_tree(n):
    """
    サイズ n のスター木を生成する関数
    """
    tree = {0: list(range(1, n))}
    for i in range(1, n):
        tree[i] = []
    return tree


