def build_graph(node, graph, nodes, parent=None):
    """
    ツリー構造 (zss.Node) を無向グラフに変換します。
    graph は、各ノードの id をキー、その隣接ノード（id）のリストを値とする辞書です。
    nodes は、id をキー、対応する zss.Node を値とする辞書です。
    """
    node_id = id(node)
    if node_id not in graph:
        graph[node_id] = []
        nodes[node_id] = node
    if parent is not None:
        parent_id = id(parent)
        # 双方向リンクを追加
        graph[node_id].append(parent_id)
        graph[parent_id].append(node_id)
    for child in node.children:
        build_graph(child, graph, nodes, node)

def dfs_graph(graph, node_id, parent_id, depth):
    """
    無向グラフ上で DFS を行い、現在のノードから到達可能な最大の深さと
    そのノードの id を返します。
    
    引数:
      graph     : ノードidをキーとした隣接リストの辞書
      node_id   : 現在のノードの id
      parent_id : 一つ前に訪れたノードの id（これにより逆流を防止）
      depth     : 現在までの深さ
    戻り値:
      (max_depth, farthest_node_id)
    """
    max_depth = depth
    farthest_node_id = node_id
    for neighbor in graph[node_id]:
        if neighbor == parent_id:
            continue
        child_depth, candidate = dfs_graph(graph, neighbor, node_id, depth + 1)
        if child_depth > max_depth:
            max_depth = child_depth
            farthest_node_id = candidate
    return max_depth, farthest_node_id

def tree_diameter(root):
    """
    zss.Node で表現されたツリーの直径（最長経路の辺の数）を DFS 2 回で求める関数です。
    
    手順:
      1. ツリーを無向グラフに変換（id をキーとする）
      2. 任意のノード（ここでは root）から DFS を実施し、最も遠いノード node_u を取得する
      3. node_u を起点に DFS を実施し、そこから到達可能な最遠距離が直径となる
    """
    graph = {}
    nodes = {}
    build_graph(root, graph, nodes)
    
    # 1回目の DFS: root から最も遠いノードを求める
    _, farthest_node_id = dfs_graph(graph, id(root), None, 0)
    # print(f"First DFS: farthest node from 'root' is '{nodes[farthest_node_id].label}'")
    
    # 2回目の DFS: farthest_node_id からの最大距離を直径とする
    diameter, _ = dfs_graph(graph, farthest_node_id, None, 0)
    return diameter

def tree_height(node):
    """
    zss.Node で表現された木の高さ（根から最も深い葉までのエッジ数）を再帰的に求める関数です。
    
    もしノードが葉（子ノードが存在しない）であれば高さは 0 と定義します。
    ノードに子がある場合、その最大の高さに 1 を加えたものがそのノードの高さとなります。
    """
    # 葉ノードの場合は 0 を返す
    if not node.children:
        return 0
    # 各子ノードについて再帰的に木の高さを計算し、最大値に 1 を加える
    return max(tree_height(child) for child in node.children) + 1


def compute_branch_factors(root):
    """
    ツリー内の全ノードについて、分岐数（子ノードの数）を集計し、
    平均分岐数と最大分岐数を返す関数です。
    
    平均分岐数は「すべてのノードの子の数の合計 ÷ ノード数」
    最大分岐数は「各ノードの子ノード数の中で最大の値」となります。
    
    Args:
        root (zss.Node): ツリーの根ノード
        
    Returns:
        tuple: (平均分岐数, 最大分岐数)
    """
    total_children = 0  # 全ノードの子ノードの合計
    total_nodes = 0     # ツリー内のノード数
    max_children = 0    # 各ノードの子ノード数の最大値

    def dfs(node):
        nonlocal total_children, total_nodes, max_children
        # 現在のノードの子ノード数を取得
        children_count = len(node.children)
        total_children += children_count
        total_nodes += 1
        # 最大値の更新
        if children_count > max_children:
            max_children = children_count
        # 子ノードについて再帰呼び出し
        for child in node.children:
            dfs(child)

    dfs(root)
    # 平均分岐数はノード数が 0 でなければ計算する
    average = total_children / total_nodes if total_nodes > 0 else 0
    return average, max_children