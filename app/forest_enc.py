"""随机森林的 z3 编码（试配求解与容差核查共用）。

按原分类器的真实阈值语义编码：sklearn 树内部以 float32 特征与 float64
阈值比较，这里把每个分裂条件精确翻译成整数克数比较（g <= K），
K 由 model.cutoff_left 预计算。每棵树被约束为恰好落在一个叶子上，
叶子概率以精确有理数参与 z3 算术。
"""
from __future__ import annotations

import z3

from .model import ForestSpec, Leaf, cutoff_left


def encode_forest_leaves(container, forest: ForestSpec, g) -> list[list[tuple]]:
    """把整座森林编码进 container（Solver/Optimize），返回每棵树的叶子选择表。

    返回值：trees_select，其中 trees_select[t] = [(Bool 变量, 叶子 prob1 Fraction), ...]，
    并附带两个约束：
      1. 每个叶子布尔变量 == 根到该叶子全部路径条件的合取；
      2. 每棵树恰好有一个叶子生效（PbEq == 1）。
    """
    trees_select = []
    for ti, tree in enumerate(forest.trees):
        # 父指针表：子节点 id -> (父节点 id, 是否走左分支)
        parent = {}
        for pid, pn in enumerate(tree.nodes):
            if not isinstance(pn, Leaf):
                parent[pn.left] = (pid, True)
                parent[pn.right] = (pid, False)
        leaf_selects = []
        for nid, node in enumerate(tree.nodes):
            if not isinstance(node, Leaf):
                continue
            # 回溯根到该叶子的路径条件
            conds = []
            cur = nid
            while cur in parent:
                pid, went_left = parent[cur]
                pn = tree.nodes[pid]
                k = cutoff_left(pn.threshold)
                conds.append(g[pn.feature] <= k if went_left else g[pn.feature] >= k + 1)
                cur = pid
            b = z3.Bool(f"t{ti}_leaf{nid}")
            container.add(b == z3.And(*conds))
            leaf_selects.append((b, node.prob1))
        container.add(z3.PbEq([(b, 1) for b, _ in leaf_selects], 1))  # 恰好一个叶子生效
        trees_select.append(leaf_selects)
    return trees_select


def tree_prob_expr(leaf_selects) -> "z3.ArithRef":
    """一棵树的合格概率表达式（对当前选中叶子的 prob1 求和）。"""
    return z3.Sum(
        [z3.If(b, z3.Q(p.numerator, p.denominator), 0) for b, p in leaf_selects]
    )


def forest_prob_sum_expr(trees_select) -> "z3.ArithRef":
    """全森林合格概率之和（5 棵树）的精确有理表达式；除以 n_trees 即平均概率。"""
    return z3.Sum([tree_prob_expr(sel) for sel in trees_select])
