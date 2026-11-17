"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import z3

from app.model import (
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性
    opt, g, num_changed, total_change = _build_optimize(forest, inp)
    opt.add(num_changed <= k - 1)
    check("z3 反证：更少种数无解", opt.check() == z3.unsat)
    opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
    opt2.add(num_changed2 == k, total_change2 <= m - 1)
    check("z3 反证：同种数更小改动量无解", opt2.check() == z3.unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 容差核查 ==")
    from app.tolerance import (
        TOL_FAIL,
        TOL_PASS,
        TOL_PENDING,
        ToleranceInput,
        tolerance_check,
        tolerance_ranges,
        count_feasible,
    )
    import app.tolerance as tmod

    def brute_tolerance_worst(inp_t: ToleranceInput):
        """对整个容差可行域做 sklearn 暴力枚举，返回 (最差配方, 最低概率, 点数)。

        锁定项区间在 tolerance_ranges 中已退化为单点，枚举天然尊重锁定。
        """
        ranges = tolerance_ranges(inp_t)
        pts = []
        (l0, h0), (l1, h1), (l2, h2), (l3, h3) = ranges
        for g0 in range(l0, h0 + 1):
            for g1 in range(l1, h1 + 1):
                for g2 in range(l2, h2 + 1):
                    g3 = TOTAL_GRAMS - g0 - g1 - g2
                    if l3 <= g3 <= h3:
                        pts.append((g0, g1, g2, g3))
        if not pts:
            return None, None, 0
        probs = []
        for s in range(0, len(pts), 2_000_000):
            arr = np.array(pts[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
            probs.append(clf.predict_proba(arr)[:, 1])
        p = np.concatenate(probs)
        min_p = float(p.min())
        # 1e-12 容差吸收分批 predict_proba 的求和顺序尾数差异（森林概率最小间隔约 1e-4）
        worst_set = [list(pts[k]) for k in range(len(pts)) if abs(float(p[k]) - min_p) <= 1e-12]
        return worst_set, min_p, len(pts)

    # 9.1 放行样例：z3 结论 pass，且与全域枚举最低点一致
    tp = samples["tolerance_pass"]
    inp_tp = ToleranceInput(tp["base"], tp["tolerance"], tp["locked"])
    res_tp = tolerance_check(forest, clf, inp_tp)
    check("放行样例核查为 pass", res_tp.status == TOL_PASS, res_tp.reason)
    check("放行样例最差概率 ≥ 0.7（原分类器复算）",
          res_tp.min_probability is not None and is_qualified(res_tp.min_probability),
          f"p={res_tp.min_probability:.6f}")
    check("最差配方在容差域内且总重 1000",
          sum(res_tp.worst) == TOTAL_GRAMS and all(
              tolerance_ranges(inp_tp)[i][0] <= res_tp.worst[i] <= tolerance_ranges(inp_tp)[i][1]
              for i in range(4)))
    check("锁定项在最差配方中保持基准用量",
          all(res_tp.worst[i] == tp["base"][i] for i in range(4) if tp["locked"][i]))
    bw, bp, bn = brute_tolerance_worst(inp_tp)
    check("放行样例行数统计与枚举点数一致（含锁定单点）", res_tp.region_size == bn,
          f"dp={res_tp.region_size}, brute={bn}")
    check("放行样例 z3 最低概率 == 全域枚举最低概率，且配方在并列最低点集合中",
          abs(res_tp.min_probability - bp) <= 1e-12 and res_tp.worst in bw,
          f"z3={res_tp.worst}@{res_tp.min_probability:.6f} brute_min={bp:.6f} ties={len(bw)}")

    # 9.2 不合格反例样例：fail，反例概率 < 0.7，且为枚举最低点
    tf = samples["tolerance_fail"]
    inp_tf = ToleranceInput(tf["base"], tf["tolerance"], tf["locked"])
    res_tf = tolerance_check(forest, clf, inp_tf)
    check("反例样例核查为 fail", res_tf.status == TOL_FAIL, res_tf.reason)
    check("反例概率 < 0.7（原分类器复算）",
          res_tf.min_probability is not None and not is_qualified(res_tf.min_probability),
          f"p={res_tf.min_probability:.6f}")
    bw2, bp2, bn2 = brute_tolerance_worst(inp_tf)
    check("反例样例 z3 最低点 == 全域枚举最低概率（并列集合成员）",
          abs(res_tf.min_probability - bp2) <= 1e-12 and res_tf.worst in bw2,
          f"z3={res_tf.worst}@{res_tf.min_probability:.6f} brute_min={bp2:.6f} ties={len(bw2)}")
    check("反例样例行数统计与枚举点数一致", res_tf.region_size == bn2)

    # 9.3 精确有理概率与 sklearn 复算一致
    rat = forest_proba_rational(forest, res_tf.worst)
    check("最差配方精确有理概率与返回值一致",
          res_tf.rational_min == [rat.numerator, rat.denominator])
    check("有理概率与 sklearn float64 复算一致", abs(float(rat) - res_tf.min_probability) <= 1e-12)

    # 9.4 小容差（域小）专项：手工可核对的枚举复核
    inp_small = ToleranceInput([500, 250, 150, 100], [5, 5, 5, 5], [False] * 4)
    res_small = tolerance_check(forest, clf, inp_small)
    bw3, bp3, bn3 = brute_tolerance_worst(inp_small)
    check("小容差域最低点与枚举一致（并列集合成员）",
          res_small.worst in bw3 and abs(res_small.min_probability - bp3) <= 1e-12
          and res_small.region_size == bn3,
          f"{res_small.worst}@{res_small.min_probability:.6f} vs min {bp3:.6f} n={bn3}")

    # 9.5 边界：全锁定 → 域内唯一配方（基准自身）
    inp_lock = ToleranceInput([700, 100, 100, 100], [99] * 4, [True] * 4)
    res_lock = tolerance_check(forest, clf, inp_lock)
    check("全锁定 → 域内唯一配方且结论可判定", res_lock.status == TOL_PASS
          and res_lock.worst == [700, 100, 100, 100] and res_lock.region_size == 1)
    inp_lock_bad = ToleranceInput([150, 130, 620, 100], [99] * 4, [True] * 4)
    res_lock_bad = tolerance_check(forest, clf, inp_lock_bad)
    check("全锁定在不合格配方上 → fail", res_lock_bad.status == TOL_FAIL
          and res_lock_bad.region_size == 1)

    # 9.6 非法输入
    bad_tol = [
        ToleranceInput([250, 250, 350, 149], [10] * 4, [False] * 4),   # 总重 999
        ToleranceInput([250, 250, 350, 150], [-1, 0, 0, 0], [False] * 4),  # 负偏差
        ToleranceInput([250, 250, 350, 150.0], [10] * 4, [False] * 4),  # 非整数克数
        ToleranceInput([250, 250, 350, 150], [10] * 4, [True, "x", False, False]),  # 非布尔锁定
    ]
    check("容差非法输入均判 invalid",
          all(tolerance_check(forest, clf, c).status == INVALID for c in bad_tol))
    # 不变式：基准总重 1000 + 对称偏差闭区间 + 非负 ⇒ 基准自身恒在容差域内，域恒非空。
    # 以 5000 组随机合法输入核对 DP 计数恒 ≥ 1，并与暴力枚举逐例一致。
    import random
    rng_t = random.Random(1)
    invariant_ok, fuzz_dp_ok = True, True
    for _ in range(5000):
        rem = TOTAL_GRAMS
        base_f = []
        for _i in range(3):
            v = rng_t.randint(0, rem)
            base_f.append(v)
            rem -= v
        base_f.append(rem)
        tol_f = [rng_t.randint(0, 50) for _ in range(4)]
        lk_f = [rng_t.random() < 0.5 for _ in range(4)]
        cf = ToleranceInput(base_f, tol_f, lk_f)
        rf = tolerance_ranges(cf)
        n_dp = count_feasible(rf)
        if n_dp < 1:
            invariant_ok = False
        # 小域才暴力，避免 5000 次三重循环过慢
        if all(h - lo <= 12 for lo, h in rf):
            _, _, n_br = brute_tolerance_worst(cf)
            if n_dp != n_br:
                fuzz_dp_ok = False
    check("合法输入容差域恒非空（5000 组随机）", invariant_ok)
    check("随机小域 DP 计数与暴力枚举一致", fuzz_dp_ok)

    # 9.7 待判路径：注入 unknown / 界不紧 / 异常，均不得给出 pass/fail
    orig_tb = tmod._build_tolerance_optimize

    class _FakeOpt:
        def set(self, *a, **k): pass
        def check(self): return z3.unknown
        def minimize(self, e): return "h"
        def lower(self, h): return z3.RealVal(1)
        def upper(self, h): return z3.RealVal(2)
        def model(self): return None

    def fake_unknown(f, i):
        return _FakeOpt(), z3.IntVector("gx", 4), "h", tolerance_ranges(i)
    tmod._build_tolerance_optimize = fake_unknown
    check("unknown → 待判（非 pass/fail）", tolerance_check(forest, clf, inp_tf).status == TOL_PENDING)

    class _FakeLoose(_FakeOpt):
        def check(self): return z3.sat
    tmod._build_tolerance_optimize = lambda f, i: (
        _FakeLoose(), z3.IntVector("gy", 4), "h", tolerance_ranges(i))
    check("sat 但界不紧 → 待判", tolerance_check(forest, clf, inp_tf).status == TOL_PENDING)

    def fake_boom(f, i):
        raise z3.Z3Exception("injected")
    tmod._build_tolerance_optimize = fake_boom
    check("求解器异常 → 待判", tolerance_check(forest, clf, inp_tf).status == TOL_PENDING)
    tmod._build_tolerance_optimize = orig_tb
    check("注入恢复后结论正常", tolerance_check(forest, clf, inp_tf).status == TOL_FAIL)

    # 9.8 API：/api/tolerance 四个状态分支
    r = client.post("/api/tolerance", json={"base": tp["base"], "tolerance": tp["tolerance"],
                                            "locked": tp["locked"]})
    rj = r.json()
    check("POST /api/tolerance 放行分支", r.status_code == 200 and rj["status"] == "pass"
          and rj["min_probability"] >= 0.7 and rj["worst"] is not None
          and abs(rj["min_probability"] - res_tp.min_probability) <= 1e-12)
    # 并列最优点时 z3 返回的具体配方可能不同，但返回配方必须在容差域内且复算概率一致
    check("API 放行配方在容差域内且复算达标",
          sum(rj["worst"]) == TOTAL_GRAMS
          and all(tolerance_ranges(inp_tp)[i][0] <= rj["worst"][i] <= tolerance_ranges(inp_tp)[i][1]
                  for i in range(4))
          and abs(predict_proba_qualified(clf, rj["worst"]) - rj["min_probability"]) <= 1e-12)
    r = client.post("/api/tolerance", json={"base": tf["base"], "tolerance": tf["tolerance"],
                                            "locked": tf["locked"]})
    check("POST /api/tolerance 不合格分支", r.json()["status"] == "fail"
          and r.json()["min_probability"] < 0.7)
    r = client.post("/api/tolerance", json={"base": [1, 2, 3, 4], "tolerance": [5] * 4,
                                            "locked": [False] * 4})
    check("POST /api/tolerance 非法输入分支", r.json()["status"] == "invalid")
    html2 = client.get("/").text
    check("页面含容差核查与独立撤回逻辑", "toleranceCheck" in html2 and "tolEpoch" in html2
          and "/api/tolerance" in html2 and "my !== tolEpoch" in html2)

    # 9.9 并发回归：z3 非线程安全，服务端以 SOLVER_LOCK 串行化；
    # 线程池并发求解必须全部成功且结果一致（无锁时 z3 会进程级崩溃）
    import concurrent.futures

    def _post_tol(_):
        r = client.post("/api/tolerance", json={
            "base": tf["base"], "tolerance": tf["tolerance"], "locked": tf["locked"]})
        return r.status_code, r.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(_post_tol, range(8)))
    check("8 路并发容差核查全部 200 且结论一致",
          all(code == 200 for code, _ in outs)
          and len({d["status"] for _, d in outs}) == 1
          and outs[0][1]["status"] == TOL_FAIL
          and all(abs(d["min_probability"] - res_tf.min_probability) <= 1e-12 for _, d in outs),
          f"statuses={[c for c, _ in outs]}")

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
