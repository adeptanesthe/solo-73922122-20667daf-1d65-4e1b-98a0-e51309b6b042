"""容差核查：在称量误差容差域内寻找最低合格概率（最差情形分析）。

输入：四种原料总重 1000 克的整数基准配方、各原料允许偏差（整数克）、锁定项。
容差域（全部为整数克数配方）：
  g_i ∈ [max(0, base_i − tol_i), min(1000, base_i + tol_i)]，锁定项 g_i = base_i，
  g_i ≥ 0，且 sum(g_i) = 1000。

在该域上以 z3 Optimize **最小化**森林平均合格概率。编码与试配求解共用
（forest_enc），分裂条件按原分类器 float32 特征 vs float64 阈值的真实语义
精确翻译成整数克数比较，叶子概率为精确有理数。

可判定口径（只有同时满足才下结论，否则一律“待判”）：
  1. z3 返回 sat 且最小化目标的上下界严格相等（最优值已被证明，而非近似）；
  2. 最差配方确实落在容差域内；
  3. 精确有理概率与原 sklearn 分类器 float32 推理复算值一致；
  4. 有理侧与浮点侧对“是否 ≥ 0.7”的判定一致。
结论：最差概率 ≥ 0.7 → 可放行；已证实 < 0.7 → 给出不合格反例；其余 → 待判。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction

import z3

from .forest_enc import encode_forest_leaves, forest_prob_sum_expr
from .model import (
    FEATURE_NAMES,
    ForestSpec,
    N_FEATURES,
    N_TREES,
    PROBA_THRESHOLD,
    TOTAL_GRAMS,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
)
from .solver import INVALID

TOLERANCE_TIMEOUT_MS = 60_000

# 状态取值
TOL_PASS = "pass"        # 已证明容差域内最差概率 ≥ 0.7，可放行
TOL_FAIL = "fail"        # 已证实存在概率 < 0.7 的不合格反例（且为域内最低点）
TOL_PENDING = "pending"  # 求解未能判定（超时/未知/复核不一致），保持待判

# 精确的阈值有理值：网格点的概率分母均为小整数，不可能落在 7/10 与
# float64(0.7) 之间，故以十进制 0.7 = 7/10 作为精确判定口径。
THRESHOLD_FRAC = Fraction(7, 10)
# 有理概率与 sklearn float64 复算值的一致性容差（吸收求和顺序尾数差异）。
RECHECK_EPS = 1e-12


@dataclass
class ToleranceInput:
    base: list[int]        # 基准配方（克），总重 1000
    tolerance: list[int]   # 各原料允许偏差（整数克，单边）
    locked: list[bool]


@dataclass
class ToleranceResult:
    status: str
    reason: str = ""
    worst: list[int] | None = None              # 最差配方（克）
    min_probability: float | None = None        # 原分类器复算的最差概率
    rational_min: list[int] | None = None       # 最差概率的精确有理值 [分子, 分母]
    base_probability: float | None = None       # 基准配方自身的复算概率
    region_size: int | None = None              # 容差域内可行配方总数
    per_ingredient: list[dict] | None = None
    solve_ms: float = 0.0
    optimality: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "worst": self.worst,
            "min_probability": self.min_probability,
            "rational_min": self.rational_min,
            "base_probability": self.base_probability,
            "region_size": self.region_size,
            "per_ingredient": self.per_ingredient,
            "solve_ms": round(self.solve_ms, 1),
            "optimality": self.optimality,
            "threshold": PROBA_THRESHOLD,
        }


def validate_tolerance(inp: ToleranceInput) -> list[str]:
    errs = []
    if len(inp.base) != N_FEATURES or any(
        (not isinstance(v, int)) or isinstance(v, bool) or v < 0 for v in inp.base
    ):
        errs.append(f"基准配方必须是 {N_FEATURES} 个非负整数克数")
    if len(inp.tolerance) != N_FEATURES or any(
        (not isinstance(v, int)) or isinstance(v, bool) or v < 0 for v in inp.tolerance
    ):
        errs.append(f"允许偏差必须是 {N_FEATURES} 个非负整数克数")
    if len(inp.locked) != N_FEATURES or any(not isinstance(v, bool) for v in inp.locked):
        errs.append(f"locked 必须是 {N_FEATURES} 个布尔值")
    if errs:
        return errs
    if sum(inp.base) != TOTAL_GRAMS:
        errs.append(f"基准配方总克数必须等于 {TOTAL_GRAMS}（当前 {sum(inp.base)}）")
    # 注：基准总重 1000、用量非负且偏差为对称闭区间时，基准配方自身就在容差域内，
    # 因此容差域恒非空（selftest 第 9 节以 5000 组随机合法输入验证计数恒 ≥ 1），
    # 无需额外的可配平校验；count_feasible 仍对极端区间做防御性统计。
    return errs


def tolerance_ranges(inp: ToleranceInput) -> list[tuple[int, int]]:
    """各原料在容差域内的整数闭区间（已截断到 [0, 1000]，锁定项退化为单点）。"""
    out = []
    for i in range(N_FEATURES):
        if inp.locked[i]:
            out.append((inp.base[i], inp.base[i]))
        else:
            out.append((
                max(0, inp.base[i] - inp.tolerance[i]),
                min(TOTAL_GRAMS, inp.base[i] + inp.tolerance[i]),
            ))
    return out


def count_feasible(ranges: list[tuple[int, int]]) -> int:
    """动态规划统计满足各原料区间且总重恰好 1000 的整数配方数。

    递推：ndp[s] = Σ_{j ∈ [lo, hi]} dp[s-j]，用 dp 的前缀和 O(1) 求区间和。
    """
    dp = [0] * (TOTAL_GRAMS + 1)
    dp[0] = 1
    for lo, hi in ranges:
        prefix = [0] * (TOTAL_GRAMS + 2)  # prefix[k] = Σ_{t=0}^{k-1} dp[t]
        for t in range(TOTAL_GRAMS + 1):
            prefix[t + 1] = prefix[t] + dp[t]
        ndp = [0] * (TOTAL_GRAMS + 1)
        for s in range(TOTAL_GRAMS + 1):
            # 求 Σ dp[t]，t 满足 lo ≤ s-t ≤ hi 且 t ≥ 0
            # 即 t ∈ [max(0, s-hi), s-lo]；右端钳制到 s 防止 j=s-t 为负
            a = max(0, s - hi)
            b = min(s, s - lo)
            if b >= a:
                ndp[s] = prefix[b + 1] - prefix[a]
        dp = ndp
    return dp[TOTAL_GRAMS]


def _build_tolerance_optimize(forest: ForestSpec, inp: ToleranceInput):
    """构造容差核查的 Optimize（供求解与自测复用）。"""
    ranges = tolerance_ranges(inp)
    g = z3.IntVector("g", N_FEATURES)
    opt = z3.Optimize()
    opt.set("timeout", TOLERANCE_TIMEOUT_MS)
    for i in range(N_FEATURES):
        lo, hi = ranges[i]
        opt.add(g[i] >= lo, g[i] <= hi)
        if inp.locked[i]:
            opt.add(g[i] == inp.base[i])
    opt.add(z3.Sum(g) == TOTAL_GRAMS)
    trees_select = encode_forest_leaves(opt, forest, g)
    total_expr = forest_prob_sum_expr(trees_select)  # 5 棵树概率之和（精确有理）
    handle = opt.minimize(total_expr)
    return opt, g, handle, ranges


def tolerance_check(forest: ForestSpec, clf, inp: ToleranceInput) -> ToleranceResult:
    """最小化容差域内合格概率；仅在严格证明下给放行/不合格结论，否则待判。"""
    t0 = time.perf_counter()
    errs = validate_tolerance(inp)
    if errs:
        return ToleranceResult(status=INVALID, reason="；".join(errs))

    base_proba = predict_proba_qualified(clf, inp.base)
    ranges = tolerance_ranges(inp)
    region_size = count_feasible(ranges)
    elapsed = lambda: (time.perf_counter() - t0) * 1000

    try:
        opt, g, handle, ranges = _build_tolerance_optimize(forest, inp)
        r = opt.check()
        if r != z3.sat:
            return ToleranceResult(
                status=TOL_PENDING,
                reason=f"求解器未能判定（{r}，可能超时）：未能证明容差域内的最低概率，结论保持待判。",
                base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
            )
        lb, ub = opt.lower(handle), opt.upper(handle)
        # 只有上下界相等才表示最优值已被证明；unknown 下可能有模型但界未收敛。
        if not lb.eq(ub):
            return ToleranceResult(
                status=TOL_PENDING,
                reason="求解器在时限内未把最低概率收敛到唯一最优值，结论未经证明，保持待判。",
                base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
            )
        m = opt.model()
        try:
            worst = [m.eval(g[i], model_completion=True).as_long() for i in range(N_FEATURES)]
        except (AttributeError, z3.Z3Exception):
            return ToleranceResult(
                status=TOL_PENDING,
                reason="求解器返回的模型未给出完整整数配方，结论保持待判。",
                base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
            )
    except z3.Z3Exception as exc:
        return ToleranceResult(
            status=TOL_PENDING,
            reason=f"求解器异常：{exc}；结论保持待判。",
            base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
        )

    # ---- 证明后的独立复核 ----
    if (
        sum(worst) != TOTAL_GRAMS
        or any(worst[i] < ranges[i][0] or worst[i] > ranges[i][1] for i in range(N_FEATURES))
    ):
        return ToleranceResult(
            status=TOL_PENDING,
            reason="内部不一致：求解器给出的最差配方不在容差域内，结论保持待判。",
            base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
        )

    rational = forest_proba_rational(forest, worst)          # 精确有理平均概率
    bound_frac = lb.as_fraction() / N_TREES                  # z3 证明的最优值
    if bound_frac != rational:
        return ToleranceResult(
            status=TOL_PENDING,
            reason="内部不一致：z3 最优界与走树复算的精确概率不相等，结论保持待判。",
            base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
        )

    proba = predict_proba_qualified(clf, worst)              # 原分类器 float32 推理复算
    if abs(proba - float(rational)) > RECHECK_EPS:
        return ToleranceResult(
            status=TOL_PENDING,
            reason=(
                f"内部不一致：原分类器复算概率 {proba:.6f} 与精确编码概率 "
                f"{float(rational):.6f} 不一致，结论保持待判。"
            ),
            base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
        )

    rational_pass = rational >= THRESHOLD_FRAC
    float_pass = is_qualified(proba)
    if rational_pass != float_pass:
        # 理论上不会发生（网格概率间隔远大于浮点误差），防御性处理为待判。
        return ToleranceResult(
            status=TOL_PENDING,
            reason=(
                f"内部不一致：精确判定（{rational} ≥ 0.7 为 {rational_pass}）"
                f"与浮点判定（{proba:.6f} ≥ 0.7 为 {float_pass}）冲突，结论保持待判。"
            ),
            base_probability=base_proba, region_size=region_size, solve_ms=elapsed(),
        )

    per = [
        {
            "name": FEATURE_NAMES[i],
            "base": inp.base[i],
            "worst": worst[i],
            "delta": worst[i] - inp.base[i],
            "tolerance": inp.tolerance[i],
            "lower": ranges[i][0],
            "upper": ranges[i][1],
            "locked": inp.locked[i],
        }
        for i in range(N_FEATURES)
    ]
    ms = elapsed()
    if rational_pass:
        status = TOL_PASS
        optimality = (
            f"已证明：容差域内全部 {region_size} 个可行配方中，合格概率最低点为 "
            f"{worst}（克），其概率 {proba:.6f} ≥ 0.7；即所有称量误差内配方均合格，可放行。"
        )
        reason = "最差概率已证明不低于阈值 0.7。"
    else:
        status = TOL_FAIL
        optimality = (
            f"已证实不合格反例：配方 {worst}（克）在容差域内，合格概率仅 {proba:.6f} < 0.7；"
            f"z3 已证明该点是容差域内全部 {region_size} 个可行配方的概率最低点，不可放行。"
        )
        reason = f"容差域内存在合格概率低于 0.7 的配方（最低 {proba:.6f}）。"
    return ToleranceResult(
        status=status,
        reason=reason,
        worst=worst,
        min_probability=proba,
        rational_min=[rational.numerator, rational.denominator],
        base_probability=base_proba,
        region_size=region_size,
        per_ingredient=per,
        solve_ms=ms,
        optimality=optimality,
    )
