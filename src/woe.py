"""Weight-of-Evidence binning.

Convention: WoE = ln(%good / %bad). Higher WoE means lower risk.
Missing values always get their own bin.

Numeric characteristics: start from quantile bins, then merge neighbours until
(a) every bin holds at least `min_bin_share` of the sample and
(b) the bad rate moves in one direction (monotonic), so the scorecard
    never rewards a characteristic for getting worse.
Categorical characteristics: categories below `min_bin_share` are pooled into
"OTHER"; unseen categories at scoring time also map to "OTHER".
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5  # added to good/bad counts so empty cells don't give infinite WoE


@dataclass
class Binning:
    name: str
    kind: str                       # "numeric" | "categorical"
    edges: list = field(default_factory=list)      # numeric: inner cut points
    groups: dict = field(default_factory=dict)     # categorical: value -> label
    table: pd.DataFrame = None
    iv: float = 0.0

    # --- scoring -----------------------------------------------------------
    def assign(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            labels = pd.Series(np.digitize(x.to_numpy(dtype=float), self.edges), index=x.index)
            labels = labels.map(lambda i: f"B{i:02d}")
            labels[x.isna()] = "MISSING"
            return labels
        mapped = x.fillna("MISSING").map(self.groups)
        return mapped.fillna("OTHER")

    def transform(self, x: pd.Series) -> pd.Series:
        woe = self.table.set_index("bin")["woe"]
        return self.assign(x).map(woe).fillna(0.0)


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": bins, "bad": y}).groupby("bin")["bad"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    tot_good, tot_bad = t["good"].sum(), t["bad"].sum()
    dist_good = (t["good"] + SMOOTH) / (tot_good + SMOOTH * len(t))
    dist_bad = (t["bad"] + SMOOTH) / (tot_bad + SMOOTH * len(t))
    t["share"] = t["n"] / t["n"].sum()
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t.reset_index()


def _merge_numeric(x: pd.Series, y: pd.Series, max_bins: int, min_share: float) -> list:
    present = x.notna()
    xs, ys = x[present], y[present]
    edges = list(np.unique(np.quantile(xs, np.linspace(0, 1, max_bins + 1)[1:-1])))

    xv, yv = xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)

    def stats(edges):
        idx = np.digitize(xv, edges)
        counts = np.bincount(idx, minlength=len(edges) + 1)
        bads = np.bincount(idx, weights=yv, minlength=len(edges) + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return counts, bads / counts

    # Direction of risk: sign of rank correlation between x and bad.
    direction = np.sign(pd.Series(xs).corr(ys, method="spearman")) or 1.0

    changed = True
    while changed and edges:
        changed = False
        counts, rates = stats(edges)
        empty = np.where(counts == 0)[0]
        if len(empty):                             # remove the cut that creates the empty bin
            i = empty[0]
            edges.pop(i - 1 if i > 0 else 0)
            changed = True
            continue
        shares = counts / counts.sum()
        # 1) too-small bins: merge with the neighbour closest in bad rate
        small = np.where(shares < min_share)[0]
        if len(small):
            i = small[0]
            if i == 0:
                j = 0
            elif i == len(counts) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1]) else i
            edges.pop(j)
            changed = True
            continue
        # 2) monotonic bad rate in the expected direction
        diffs = np.diff(rates) * direction
        bad_steps = np.where(diffs < 0)[0]
        if len(bad_steps):
            edges.pop(bad_steps[0])
            changed = True
    return [float(e) for e in edges]


def _group_categories(x: pd.Series, y: pd.Series, min_share: float) -> dict:
    """Large categories keep their own bin. Small ones are sorted by bad rate and
    pooled with neighbours of similar risk until each pool reaches `min_share`,
    so a rare but risky category is not averaged away with rare safe ones."""
    stats = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["size", "mean"])
    stats["share"] = stats["size"] / stats["size"].sum()
    groups = {v: v for v in stats.index[stats["share"] >= min_share]}
    small = stats[stats["share"] < min_share].sort_values("mean")

    pools, current, acc = [], [], 0.0
    for v, r in small.iterrows():
        current.append(v); acc += r["share"]
        if acc >= min_share:
            pools.append(current); current, acc = [], 0.0
    if current:                                   # leftover too small: join the last pool
        if pools:
            pools[-1].extend(current)
        else:
            pools.append(current)
    for k, members in enumerate(pools, 1):
        for v in members:
            groups[v] = f"POOL{k}"
    return groups


def fit_binning(x: pd.Series, y: pd.Series, name: str, kind: str,
                max_bins: int = 10, min_bin_share: float = 0.05) -> Binning:
    b = Binning(name=name, kind=kind)
    if kind == "numeric":
        b.edges = _merge_numeric(x, y, max_bins, min_bin_share)
    else:
        b.groups = _group_categories(x.fillna("MISSING"), y, min_bin_share)
    b.table = _woe_table(b.assign(x), y)
    b.iv = float(b.table["iv"].sum())
    return b


def describe_bins(b: Binning) -> pd.DataFrame:
    """Readable bin labels, e.g. '[12.5, 18.0)'."""
    t = b.table.copy()
    if b.kind == "numeric":
        bounds = [-np.inf] + b.edges + [np.inf]
        names = {f"B{i:02d}": f"[{bounds[i]:,.4g}, {bounds[i + 1]:,.4g})"
                 for i in range(len(bounds) - 1)}
        names["MISSING"] = "Missing"
        t["range"] = t["bin"].map(names)
    else:
        members = {}
        for v, g in b.groups.items():
            members.setdefault(g, []).append(str(v))
        t["range"] = t["bin"].map(lambda g: ", ".join(sorted(members.get(g, [g]))))
    t.insert(0, "characteristic", b.name)
    return t
