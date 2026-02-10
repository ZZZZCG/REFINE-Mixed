# -*- coding: utf-8 -*-


import os
import json
import math
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd



def log(msg: str):
    if CONFIG["verbose"]:
        print(msg)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ==============================
# 读取数据
# ==============================
def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    """读取融合向量，并做 L2 归一（仅用于余弦计算）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = list(data.keys())
    mat = np.array([np.array(data[_id], dtype=np.float64) for _id in ids], dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < CONFIG["eps"]] = 1.0
    mat = mat / norms
    return ids, mat

def load_licenses(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    assert {"id", "license"}.issubset(df.columns), "licenses.csv 需要列: id, license"
    return dict(zip(df["id"].astype(str), df["license"].astype(str)))

def load_edges(path: str) -> List[Tuple[str, str, float]]:
    df = pd.read_csv(path)
    assert {"src", "dst"}.issubset(df.columns), "edges.csv 需要列: src, dst[, weight]"
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
        df = df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    else:
        df["weight"] = 1.0
        df = df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    return list(df.itertuples(index=False, name=None))  # (src, dst, weight)

# ==============================
# 结构
# ==============================
@dataclass
class Cluster:
    id: int
    members: Set[int] = field(default_factory=set)
    size: int = 0
    lic_counter: Counter = field(default_factory=Counter)

class AGNESRefine:
    def __init__(self, ids: List[str], vecs: np.ndarray,
                 id2license: Dict[str, str],
                 edges: List[Tuple[str, str, float]]):

        self.ids = ids
        self.vecs = vecs          # 已归一化 (n,d)
        self.n = len(ids)
        self.id2idx = {nid: i for i, nid in enumerate(ids)}
        self.id2license = id2license

        # 覆盖检查
        miss = [nid for nid in ids if nid not in id2license]
        if miss:
            raise ValueError(f"licenses.csv 缺少 {len(miss)} 个 id，例如：{miss[:5]}")

        # 初始化单点簇
        self.active: Dict[int, Cluster] = {}
        for i in range(self.n):
            lic = id2license[self.ids[i]]
            self.active[i] = Cluster(
                id=i, members={i}, size=1, lic_counter=Counter({lic: 1})
            )
        self.next_cluster_id = self.n

        # 有向依赖聚合：W[A][B]、OUT[A]
        self.W: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.OUT: Dict[int, float] = defaultdict(float)
        log("Initializing dependency aggregations (W/OUT)...")
        for s, t, w in edges:
            if s not in self.id2idx or t not in self.id2idx:
                continue
            i, j = self.id2idx[s], self.id2idx[t]
            self.W[i][j] = self.W[i].get(j, 0.0) + float(w)
            self.OUT[i] = self.OUT.get(i, 0.0) + float(w)

        # 堆（无全局版本号；弹出时仅检查是否仍活跃）
        self.heap: List[Tuple[float, int, int]] = []  # (linkv, a, b)

        # 估计 λ
        self.lmbd = self._estimate_lambda() if CONFIG["lambda_mode"] == "auto" else CONFIG["lambda_fixed"]
        log(f"λ (lambda) = {self.lmbd:.6f}")

        # 初始化候选堆
        self._init_heap()

    # ---------- Linkage 每次现算 ----------
    def _D(self, a: int, b: int) -> float:
        """
        链接距离 D(A,B)：
          avg:        1 - mean cos
          complete:   1 - max  cos
          percentile: 1 - qth-percentile cos
          centroid:   1 - cos(centroid_A, centroid_B)
        余弦基于已单位化向量。
        """
        A_idx = np.fromiter(self.active[a].members, dtype=int)
        B_idx = np.fromiter(self.active[b].members, dtype=int)

        lk = CONFIG.get("linkage", "complete")

        if lk == "centroid":
            ca = np.mean(self.vecs[A_idx], axis=0)
            cb = np.mean(self.vecs[B_idx], axis=0)
            na = np.linalg.norm(ca) + CONFIG["eps"]
            nb = np.linalg.norm(cb) + CONFIG["eps"]
            cosv = float((ca/na).dot(cb/nb))
            return 1.0 - cosv

        VA = self.vecs[A_idx]
        VB = self.vecs[B_idx]
        sims = VA @ VB.T
        if sims.size == 0:
            return 1.0

        if lk == "avg":
            stat = float(np.mean(sims))
        elif lk == "percentile":
            q = np.clip(CONFIG.get("percentile_q", 90), 1, 99)
            stat = float(np.percentile(sims, q))
        elif lk == "complete":
            stat = float(np.max(sims))
        else:
            stat = float(np.max(sims))
        return 1.0 - stat

    def _R_src_union(self, a: int, b: int) -> float:
        ca = self.active[a].lic_counter
        cb = self.active[b].lic_counter
        c = ca + cb
        n = sum(c.values())
        if n <= 0: return 0.0
        s2 = sum((v / n) ** 2 for v in c.values())
        return 1.0 - s2

    def _DPull(self, a: int, b: int) -> float:
        out = self.OUT.get(a, 0.0)
        if out <= CONFIG["eps"]:
            return 0.0
        return self.W[a].get(b, 0.0) / out

    def _M_dep(self, a: int, b: int) -> float:
        return 1.0 - max(self._DPull(a, b), self._DPull(b, a))

    def _link(self, a: int, b: int) -> float:
        if a > b: a, b = b, a
        D = self._D(a, b)
        R = self._R_src_union(a, b)
        M = self._M_dep(a, b)
        base = D + self.lmbd * R * M

        # --- 规模平衡 ---
        sa, sb = self.active[a].size, self.active[b].size
        merged = sa + sb
        beta = CONFIG.get("size_balance_beta", 0.0)
        size_penalty = beta * math.log1p(merged)
        return base + size_penalty

    # ---------- λ 自动估计 ----------
    def _estimate_lambda(self) -> float:
        alpha = CONFIG["lambda_alpha"]
        eps = CONFIG["eps"]
        n = self.n
        total_pairs = n * (n - 1) // 2
        cap = CONFIG["lambda_sample_max_pairs"]

        pairs = []
        if total_pairs <= cap:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
        else:
            step = max(1, total_pairs // cap)
            cnt = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if cnt % step == 0:
                        pairs.append((i, j))
                    cnt += 1
        if not pairs:
            log("λ-Estimate: no pairs; fallback λ=1.0")
            return 1.0

        D_vals, PM_vals = [], []
        for (i, j) in pairs:
            cos_ij = float(self.vecs[i].dot(self.vecs[j]))
            d = 1.0 - cos_ij
            li, lj = self.id2license[self.ids[i]], self.id2license[self.ids[j]]
            r = 0.0 if li == lj else 0.5
            dpu = self._DPull(i, j)
            dpv = self._DPull(j, i)
            m = 1.0 - max(dpu, dpv)
            D_vals.append(d)
            PM_vals.append(r * m)

        mean_D = float(np.mean(D_vals)) if D_vals else 1.0
        mean_PM = float(np.mean(PM_vals)) if PM_vals else 1.0
        print(f"[λ-Estimate] mean(D)={mean_D:.6f}, mean(R·M)={mean_PM:.6f}")
        if mean_PM < eps:
            return 1.0
        return alpha * (mean_D / (mean_PM + eps))

    # ---------- 堆初始化 ----------
    def _init_heap(self):
        self.heap = []
        keys = sorted(self.active.keys())
        for ix, a in enumerate(keys):
            for b in keys[ix + 1:]:
                linkv = self._link(a, b)
                heapq.heappush(self.heap, (linkv, a, b))

    # ---------- 最近邻工具 ----------
    def _best_neighbor(self, a: int) -> Tuple[int, float]:
        """返回 a 的最近邻 (b, linkv)。若无其他簇则 (a, +inf)。"""
        best_b, best_v = a, float("inf")
        for b in self.active.keys():
            if b == a: 
                continue
            v = self._link(min(a,b), max(a,b))
            if v < best_v:
                best_v = v
                best_b = b
        return best_b, best_v

    # ---------- 合并（维护 license / 依赖） ----------
    def _merge(self, a: int, b: int) -> int:
        if a not in self.active or b not in self.active:
            return -1
        if a > b: a, b = b, a

        A, B = self.active[a], self.active[b]
        c = self.next_cluster_id
        self.next_cluster_id += 1

        C = Cluster(
            id=c,
            members=A.members | B.members,
            size=A.size + B.size,
            lic_counter=A.lic_counter + B.lic_counter
        )
        self.active[c] = C
        del self.active[a]
        del self.active[b]

        # 依赖聚合：出边
        self.W[c] = {}
        out_c = 0.0
        for x in list(self.active.keys()) + [c]:
            if x == c:
                continue
            wa = self.W.get(a, {}).get(x, 0.0)
            wb = self.W.get(b, {}).get(x, 0.0)
            v = wa + wb
            if v > 0.0:
                self.W[c][x] = v
                out_c += v
        self.OUT[c] = out_c

        # 入边：W[x][c] = W[x][a] + W[x][b]
        for x in list(self.W.keys()):
            if x in (a, b, c):
                continue
            wxa = self.W.get(x, {}).get(a, 0.0)
            wxb = self.W.get(x, {}).get(b, 0.0)
            v = wxa + wxb
            if v > 0.0:
                self.W[x][c] = v
            if a in self.W.get(x, {}):
                del self.W[x][a]
            if b in self.W.get(x, {}):
                del self.W[x][b]

        # 清理旧项
        if a in self.W: del self.W[a]
        if b in self.W: del self.W[b]
        if a in self.OUT: del self.OUT[a]
        if b in self.OUT: del self.OUT[b]

        # 自环清理
        if c in self.W.get(c, {}):
            v = self.W[c][c]
            self.OUT[c] -= v
            del self.W[c][c]

        return c

    # ---------- 主流程 ----------
    def run_and_snapshot(self, Ks: List[int], out_dir: str):
        ensure_dir(out_dir)
        Ks = sorted(set([k for k in Ks if 1 <= k <= self.n]), reverse=True)
        if not Ks:
            log("No valid target K; exit.")
            return
        next_idx = 0
        current_target = Ks[next_idx]
        min_target = Ks[-1]
        diag_every = int(CONFIG.get("diagnose_every_merges", 0))
        nn_every = max(1, int(CONFIG.get("nn_check_every", 1)))
        require_mnn = bool(CONFIG.get("require_mutual_nn", False))
        stop_thr = CONFIG.get("stop_at_distance", None)

        merges_done = 0
        while len(self.active) > min_target and self.heap:
            # 弹出全局最小候选；仅检查 a,b 是否仍活跃
            valid_pair = None
            while self.heap:
                linkv, a, b = heapq.heappop(self.heap)
                if a in self.active and b in self.active:
                    # 可选：停止阈值
                    if stop_thr is not None and linkv > stop_thr:
                        log(f"Stop: min distance {linkv:.6f} > threshold {stop_thr}")
                        self._snapshot(out_dir, len(self.active))
                        return
                    valid_pair = (linkv, a, b)
                    break
            if valid_pair is None:
                break

            linkv, a, b = valid_pair

            # 互为最近邻约束（防链式吞并）
            if require_mnn and (merges_done % nn_every == 0):
                ba, va = self._best_neighbor(a)
                bb, vb = self._best_neighbor(b)
                if not ((ba == b) and (bb == a)):
                    # 不是互为最近邻：跳过这对，推回堆末（以免遗失），继续弹下一对
                    heapq.heappush(self.heap, (linkv, a, b))
                    continue

            # 合并
            c = self._merge(a, b)
            merges_done += 1

            # 诊断：查看是否巨簇化
            if diag_every > 0 and (merges_done % diag_every == 0):
                sizes = [cl.size for cl in self.active.values()]
                top = sorted(sizes, reverse=True)[:5]
                log(f"[merge {merges_done}] k={len(self.active)} max={top[0]} top5={top}")

            # 新一轮候选：c 与其他簇（Link 现算）
            for x in self.active.keys():
                if x == c:
                    continue
                aa, bb = (c, x) if c < x else (x, c)
                val = self._link(aa, bb)
                heapq.heappush(self.heap, (val, aa, bb))

            # 若达到目标 K，则快照
            k_now = len(self.active)
            while next_idx < len(Ks) and k_now == current_target:
                self._snapshot(out_dir, current_target)
                next_idx += 1
                if next_idx < len(Ks):
                    current_target = Ks[next_idx]

        # 结束前留一份快照
        self._snapshot(out_dir, len(self.active))

    def _snapshot(self, out_dir: str, K: int):
        payload = {}
        for cid, C in self.active.items():
            payload[str(cid)] = [self.ids[i] for i in sorted(C.members)]
        if CONFIG["include_lambda_in_filename"]:
            out_path = os.path.join(out_dir, f"clusters_{K}_lambda{self.lmbd:.4f}.json")
        else:
            out_path = os.path.join(out_dir, f"clusters_{K}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log(f"Snapshot -> {out_path}  (|clusters|={K})")

# ==============================
# 配置
# ==============================
CONFIG = {
    # ---- 输入路径 ----
    "fused_embeddings_path": r"./fused_embeddings.json",
    "licenses_csv_path":     r"./licenses.csv",     # id,path,license
    "edges_csv_path":        r"./edges.csv",        # src,dst[,weight]

    "output_dir":            r"./clusters_refine",
    "target_k_min":          15,
    "target_k_max":          40,
    "include_lambda_in_filename": False,

    # ---- 链接策略 ----
    # 可选："avg"|"complete"|"percentile"|"centroid"
    "linkage":               "complete",
    "percentile_q":          90,   # linkage="percentile" 时使用

    # ---- 规模平衡（抑制巨簇）----
    # 合并代价 += beta * log(1 + |A| + |B|)
    "size_balance_beta":     0.25,

    # ---- λ 设定 ----
    "lambda_mode":           "fixed",    # "auto"|"fixed"
    "lambda_fixed":          1.0,
    "lambda_alpha":          1.0,
    "lambda_sample_max_pairs": 200000,

    # ---- MNN 约束 & 停止阈值 ----
    "require_mutual_nn":     True,       # 开启后，只有互为最近邻的一对才允许合并
    "nn_check_every":        1,          # 每合并几次后做一次 MNN 检查（1=每次）
    "stop_at_distance":      None,       # e.g. 0.6；当最小距离>阈值时停止（不强制K）

    # ---- 数值与日志 ----
    "eps":                   1e-12,
    "verbose":               True,
    "diagnose_every_merges": 200,
}

# ==============================
# 入口
# ==============================
def main():
    cfg = CONFIG
    ensure_dir(cfg["output_dir"])

    ids, vecs = load_embeddings(cfg["fused_embeddings_path"])
    id2license = load_licenses(cfg["licenses_csv_path"])
    edges = load_edges(cfg["edges_csv_path"])

    ag = AGNESRefine(ids, vecs, id2license, edges)
    target_Ks = list(range(cfg["target_k_min"], cfg["target_k_max"] + 1))
    ag.run_and_snapshot(target_Ks, cfg["output_dir"])
    log("Done.")

if __name__ == "__main__":
    main()
