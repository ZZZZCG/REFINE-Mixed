# -*- coding: utf-8 -*-
import json
import numpy as np
import os
from typing import Set, Any

# ==============================
# 配置项（可直接修改）
# ==============================
CONFIG = {
    "functional_embeddings": r"./functional_embeddings.json",
    "dependency_embeddings": r"./dependency_embeddings.json",
    "path_embeddings": r"./path_embeddings.json",
    "license_embeddings": r"./license_embeddings.json",
    "output_path": r"./fused_embeddings.json",

    # ---- 新增：只保留 subset.json 中列出的实体 ----
    # 支持两种结构：
    #   1) ["a.c", "b.c", ...]
    #   2) {"1": ["a.c", "b.c"], "2": ["x/y.c", ...], ...}
    "subset_json_path": None,          # 例如 r"./bash_flat.json"；None 表示不筛选
    # 规范化匹配开关：True 时会统一大小写、斜杠、去掉开头的 ./ 或 / ，减少路径差异带来的漏匹配
    "canonical_match": False,
    "canon_lower": True,
    "canon_slash": True,
    "canon_strip_dot": True,
}

# ==============================
# 工具函数
# ==============================
def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # dict: id -> [vector...]

def _flatten_subset_json(obj: Any) -> Set[str]:
    """把 subset.json 展平成 {id,...} 集合；支持 list[str] 或 dict[str|int, list[str]|str]"""
    keep: Set[str] = set()
    if obj is None:
        return keep
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, str) and x:
                keep.add(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x:
                        keep.add(x)
            elif isinstance(v, str) and v:
                keep.add(v)
    elif isinstance(obj, str) and obj:
        keep.add(obj)
    return keep

def canonize_id(s: str, *, lower=True, slash=True, strip_dot=True) -> str:
    """可选的 ID 规范化：lower、'\\'->'/'、去掉开头 './' 或 '/'"""
    if s is None:
        return ""
    t = str(s)
    if slash:
        t = t.replace("\\", "/")
    if strip_dot:
        while t.startswith("./"):
            t = t[2:]
        while t.startswith("/"):
            t = t[1:]
    if lower:
        t = t.lower()
    return t

def build_canon_map(keys, *, lower=True, slash=True, strip_dot=True):
    """构建 规范化id -> 原始id 的映射（如有冲突保留第一个）"""
    m = {}
    for k in keys:
        c = canonize_id(k, lower=lower, slash=slash, strip_dot=strip_dot)
        m.setdefault(c, k)
    return m

# ==============================
# 拼接融合（仅保留 subset）
# ==============================
def fuse_embeddings(cfg):
    func_emb = load_embeddings(cfg["functional_embeddings"])
    dep_emb  = load_embeddings(cfg["dependency_embeddings"])
    path_emb = load_embeddings(cfg["path_embeddings"])
    lic_emb  = load_embeddings(cfg["license_embeddings"])

    # 1) 四种信息共同存在的键
    keys_all = set(func_emb.keys()) & set(dep_emb.keys()) & set(path_emb.keys()) & set(lic_emb.keys())
    print(f"四模态交集共有 {len(keys_all)} 个节点。")

    # 2) 如果提供了 subset.json，则进一步筛选
    if cfg["subset_json_path"]:
        with open(cfg["subset_json_path"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        subset_raw = _flatten_subset_json(raw)
        if not subset_raw:
            print("警告：subset.json 为空，结果将为空。")

        if cfg["canonical_match"]:
            # 规范化对齐
            emb_canon2orig = build_canon_map(
                keys_all,
                lower=cfg["canon_lower"],
                slash=cfg["canon_slash"],
                strip_dot=cfg["canon_strip_dot"],
            )
            matched_orig = []
            miss = []
            for k in subset_raw:
                c = canonize_id(
                    k,
                    lower=cfg["canon_lower"],
                    slash=cfg["canon_slash"],
                    strip_dot=cfg["canon_strip_dot"],
                )
                if c in emb_canon2orig:
                    matched_orig.append(emb_canon2orig[c])
                else:
                    miss.append(k)
            keys_kept = set(matched_orig)
            print(f"[subset] 目标 {len(subset_raw)} 个；命中 {len(keys_kept)} 个；未命中 {len(miss)} 个。")
            if miss:
                print(f"[subset] 示例未命中：{miss[:5]}")
        else:
            # 精确匹配（完全一致）
            keys_kept = keys_all & subset_raw
            miss = sorted(list(subset_raw - keys_all))[:5]
            print(f"[subset] 精确匹配：保留 {len(keys_kept)} 个；示例未命中：{miss}")
    else:
        keys_kept = keys_all
        print("未提供 subset.json，保留四模态交集全部节点。")

    if not keys_kept:
        print("没有任何键被保留，直接写出空文件。")
        with open(cfg["output_path"], "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    # 3) 融合
    fused = {}
    for key in keys_kept:
        v_func = np.array(func_emb[key], dtype=float)
        v_dep  = np.array(dep_emb[key],  dtype=float)
        v_path = np.array(path_emb[key], dtype=float)
        v_lic  = np.array(lic_emb[key],  dtype=float)
        fused[key] = np.concatenate([v_func, v_dep, v_path, v_lic]).tolist()

    # 4) 保存
    with open(cfg["output_path"], "w", encoding="utf-8") as f:
        json.dump(fused, f, indent=2, ensure_ascii=False)
    print(f"融合向量已保存到 {cfg['output_path']}；最终保留 {len(fused)} 个节点。")

# ==============================
# 主函数
# ==============================
if __name__ == "__main__":
    fuse_embeddings(CONFIG)
