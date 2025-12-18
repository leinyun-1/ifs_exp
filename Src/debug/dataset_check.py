"""
数据集结构校验工具，支持 A/A+/B 结构：
- A: render/image/{sub}_{vid}.png + mask 同名 + ProjParams.pkl
- A+: render/image/{sub}_{vid}_{lid}.png (lid=0/1/2) + mask 同名 + ProjParams.pkl
- B: img/{sub}/{vid}.png + mask/{sub}/{vid}.png + parameter/{sub}/{vid}_intrinsic.npy / _extrinsic.npy
"""
import argparse
import glob
import os
import pickle
from typing import Sequence, Set


def _check_param_size(param_dir: str):
    param_files = sorted(glob.glob(os.path.join(param_dir, "*_params.pkl")))
    sizes = {os.path.basename(p): os.path.getsize(p) for p in param_files}
    uniq = set(sizes.values())
    if len(uniq) <= 1:
        size_text = "unknown" if not uniq else str(uniq.pop())
        print(f"[OK] param 文件大小一致，共 {len(param_files)} 个，大小 {size_text} 字节")
    else:
        print("[WARN] param 文件大小不一致：")
        for name, sz in sizes.items():
            print(f"  {name}: {sz}")
    return param_files


def check_a_plus(render_dir: str, lids: Sequence[int] = (0, 1, 2)):
    image_dir = os.path.join(render_dir, "image")
    param_dir = os.path.join(render_dir, "param")

    param_files = _check_param_size(param_dir)
    if not param_files:
        print(f"[WARN] 未找到 param 文件: {param_dir}")
        return

    expected_lids: Set[int] = set(lids)
    missing, extra = [], []

    for pkl_path in param_files:
        sub = os.path.basename(pkl_path).replace("_params.pkl", "")
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
        vids = set(int(k.split("_")[-1]) for k in params.keys())
        for vid in vids:
            files = glob.glob(os.path.join(image_dir, f"{sub}_{vid}_*.png"))
            lids_found = set()
            for fp in files:
                try:
                    lids_found.add(int(os.path.splitext(fp)[0].split("_")[-1]))
                except Exception:
                    continue
            miss = expected_lids - lids_found
            if miss:
                missing.append((sub, vid, sorted(miss)))
            if len(lids_found) > len(expected_lids):
                extras = sorted(list(lids_found - expected_lids))
                if extras:
                    extra.append((sub, vid, extras))

    if not missing and not extra:
        print(f"[OK] image 覆盖完整，每个 (sub, vid) 有 lid {sorted(expected_lids)} 且无多余。")
    else:
        if missing:
            print("[MISS] 缺失：")
            for sub, vid, lids in missing:
                print(f"  {sub} vid={vid} 缺 lid={lids}")
        if extra:
            print("[EXTRA] 多余：")
            for sub, vid, lids in extra:
                print(f"  {sub} vid={vid} 多余 lid={lids}")


def main():
    parser = argparse.ArgumentParser(description="数据集结构校验")
    sub = parser.add_subparsers(dest="mode", required=True)

    aplus = sub.add_parser("a+", help="校验 A+ 多光照数据集")
    aplus.add_argument("--render_dir", required=True)
    aplus.add_argument("--lids", nargs="*", type=int, default=[0, 1, 2])

    args = parser.parse_args()

    if args.mode == "a+":
        check_a_plus(args.render_dir, args.lids)


if __name__ == "__main__":
    main()
