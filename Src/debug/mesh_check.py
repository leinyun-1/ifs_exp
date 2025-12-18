"""
mesh bbox 检查工具，可并行。
检查所有mesh是否bbox大小一致
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import trimesh
from tqdm import tqdm


def _check_mesh(path, threshold):
    try:
        mesh = trimesh.load(path, force="mesh")
        bmin, bmax = mesh.bounds
        bad = (bmin < -threshold).any() or (bmax > threshold).any()
        return path, bmin, bmax, bad, None
    except Exception as e:
        return path, None, None, True, str(e)


def check_mesh_bbox(mesh_dir: str, threshold: float = 1.5, workers: int = None):
    objs = glob.glob(os.path.join(mesh_dir, "*", "*.obj"))
    if not objs:
        print(f"[WARN] 未找到 obj 文件于 {mesh_dir}")
        return

    if workers is None:
        workers = os.cpu_count() or 4

    bad = []
    if workers == 1:
        for path in tqdm(objs, desc="checking bbox"):
            res = _check_mesh(path, threshold)
            if res[3]:
                bad.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_check_mesh, p, threshold): p for p in objs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="checking bbox"):
                res = fut.result()
                if res[3]:
                    bad.append(res)

    if not bad:
        print(f"[OK] 所有 mesh 的 bbox 均在 [{-threshold}, {threshold}] 范围内")
    else:
        print("[WARN] 检测到异常 bbox：")
        for path, bmin, bmax, _, err in bad:
            print(f"  {path}")
            print(f"    bmin: {bmin}, bmax: {bmax}, err: {err}")


def main():
    parser = argparse.ArgumentParser(description="mesh bbox 检查")
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    check_mesh_bbox(args.mesh_dir, args.threshold, args.workers)


if __name__ == "__main__":
    main()
