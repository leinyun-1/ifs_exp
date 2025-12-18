"""
模型简要 smoke test。
检查模型的输入输出
"""
import argparse
import torch
from types import SimpleNamespace

from Model.HourGlass import HGFilter
from Model.SurfaceClassifier import SurfaceClassifier


def test_hourglass(device="cpu"):
    config = {
        "num_stack": 4,
        "num_hourglass": 2,
        "hourglass_dim": 256,
        "hg_down": 'ave_pool',
        "norm": 'group'
    }
    cfg = SimpleNamespace(**config)
    net = HGFilter(cfg).to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    y = net(x)
    print("HG output:", y.shape)


def test_surface_classifier(device="cpu"):
    chnnels = [323, 1024, 512, 256, 128, 1]
    net = SurfaceClassifier(chnnels).to(device)
    x = torch.randn(1, 323, 8000, device=device)
    y = net(x)
    print("SurfaceClassifier output:", y.shape)


def main():
    parser = argparse.ArgumentParser(description="模型 smoke test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target", choices=["hg", "mlp", "all"], default="all")
    args = parser.parse_args()

    if args.target in ("hg", "all"):
        test_hourglass(args.device)
    if args.target in ("mlp", "all"):
        test_surface_classifier(args.device)


if __name__ == "__main__":
    main()
