"""
数据增强示例：读取单张 RGB+mask，复用 BaseIFS 的增强流水线输出多组结果。
"""
from pathlib import Path
from typing import Tuple

from PIL import Image

from Dataset.BaseIFS import BaseIFS


class _AugmentOnly(BaseIFS):
    def __init__(self):
        # 不调用父类 __init__，手动设置增强配置
        self.is_train = True
        self.roi = 512
        self.aug_cfg = {
            "enable": True,
            "color_jitter": {"p": 1.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            "gaussian_blur": {"p": 1.0, "radius": 1.0},
            "gaussian_noise": {"p": 1.0, "std": 0.01},
            "random_mask_drop": {"p": 1.0, "max_ratio": 0.2},
            "geom_jitter": {
                "enable": True,
                "p": 0.7,
                "pad_ratio": 0.1,
                "pad_p": 1.0,
                "scale_min": 0.9,
                "scale_max": 1.1,
                "scale_p": 1.0,
                "trans_ratio": 0.1,
                "trans_p": 1.0,
            },
        }

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        self.roi = min(image.size)
        image, mask, _ = self._augment_render(image, mask, intrinsic=None)
        return image, mask


def run_augment(image_path: Path, mask_path: Path, out_dir: Path, samples: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    augmenter = _AugmentOnly()

    for idx in range(samples):
        aug_img, aug_mask = augmenter(image, mask)
        aug_img.save(out_dir / f"aug_image_{idx}.png")
        aug_mask.save(out_dir / f"aug_mask_{idx}.png")
    print(f"保存增强结果到 {out_dir}, 共 {samples} 组")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BaseIFS 数据增强示例")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    run_augment(args.image, args.mask, args.output, args.samples)
