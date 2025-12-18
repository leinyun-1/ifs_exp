"""
dia/stride 可视化工具，展示采样中心在细网格坐标下的分布。
可视化Mr.AllenYu 的推理滑动窗口规划
' ' ' '
 ' ' ' ' 
''''''''
大小为4的窗口仅仅偏移了半个体素，但是实现了分辨率提升一倍
这是在窗口体素分辨率固定的情况下实现高密度体素推理的一种实现方案
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def schedule_centers(bmin, size, rom, rov, dia, stride):
    sos = size / rom
    candates = np.mgrid[0:rom - rov + 1:stride, 0:rom - rov + 1:stride, 0:rom - rov + 1:stride].reshape(3, -1)
    deltas = np.mgrid[0:dia, 0:dia, 0:dia].reshape(3, -1)
    indices, centers = [], []
    for i in range(candates.shape[1]):
        for j in range(deltas.shape[1]):
            idx = candates[:, i]
            delta = deltas[:, j]
            center = bmin + sos * idx + sos * rov / 2 + sos * delta / dia
            indices.append(idx * dia + delta)
            centers.append(center)
    return np.array(indices), np.array(centers)


def visualize(bmin=np.array([0, 0, 0]), size=np.array([1, 1, 1]), rom=128, rov=32, stride=32, dia=2):
    sos = size / rom
    _, ctr = schedule_centers(bmin, size, rom, rov, dia, stride)
    fine_centers = (ctr - bmin) / sos * dia
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fine_centers[:, 0], fine_centers[:, 1], fine_centers[:, 2], c='r', s=10, label='centers (fine grid)')
    ax.set_xlabel('x (fine vox)')
    ax.set_ylabel('y (fine vox)')
    ax.set_zlabel('z (fine vox)')
    ax.set_title(f'rom={rom}, rov={rov}, stride={stride}, dia={dia}\nlabels grid size={(rom * dia)}^3')
    ax.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="dia/stride 可视化")
    parser.add_argument("--rom", type=int, default=512)
    parser.add_argument("--rov", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--dia", type=int, default=2)
    args = parser.parse_args()
    visualize(rom=args.rom, rov=args.rov, stride=args.stride, dia=args.dia)


if __name__ == "__main__":
    main()
