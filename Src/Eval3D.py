import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage import measure

from Utils.Geometry import index, perspective
from Utils.Geometry import create_grid, generate_gaussian_grid
from Utils.Geometry import save_obj_mesh_with_color, save_obj_distance_between_gt_wo_acc
from Utils.Utils import load_model

from Dataset.ScenaroIFS import ScenaroIFS
from Model.IFSNet import IFSNet


def schedule_scanline_zigzag(smplx, bmin, sos, rom, rov, dia, stride):
    tree = cKDTree(smplx.vertices)

    # 8是余量
    margin = np.linalg.norm(sos) * (8 + rov / 2)

    indices, centers = [], []
    candates = np.mgrid[
        0 : rom - rov + 1 : stride, 0 : rom - rov + 1 : stride, 0 : rom - rov + 1 : stride
    ].reshape(3, -1)

    deltas = np.mgrid[0:dia, 0:dia, 0:dia].reshape(3, -1)

    for i in range(candates.shape[1]):
        for j in range(deltas.shape[1]):
            index = candates[:, i]
            delta = deltas[:, j]

            # i.e. bmin + sos / 2 + sos * index + sos * (rov - 1) / 2 + sos * delta / dia
            center = bmin + sos * index + sos * rov / 2 + sos * delta / dia

            dist, _ = tree.query(center)
            if dist > margin:
                continue

            indices.append(index * dia + delta)
            centers.append(center)

    return indices, centers


def scan_voxels(device, net, mvfeature, projection, indices, centers, sos, rom, rov, dia):
    labels = np.zeros((rom * dia, rom * dia, rom * dia))
    weights = np.ones((rom * dia, rom * dia, rom * dia)) * 1e-9

    beta = generate_gaussian_grid(rov, mean=[0, 0, 0], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    template = create_grid(rov, rov, rov, -sos * (rov - 1) / 2, sos * (rov - 1) / 2)[0].reshape(3, rov**3)

    for idx in tqdm(range(len(centers))):
        center = centers[idx]

        voxel = template.copy() + center[:, np.newaxis]
        voxel = torch.Tensor(voxel).unsqueeze(0).to(device)

        label = net.query(mvfeature, voxel, projection, rov)[0]
        label = label.view(rov, rov, rov).cpu().numpy()

        i, j, k = indices[idx]
        labels[i : i + rov * dia : dia, j : j + rov * dia : dia, k : k + rov * dia : dia] += label * beta
        weights[i : i + rov * dia : dia, j : j + rov * dia : dia, k : k + rov * dia : dia] += beta

    return labels / weights


def compute_i2c(bmin, sos, dia):
    projection = np.eye(3)
    projection[0, 0] = sos[0] / dia
    projection[1, 1] = sos[1] / dia
    projection[2, 2] = sos[2] / dia
    return projection, (bmin + sos / 2)[:, np.newaxis]


def generate(device, net, data, rom, rov, dia, stride):
    images = [image.to(device).unsqueeze(0) for image in data["images"]]
    mvfeature = net.encode(images)

    bmin, bmax = data["bmin"], data["bmax"]
    size = bmax - bmin

    # sos: size of sample
    sos = size / rom

    indices, centers = schedule_scanline_zigzag(data["smplx"], bmin, sos, rom, rov, dia, stride)

    projection = data["projection"].to(device)
    labels = scan_voxels(device, net, mvfeature, projection, indices, centers, sos, rom, rov, dia)

    verts, faces, _, _ = measure.marching_cubes(labels, 0.5)
    scale, translation = compute_i2c(bmin, sos, dia)
    verts = np.matmul(scale, verts.T) + translation

    # todo: 这几行代码我没管
    verts_tensor = torch.from_numpy(verts).unsqueeze(0).to(device=device).float()
    uv = perspective(verts_tensor, projection[:1])
    color = index(data["images"][0].to(device=device).unsqueeze(0), uv).detach().cpu().numpy()[0].T
    color = color * 0.5 + 0.5

    return verts.T, faces, color


if __name__ == "__main__":
    roi = 1536
    dia = 1
    rom = 768
    rov = 64

    path = "/root/leinyu/data/thuman2/ft_local/dataset grid_samples_32_24"
    gt_path = "/root/leinyu/data/thuman2/ft_local/dataset/mesh/"
    results_path = "../ifs_results_path/0721_48_16_wheel_t1536_e1536"

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # set cuda
    net = IFSNet(None, nov=8, fusion='wheel', rov=64).to(device=device)

    load_model(net, "../checkpoints/[25-07-22-19-24-34] ifs 0721_48_16_wheel_1536img refactoring version-e19.pth")

    dataset = ScenaroIFS(
        path=path, split="test", ratio=IFSNet.ratio(), roi=roi, nov=8, yaw_list=[0, 6, 12, 18, 24, 30, 36, 42]
    )

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}", exist_ok=True)
    dis_path = os.path.join(results_path, "chamfer_distance.txt")

    with torch.no_grad():
        net.eval()

        for idx in range(len(dataset)):
            data = dataset[idx]
            verts, faces, color = generate(device, net, data, rom, rov, dia, rov // 2)

            subject = data["subject"]
            save_obj_distance_between_gt_wo_acc(dis_path, verts, gt_path, subject)

            save_path = f"{results_path}/inference_eval_{subject}.obj"
            save_obj_mesh_with_color(save_path, verts, faces, color)
            print("saving to " + save_path)
