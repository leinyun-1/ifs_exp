import os
import torch
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def cross_3d(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], b[0] * a[2] - a[0] * b[2], a[0] * b[1] - b[0] * a[1]])


def index(feat, uv):
    """
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    size = torch.FloatTensor([feat.shape[-1], feat.shape[-2]]).to(uv.device)

    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    if size != None:
        uv = (uv - size / 2) / (size / 2)
        # print("index max and min",torch.max(uv),torch.min(uv))

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def perspective(points, calibrations):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """

    if isinstance(points, torch.Tensor):
        B, _, N = points.shape
        device = points.device
        points = torch.cat([points, torch.ones((B, 1, N), device=device)], dim=1)
        points = calibrations @ points
        points[:, :2, :] /= points[:, 2:, :]
        return points[:, :2, :]
    elif isinstance(points, np.ndarray):
        _, N = points.shape
        points = np.vstack((points, np.ones((1, N))))
        points = calibrations @ points
        points[:2, :] /= points[2:, :]
        return points[:2, :]
    else:
        raise TypeError("Unsupported data type. Expected NumPy array or PyTorch tensor.")


def perspective4(points, calibrations):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [BxVx3x4] Tensor of projection matrix
    :return: xy: [BxVx2xN] Tensor of xy coordinates in the image plane
    """
    points = points.unsqueeze(1)
    B, _, _, N = points.shape

    device = points.device
    points = torch.cat([points, torch.ones((B, 1, 1, N), device=device)], dim=2)
    points = calibrations @ points
    points[:, :, :2, :] /= points[:, :, 2:, :]
    return points[:, :, :2, :]


def create_grid(resX, resY, resZ, bmin=np.array([0, 0, 0]), bmax=np.array([1, 1, 1]), transform=None):
    """
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param bmin: vec3 (x_min, y_min, z_min) bounding box corner
    :param bmax: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    """
    # print('start creating grid')
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = np.float32(coords)  # abl at 2024-0616 处理mgrid占用内存过大问题
    coords = coords.reshape(3, -1)

    coords_matrix = np.eye(4)
    size = bmax - bmin
    coords_matrix[0, 0] = size[0] / (resX - 1)
    coords_matrix[1, 1] = size[1] / (resY - 1)
    coords_matrix[2, 2] = size[2] / (resZ - 1)
    coords_matrix[0:3, 3] = bmin

    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)

    coords = coords.reshape(3, resX, resY, resZ)
    # print('creating_grid_done')
    return coords, coords_matrix


def generate_gaussian_grid(n, mean, cov):
    import numpy as np
    from scipy.stats import multivariate_normal

    # 生成网格坐标
    x, y, z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # 计算三维高斯分布的概率密度
    gaussian_dist = multivariate_normal(mean=mean, cov=cov)
    density = gaussian_dist.pdf(grid_points)

    # 归一化概率密度
    density /= density.sum()

    # 将一维概率密度转换为三维网格
    grid = density.reshape((n, n, n))
    grid_max = np.max(grid)
    grid_min = np.min(grid)
    grid = grid - grid_min
    grid = grid / (grid_max - grid_min)

    return grid


def save_obj_mesh_with_color(mesh_path, verts, faces, colors, reverse=False):
    file = open(mesh_path, "w")

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        if reverse:
            file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
        else:
            file.write("f %d %d %d\n" % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def array_to_array(recon, gt):
    neigh = NearestNeighbors(n_neighbors=1).fit(gt)
    dis, _ = neigh.kneighbors(recon, return_distance=True)
    neigh_1 = NearestNeighbors(n_neighbors=1).fit(recon)
    dis_1, _ = neigh_1.kneighbors(gt, return_distance=True)
    return dis.ravel().mean(), dis.ravel().mean() + dis_1.ravel().mean()


def save_obj_distance_between_gt_wo_acc(dis, verts, gt_path, name):
    gt = trimesh.load(os.path.join(gt_path, name, name + ".obj"))
    p2s, chamfer = array_to_array(verts, gt.vertices)

    with open(dis, "a") as f:
        f.write("%s: %.6f ,%.6f\n" % (name, p2s, chamfer))
