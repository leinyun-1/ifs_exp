import numpy as np
import os
import trimesh
import time
from tqdm import tqdm
from pysdf import SDF

from Utils.Geometry import create_grid

"""
pip install pysdf
pip install embreex
"""


def make_ifs_grid_samples_dataset(dataset_path, output_path, num_centers, som, rov):
    # todo: 430是怎么定的？
    subs = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    existing_subs = os.listdir(output_path)
    # subs.remove('mesh_ori')
    # subs.remove('normalize.pkl')
    subs =list(set(subs) - set(existing_subs))


    rom = som * rov

    bmin, bmax = np.array([-1, -1, -1]), np.array([1, 1, 1])
    size = bmax - bmin
    sos = size / rom

    template = create_grid(rov, rov, rov, -sos * (rov - 1) / 2, sos * (rov - 1) / 2)[0]
    template = template.reshape(3, -1).transpose()

    for sub in tqdm(subs):
        mesh_path = os.path.join(dataset_path, sub, sub + ".obj")
        mesh = trimesh.load(mesh_path)

        save_sub_path = os.path.join(output_path, sub)
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)

        f = SDF(mesh.vertices, mesh.faces)

        # todo: 为什么不干脆多一倍数量num_centers * 2
        # todo: 数据集总数为200，但只用了前100，到底是哪部分
        centers = trimesh.sample.sample_surface(mesh, num_centers * 2)[0]
        np.random.shuffle(centers)
        centers_near = centers + np.random.normal(scale=0.02, size=(num_centers * 2, 3))

        centers = trimesh.sample.sample_surface(mesh, num_centers)[0]
        np.random.shuffle(centers)
        centers_near_1 = centers + np.random.normal(scale=0.05, size=(num_centers, 3))

        centers = trimesh.sample.sample_surface(mesh, num_centers)[0]
        np.random.shuffle(centers)
        empty_centers = centers + np.random.normal(scale=0.2, size=(num_centers, 3))

        centers = np.concatenate([centers_near, centers_near_1, empty_centers], axis=0)

        patch_num = 0
        for center in tqdm(centers):
            samples = center + template

            occs = mesh.contains(samples)
            sdf = f(samples)

            save_patch_path = os.path.join(save_sub_path, str(patch_num) + ".npz")
            np.savez_compressed(save_patch_path, samples=samples, occs=occs, sdf=sdf, center=center)

            patch_num += 1


def convert_grid_samples(dataset_path, output_path):
    subs = os.listdir(dataset_path)[:430]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for sub in tqdm(subs):
        for idx in range(200):
            path = os.path.join(dataset_path, sub, str(idx) + ".npz")

            data = np.load(path, allow_pickle=True)

            samples = data["array"][:, :3]
            center = (np.max(samples, axis=0) + np.min(samples, axis=0)) / 2.0

            occs = data["array"][:, 3]
            sdf = data["array"][:, 4]

            # sdf[sdf > 0.2] = 0.2
            # sdf[sdf < -0.2] = -0.2

            save_sub_path = os.path.join(output_path, sub)
            if not os.path.exists(save_sub_path):
                os.makedirs(save_sub_path)

            save_patch_path = os.path.join(save_sub_path, str(idx) + ".npz")
            np.savez_compressed(save_patch_path, samples=samples, occs=occs, sdf=sdf, center=center)


def test_grid_sample_dataset(dataset_path, subs, voxel, output_path):
    for sub in subs:
        path = os.path.join(dataset_path, sub)
        voxes = os.listdir(path)[:20]
        samples = []
        occs = []
        for vox in voxes:
            vox_path = os.path.join(path,vox)


            start = time.time()
            data = np.load(vox_path, allow_pickle=True)
            print("used time: ", time.time() - start)

            samples.append(data["samples"])
            occs.append(data["occs"])
        
        samples = np.concatenate(samples)
        occs = np.concatenate(occs)

        save_samples_truncted_prob(output_path, samples, occs)

def test_pifu_sample(subject,sigma=0.02,num_sample_inout=8192,output_path='../bbbb.ply'):
    file_name = os.path.join('/home/leinyun/dataset/mesh', subject, subject+'.obj')
    mesh = trimesh.load(file_name)

    radius_list = [sigma / 3, sigma, sigma * 2]
    surface_points = np.zeros((6 * num_sample_inout, 3))
    sample_points = np.zeros((6 * num_sample_inout, 3))
    for i in range(3):
        d = 2 * num_sample_inout
        surface_points[i * d:(i + 1) * d, :], _ = trimesh.sample.sample_surface(mesh,2 * num_sample_inout)
        sample_points[i * d:(i + 1) * d, :] = surface_points[i * d:(i + 1) * d, :] + np.random.normal(
            scale=radius_list[i], size=(2 * num_sample_inout, 3))

    # add random points within image space
    b_min = np.array([-1,-1,-1])
    b_max = np.array([1,1,1])
    length = b_max - b_min
    random_points = np.random.rand(num_sample_inout, 3) * length + b_min
    sample_points = np.concatenate([sample_points, random_points], 0)
    np.random.shuffle(sample_points)
    sample_points = sample_points[:8192]
    occs = mesh.contains(sample_points) #这句话卡住了
    save_samples_truncted_prob(output_path, sample_points, occs)



def save_samples_truncted_prob(fname, points, prob):
    """
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    """
    r = (prob).reshape([-1, 1]) * 255
    g = (1 - prob).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(
        fname,
        to_save,
        fmt="%.6f %.6f %.6f %d %d %d",
        comments="",
        header=(
            "ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header"
        ).format(points.shape[0]),
    )




if __name__ == "__main__":
    # 转化原数据集，转化完之后，删除该函数代码，不做保留
    """convert_grid_samples(
        "I:/yulei_MVS/yulei_MVS/THuman2/Thuman2.1_grid_samples_2", "../Thuman2.1_grid_samples_convert"
    )"""

    # make_ifs_grid_samples_dataset(
    #     "/home/leinyun/dataset/mesh/", "../grid_samples_64_12", num_centers=25, som=12, rov=64
    # )
    # test_grid_sample_dataset(
    #     "../grid_samples_64_12", ["0421"], "30.npz", "../aaaa.ply"
    # )
    test_pifu_sample('0421')
