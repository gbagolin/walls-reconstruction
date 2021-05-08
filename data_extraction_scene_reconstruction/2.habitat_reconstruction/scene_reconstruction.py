from __future__ import print_function
import lib

import yaml

import lib.utils.io as io_local
import lib.utils.filetool as ft
import lib.utils.transformation as tf

import os
import numpy as np
import time
import open3d as o3d
from os.path import expanduser
import scipy.io as sio
import parse
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import json

import copy


# floor_color = np.asarray([243, 246, 208], dtype=np.uint8)
# wall_color = np.asarray([148, 232, 164], dtype=np.uint8)


# function to extract poses from rgb file name for AI2THOR
def extract_poses(rgb_files):
    pose_list = []
    for rgb_filename in rgb_files:
        transform = np.eye(4)
        format_string = "{08.4f}_{08.4f}_{08.4f}_{08.4f}_{08.4f}_{08.4f}.png"
        parsed = parse.parse(format_string, rgb_filename)
        (x, y, z, ang_x, ang_y, ang_z) = parsed  # all in str
        r = R.from_euler(
            "zyx", [float(ang_z), float(ang_y), float(ang_x)], degrees=True
        )
        rot_mtx = r.as_matrix()
        transform[:3, :3] = rot_mtx
        transform[:3, 3] = np.asarray([float(x), -float(y), -float(z)])
        pose_list.append(transform)
    return pose_list


# function to get the image indices by indicating the index
def get_semantic_index(semantic_image, color):
    indices = np.where(semantic_image[:, :] == color)
    return indices


# main function to reconstruct for a scene
def reconstruct_once(scene_name):
    print("Reconstructing {}".format(scene_name))
    # prepare the result pcd folder
    pcd_folder = Path("pcd_folder")
    if not os.path.exists(pcd_folder):
        os.makedirs(pcd_folder)

    result_file_pcd = os.path.join(pcd_folder, "{}.ply".format(scene_name))

    if ft.file_exist(result_file_pcd):
        # load data
        print("Ply exists, do not reconstruct the scene ... ")
        pcd = o3d.io.read_point_cloud(result_file_pcd)
        # o3d.visualization.draw_geometries([pcd])
        return pcd

    depth_folder = os.path.join(data_path, scene_name, "DEPTH")
    semantic_folder = os.path.join(data_path, scene_name, "SEM_CLASS")
    rgb_folder = os.path.join(data_path, scene_name, "RGB")
    rgb_list = ft.grab_files(rgb_folder, fullpath=False, extra="*.png")

    cam_poses = extract_poses(rgb_list)
    num_poses = len(cam_poses)
    # num_poses = 1024
    current_index = 0
    # initialize the 3d volume
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.03,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    )  # 4.0 / 512.0
    print("There are in total {:d} frames to integrate".format(num_poses))
    list_to_visualise = []
    while current_index < num_poses:
        print(
            "Integrating {:d} frames out of {:d}".format(current_index, len(cam_poses))
        )
        depth_path = os.path.join(
            depth_folder, rgb_list[current_index].replace("png", "npy")
        )
        if ExtractFloor:
            # read the semantic_class and set zeros the depth that are non-floor
            semantic_file = os.path.join(
                semantic_folder, rgb_list[current_index].replace("png", "npy")
            )
            semantic_image = np.load(semantic_file)
            indices_floor = get_semantic_index(semantic_image, floor_color)
            indices_wall = get_semantic_index(semantic_image, wall_color)
            depth_raw = np.load(depth_path)
            depth_data = np.zeros_like(depth_raw)
            depth_data[indices_floor] = depth_raw[indices_floor] * 1000
            depth_data[indices_wall] = depth_raw[indices_wall] * 1000
            depth = o3d.geometry.Image(depth_data)
        else:
            depth_data = np.load(depth_path)
            print(type(depth_data[0, 0]))
            depth = o3d.geometry.Image((depth_data * 1000.0))

        rgb_path = os.path.join(rgb_folder, rgb_list[current_index])
        color = o3d.io.read_image(rgb_path)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        current_cam_pose = cam_poses[current_index]

        volume.integrate(rgbd, intrinsic, np.linalg.inv(current_cam_pose))

        current_index = current_index + 1

    pcd_scene = volume.extract_point_cloud()

    o3d.io.write_point_cloud(result_file_pcd, pcd_scene)
    print("Finished writing one ply file")

    list_to_visualise.append(pcd_scene)

    world_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    list_to_visualise.append(world_mesh_frame)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    # mesh_frame.transform(current_cam_pose)
    # list_to_visualise.append(mesh_frame)
    o3d.visualization.draw_geometries(list_to_visualise)

    return pcd_scene


# main function to extract walls from a reconstructed scene
def extract_wall(pcd, name):
    wall_folder = Path("walls")
    if not os.path.exists(wall_folder):
        os.makedirs(wall_folder)
    wall_file_mat = os.path.join(wall_folder, "{}_wall.mat".format(name))

    if ft.file_exist(wall_file_mat):
        print("Wall file exists, skip ... ")
        data = sio.loadmat(wall_file_mat)
        x = data["x"] / 1000.0
        z = data["z"] / 1000.0
        floor_level = data["floor_level"] / 1000.0
        wall_zyx = np.zeros((x.shape[0], 3))
        wall_zyx[:, 0] = x.ravel()
        wall_zyx[:, 1] = floor_level
        wall_zyx[:, 2] = z.ravel()

        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(wall_zyx)
        wall_pcd.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([wall_pcd])

        list_to_show = []
        pcd2 = copy.deepcopy(pcd)
        list_to_show.append(pcd2.paint_uniform_color([0, 1, 0]))
        list_to_show.append(wall_pcd)
        o3d.visualization.draw_geometries(list_to_show)

        return pcd

    print("extract walls...")
    print("step 1: downsample first")
    down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
    print("step 1: flip y up ")
    xyz = np.asarray(down_pcd.points)
    xyz[:, 1] = -xyz[:, 1]

    print("step 3: extract the range of floor area from the floor")
    floor_level = np.min(xyz[:, 1])
    floor_indices = np.where(xyz[:, 1] <= (floor_level + 0.1))
    floor_points = xyz[floor_indices[0], :]
    max_x = np.max(floor_points[:, 0])
    min_x = np.min(floor_points[:, 0])
    max_z = np.max(floor_points[:, 2])
    min_z = np.min(floor_points[:, 2])

    # draw the boundaries
    points = [
        [max_x, floor_level, max_z],
        [min_x, floor_level, max_z],
        [min_x, floor_level, min_z],
        [max_x, floor_level, min_z],
    ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    valid_points_indices = np.where(
        np.all(
            [
                xyz[:, 0] > min_x,
                xyz[:, 0] < max_x,
                xyz[:, 2] > min_z,
                xyz[:, 2] < max_z,
                xyz[:, 1] > (floor_level + 0.1),
                xyz[:, 1] < (2),
            ],
            axis=0,
        )
    )
    xyz_valid = xyz[valid_points_indices[0], :]
    print(
        "step 4: only using points within the floor area, set all points to the floor level"
    )
    xyz_valid[:, 1] = floor_level
    down_pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    # may be duplicated points, just merge them by down sample
    wall_pcd = down_pcd.voxel_down_sample(voxel_size=0.1)
    wall_xyz = np.asarray(wall_pcd.points)
    wall_pcd.paint_uniform_color([1, 0, 0])
    print("step 4: save to the wall map")
    print("Save wall xy to mat file ... ")
    data_to_save = {
        "x": wall_xyz[:, 0] * 1000.0,
        "z": wall_xyz[:, 2] * 1000.0,
        "floor_level": floor_level * 1000,
    }  # saved in mm
    sio.savemat(wall_file_mat, data_to_save, oned_as="column")
    o3d.visualization.draw_geometries([wall_pcd, line_set])
    return wall_pcd


# main script start from here
# home = expanduser("~")
# project_name = "ActiveVisualSearch"
# proj_path = os.path.join(home, project_name)

# config = io_local.load_yaml(os.path.join(proj_path, "config_ai2thor.yml"))
ExtractFloor = True

# data_path = os.path.join(proj_path, config["datapath"])
data_path = Path("../images_correct_rotation_1m_79fov")

all_room_names = ft.grab_directory(data_path)
all_room_names.sort()
print(all_room_names)

selected_room_names = all_room_names
intrinsic_file = "camera_HABITAT.json"

intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_file)
print("The intrinsic is ", intrinsic.intrinsic_matrix)
start_t = time.time()

global floor_color
global wall_color
for name in selected_room_names:
    with open(data_path / name / "id_sem.json", "r") as ann_id:
        json_ann = json.load(ann_id)
    for key, val in json_ann["lab"].items():
        if val == "floor":
            floor_color = int(key)
        if val == "wall":
            wall_color = int(key)

    print("Current sequence is {}".format(name))
    pcd = reconstruct_once(name)
    wall_pcd = extract_wall(pcd, name)

end_t = time.time()
elapsed_t = (end_t - start_t) / float(3600)

print("Program finished cleanly with elapsed time %.2f " % (elapsed_t))
