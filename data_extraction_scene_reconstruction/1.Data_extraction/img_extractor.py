import numpy as np
from data_extractor import ImageExtractor
from PIL import Image
from pathlib import Path
import habitat_sim

# For viewing the extractor output
from habitat_sim import registry as registry
from habitat_sim.utils.data.pose_extractor import PoseExtractor
from tqdm import tqdm
import json


@registry.register_pose_extractor(name="rot30")
class Rot30(PoseExtractor):
    def extract_poses(self, labels, view, fp):
        height, width = view.shape
        poses = []
        for row in range(height):
            for col in range(width):
                if self._valid_point(row, col, view):
                    rots = np.linspace(0, np.pi * 2, 13)[:12]
                    rots[0] = rots[0] - 1e-3
                    for rad in rots:
                        vec = [1 * np.cos(rad), 1 * np.sin(rad)]
                        point = [row, col]
                        point_of_interest = (row + vec[0], col + vec[1])
                        pose = (point, point_of_interest, labels, fp)
                        poses.append(pose)
        return poses


test_set_file = "test.txt"
with open(test_set_file, "r") as f:
    lines = f.readlines()
test_sets = [x.strip("\n") for x in lines]
test_sets = ["2azQ1b91cZZ"]
# scene='1LXtFkjw3qL'

for scene in test_sets:
    # scene_filepath = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    scene_filepath = (
        f"/media/DATA/DATASET/habitat-lab/data/scene_datasets/mp3d/{scene}/{scene}.glb"
    )
    extractor = ImageExtractor(
        scene_filepath,
        img_size=(480, 640),
        output=["rgba", "depth", "semantic"],
        pose_extractor_name="rot30",
        meters_per_pixel=1,
        shuffle=False,
        ypos=0.14,
    )
    # if len(extractor.sim.semantic_scene.levels) > 2:
    #   extractor.close()
    #   continue

    # Use the list of train outputs instead of the default, which is the full list
    # of outputs (test + train)
    extractor.set_mode("full")

    # Index in to the extractor like a normal python list

    # Or use slicing
    samples = extractor

    # display_sample(sample)

    out_folder = Path(f"0.25m/{scene}_level_0/")
    import os

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(out_folder / "RGB", exist_ok=True)
    os.makedirs(out_folder / "SEM_ID", exist_ok=True)
    os.makedirs(out_folder / "SEM_CLASS", exist_ok=True)
    os.makedirs(out_folder / "DEPTH", exist_ok=True)
    # Close the extractor so we can instantiate another one later
    # (see close method for detailed explanation)
    tvd = extractor.tdv_fp_ref_triples[0][0].topdown_view
    Im = Image.fromarray((tvd * 255).astype(np.uint8))
    Im.save(out_folder / "tvd.png")

    # object_id_to_class_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in extractor.sim.semantic_scene.objects}
    object_id_to_class = extractor.instance_id_to_name
    lab_to_id = {val: key for key, val in enumerate(set(object_id_to_class.values()))}
    inv_lab_to_id = {val: key for key, val in lab_to_id.items()}
    id_to_class = np.vectorize(lambda x: lab_to_id[object_id_to_class[x]])

    with open(out_folder / "id_sem.json", "w") as f:
        json.dump({"lab": inv_lab_to_id}, f)

    for sample in tqdm(samples):
        img = sample["rgba"][:, :, :3]
        depth = sample["depth"]
        semantic_id = sample["semantic"]
        semantic_class = id_to_class(semantic_id)
        pos = sample["pos"]
        pos[1] += 0.88
        rot = habitat_sim.utils.common.quat_rotate_vector(
            sample["rot"], habitat_sim.geo.FRONT
        )
        rot = [0, (np.rad2deg(-np.arctan2(rot[0], rot[2])) - 180) % 360, 0]
        p = "_".join(["%08.4f" % x for x in pos])
        r = "_".join(["%08.4f" % x for x in rot])
        Image.fromarray(img).save(out_folder / "RGB" / f"{p}_{r}.png")
        np.save(out_folder / "DEPTH" / f"{p}_{r}.npy", depth)
        np.save(out_folder / "SEM_ID" / f"{p}_{r}.npy", semantic_id)
        np.save(out_folder / "SEM_CLASS" / f"{p}_{r}.npy", semantic_class)

    extractor.close()
