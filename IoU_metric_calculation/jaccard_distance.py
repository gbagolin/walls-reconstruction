import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iou_net = []
iou_original = []


def print_jaccard_distance(mat_net, mat_original, mat_francesco):
    """Prints the Jaccard distance 

    Args:
        mat_net ([str]): 
        mat_francesco ([str]): 
    """
    wall_net = sio.loadmat(mat_net)
    wall_francesco = sio.loadmat(mat_francesco)
    wall_original = sio.loadmat(mat_original)

    decimal_digits = 1
    x_net = (wall_net['x'].ravel() / 1000.0).round(decimal_digits)
    z_net = (wall_net['z'].ravel() / 1000.0).round(decimal_digits)

    # fig, axs = plt.subplots(1, 3)

    # axs[0].scatter(x_net, z_net)
    # axs[0].set_title("Net walls")
    # # axs[0,0].set_xticks(np.arange(start=np.min(x_net), stop=np.max(x_net), step=1))

    x_francesco = (wall_francesco['x'].ravel() / 1000.0).round(decimal_digits)
    z_francesco = (wall_francesco['y'].ravel() / 1000.0).round(decimal_digits)

    # axs[1].scatter(x_francesco, z_francesco)
    # axs[1].set_title("Francesco walls")
    # # axs[0,1].set_xticks(np.arange(start=np.min(x_francesco), stop=np.max(x_francesco), step=1))

    x_original = (wall_original['x'].ravel() / 1000.0).round(decimal_digits)
    z_original = (wall_original['z'].ravel() / 1000.0).round(decimal_digits)

    # axs[2].scatter(x_original, z_original)
    # axs[2].set_title("Original walls")
    # # axs[1,0].set_xticks(np.arange(start=np.min(x_original), stop=np.max(x_original), step=1))
    # plt.show()

    set_net = set(zip(x_net, z_net))
    set_original = set(zip(x_original, z_original))
    set_francesco = set(zip(x_francesco, z_francesco))

    intersection_num_elements = len(set_net.intersection(set_francesco))
    union_num_elements = len(set_net.union(set_francesco))

    print(f"# intersection with net elements {intersection_num_elements}")
    print(f"# union with net elements {union_num_elements}")
    print(
        f"IoU of net method: {intersection_num_elements / union_num_elements}")

    iou_net.append(intersection_num_elements / union_num_elements)

    intersection_num_elements = len(set_original.intersection(set_francesco))
    union_num_elements = len(set_original.union(set_francesco))

    print(f"# intersection with original elements {intersection_num_elements}")
    print(f"# union with original elements {union_num_elements}")
    print(
        f"IoU of original method: {intersection_num_elements / union_num_elements}")

    iou_original.append(intersection_num_elements / union_num_elements)


if __name__ == "__main__":
    dataset_list = [
        '2azQ1b91cZZ_level_0',
        '8194nk5LbLH_level_0',
        'EU6Fwq7SyZv_level_0',
        'QUCTc6BB5sX_level_0',
        'TbHJrupSAjP_level_0',
        'X7HyMhZNoso_level_0',
        'x8F5xyUWy9e_level_0',
        'zsNo4HB9uLZ_level_0'
    ]
    for file in dataset_list:
        mat_original = f'/home/giovanni/Scrivania/progetto_deep_learning/habitat/src/habitat_reconstruction/walls_original_04052021/{file}_wall.mat'
        mat_net = f'/home/giovanni/Scrivania/progetto_deep_learning/habitat/src/habitat_reconstruction/walls_net_04052021_pspnet/{file}_wall.mat'
        mat_francesco = f'/home/giovanni/Scrivania/progetto_deep_learning/walls_francesco/mat/{file}_wall.mat'
        print(f"House name: {file}")
        print_jaccard_distance(mat_net, mat_original, mat_francesco)

    data = {
        'House_name': dataset_list, 
        'IoU_pspnet': iou_net, 
        'IoU_original': iou_original
    }

    df = pd.DataFrame(data, columns= ['House_name', 'IoU_pspnet', 'IoU_original'])
    print(df)
    df.to_csv ('/home/giovanni/Scrivania/IoU_pspnet.csv', index = False, header=True)