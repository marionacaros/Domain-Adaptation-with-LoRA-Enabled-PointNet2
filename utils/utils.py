import glob
import numpy as np
import torch
from progressbar import progressbar
import os
import pickle
import laspy
from sklearn.neighbors import KDTree, NearestNeighbors
import random
from scipy.spatial import cKDTree
import math


# -----------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Preprocessing ----------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

def preprocessing(pc, max_h=200.0, n_points=8000, max_points=400000):
    """
    Perform preprocessing on a given point cloud.

    Steps:
    1. Remove outliers and points with negative z-coordinates.
    2. Calculate and clip the Normalized Difference Vegetation Index (NDVI) within the range [-1, 1].
    3. Check the number of non-ground points and augment the dataset by adding more ground points if needed.
    4. If there are already enough non-ground points, remove the ground points.

    Parameters:
    - pc (numpy.ndarray): Input point cloud data.

    Returns:
    - pc numpy.ndarray: Processed point cloud after the specified preprocessing steps.
    """
    # Remove outliers (points above max_z)
    pc = pc[pc[:, 10] <= max_h]
    pc[:, 10] = pc[:, 10]/max_h
    
    # Remove points z < 0
    pc = pc[pc[:, 10] >= 0]

    # add NDVI
    pc[:, 9] = get_ndvi(pc[:, 8], pc[:, 5])  # range [-1, 1]
    pc[:, 9] = np.clip(pc[:, 9], -1, 1.0)

    # Check if num. points different from ground < n_points
    len_pc = pc[pc[:, 3] != 2].shape[0]
    if 100 < len_pc < n_points:
        # if there are few points we will keep ground points
        len_needed_p = n_points - len_pc

        # Get indices of ground points
        labels = pc[:, 3]
        i_terrain = np.where(labels == 2.0)[0]

        # if we have enough points of ground to cover missed points
        if len_needed_p < len(i_terrain):
            needed_i = random.sample(list(i_terrain), k=len_needed_p)
        else:
            needed_i = i_terrain

        # store points needed
        points_needed_terrain = pc[needed_i, :]

        # remove terrain points
        pc = pc[pc[:, 3] != 2, :]

        # store only needed terrain points
        pc = np.concatenate((pc, points_needed_terrain), axis=0)

    # if enough points, remove ground
    elif len_pc >= n_points:
        pc = pc[pc[:, 3] != 2, :]

    # max number of points is MAX_N_PTS 
    if len_pc > max_points:
        print(f'PC with more than 400K pts. shape= {pc.shape}')
        # Reduce number of points
        # sampled_indices = np.random.choice(pc.shape[0], max_points, replace=False)
        # pc = pc[sampled_indices,:]

    return pc


# -----------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Post-processing ----------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

def refine(xyz, preds, clase_objetivo=2, iteraciones=1, altura_exclusion=5.0, radio_asignacion=1.0):
    """
    Refinar la clasificación de una clase específica en una nube de puntos, evitando modificar puntos en el suelo.

    xyz: np.array de shape (N, 4) con coordenadas XYZ + etiqueta inicial
    preds: np.array de shape (N,) con etiquetas de clase.
    clase_objetivo: La clase a corregir (ej. 2 para torres eléctricas, 1 para líneas, etc.).
    iteraciones: número de veces que se repite la asignación.
    altura_exclusion: Altura (en metros) bajo la cual los puntos no serán modificados.
    radio_asignacion

    Retorna:
    - labels: etiquetas refinadas
    """
    for _ in range(iteraciones):
        # Filtrar puntos de la clase objetivo
        mask_clase_objetivo = preds == clase_objetivo
        puntos_clase = xyz[mask_clase_objetivo]

        if puntos_clase.shape[0] == 0:
            return preds  # No hay puntos de la clase objetivo para procesar

        # Construir un KDTree con los puntos de la clase objetivo detectados
        kdtree = cKDTree(puntos_clase)

        # Filtrar puntos que cumplen con la altura mínima , no pertenecen a la clase objetivo y son de la clase default
        mask_altura = xyz[:, 2] > altura_exclusion
        mask_no_objetivo = preds != clase_objetivo
        mask_default_pts = xyz[:,3] != 12.
        mask_candidatos = mask_altura & mask_no_objetivo & mask_default_pts

        # Buscar vecinos en el radio permitido para los puntos candidatos
        puntos_candidatos = xyz[mask_candidatos]
        vecinos_idx = kdtree.query_ball_point(puntos_candidatos, r=radio_asignacion)

        # Reasignar etiquetas para los puntos con vecinos en la clase objetivo
        indices_candidatos = np.where(mask_candidatos)[0]
        for i, vecinos in zip(indices_candidatos, vecinos_idx):
            if len(vecinos) > 0:
                preds[i] = clase_objetivo

    return preds


def store_xyz(xyz, xyz_tile, preds, ids, ids_ini, n_o_points, knn_smpl, plot_objects, file_name, output_dir, store_pts_class=False):

    # one hot encoding
    preds_1hot = torch.nn.functional.one_hot(preds, num_classes=5).to(torch.int64)  # [n_points, classes]

    # check if any of the classes different from 3 has < MIN_POINTS_WINDTURBINE points (NOISE)
    # 1. Sum the one-hot encoded tensor to get the number of points per class
    points_per_class = preds_1hot.sum(dim=0).numpy()

    # if there are few points of wind turbine and class tower has been detected it has to be a tower.
    # if 0 < points_per_class[4] <= MIN_POINTS_WINDTURBINE and points_per_class[1]>10:
    #     # 2. Identify classes with few points, excluding class 3
    #     # Identify classes where the number of points is above 0 and below MAX, excluding class 3 and class 0
    #     # classes_to_replace = (points_per_class > 0) & (points_per_class <= MIN_POINTS_OBJ) & (torch.arange(points_per_class.size(0)) == 4) & (torch.arange(points_per_class.size(0)) != 0)
    #     # Create a mask for rows that belong to the classes to be replaced
    #     rows_to_replace = preds_1hot[:, 4] > 0
    #     # 3. For rows belonging to classes with less than MIN_POINTS_OBJ points, set their encoding to class 3
    #     preds_1hot[rows_to_replace] = 0  # Reset these rows
    #     preds_1hot[rows_to_replace, 1] = 1  # Assign them to class tower (1)
    #     # one-hot --> labels
    #     preds = torch.argmax(preds_1hot, dim=1)

    if store_pts_class:
        if points_per_class[4] > MIN_POINTS_WINDTURBINE:
            print(f'Points class 4 (windturbine):{points_per_class[4]}')
            print(f'Points class 1 (tower):{points_per_class[1]}')

        if points_per_class[4] > 0:
            # store number of points per wind turbine
            with open('src/files/points_windturbines.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([points_per_class[4]])
        if points_per_class[1] > 0:
            # store number of points per wind turbine
            with open('src/files/points_towers.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([points_per_class[1]])

    preds = preds.numpy()
    set_preds = set(preds)
    
    # check if there are any target classes (e.g. power lines) and save coordinates
    if 2 in set_preds or 1 in set_preds or 4 in set_preds:
        if not knn_smpl:
            preds = preds[:n_o_points]        

        # get indices of target classes
        target_ix = np.where((preds == 2) | (preds == 1) | (preds == 4))[0]

        if len(target_ix) >= MIN_PTS_STORE:
            
            # get ids of target points
            ids_target = ids[target_ix]

            # get indices for xyz
            indices_xyz = find_indices_in_tensors(ids_target, ids_ini[:n_o_points])

            # store id, x, y, z, class in array
            class_arr = preds[target_ix] # target classes (powerlines and wind turbines)
            xyz_obj = np.concatenate((ids_target[:, np.newaxis], xyz[indices_xyz,:], class_arr[:, np.newaxis]), axis=1)
            xyz_tile = np.concatenate((xyz_tile, xyz_obj), axis=0)
            
            if plot_objects:
                indices_full_xyz = find_indices_in_tensors(ids[:n_o_points], ids_ini[:n_o_points])
                plot_two_pointclouds(xyz[indices_full_xyz, :],  #pc[:n_o_points,:].cpu().numpy(),
                                    pc_2=xyz_obj[:, 1:4],
                                    labels=preds,
                                    labels_2=xyz_obj[:, 4],
                                    name=file_name,
                                    path_plot=os.path.join(output_dir, 'plots', 'testing'),
                                    point_size=2)
    return xyz_tile


def find_indices_np(values, target_array):
    """
    target_array: numpy array
    """
    indices = []
    for value in values:
        result = np.where(target_array == value)[0]
        if result.size > 0:
            indices.append(result[0])
        else:
            indices.append(None)
    return indices


def find_indices_in_tensors(ids_target, all_ids):
    """
    Finds the indices of `ids_target` in `all_ids` efficiently.

    Args:
        ids_target (torch.Tensor): Tensor of target IDs.
        all_ids (torch.Tensor): Tensor of all possible IDs.

    Returns:
        torch.Tensor: Indices of `ids_target` in `all_ids`, or -1 if not found.
    """
    # Sort all_ids for efficient searching
    sorted_all_ids, sorted_indices = torch.sort(all_ids)

    # Use searchsorted to find indices
    pos = torch.searchsorted(sorted_all_ids, ids_target)

    # Validate found indices
    valid_mask = (pos < sorted_all_ids.size(0)) & (sorted_all_ids[pos] == ids_target)

    # Initialize result with -1 for missing values
    result_indices = torch.full_like(ids_target, fill_value=-1, dtype=torch.long)
    result_indices[valid_mask] = sorted_indices[pos[valid_mask]]

    return result_indices


def pc_normalize_neg_one(pc):
    """
    Normalize between -1 and 1
    [npoints, dim]
    """
    pc[:, 0] = pc[:, 0] * 2 - 1
    pc[:, 1] = pc[:, 1] * 2 - 1
    return pc


def rm_padding(preds, targets):
    mask = targets != -1
    targets = targets[mask]
    preds = preds[mask]

    return preds, targets, mask


def transform_2d_img_to_point_cloud(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i] / 2) / img_array.shape[i]
    return indices.astype(np.float32)


def save_checkpoint_segmen_model(name, task, epoch, epochs_since_improvement, base_pointnet, segmen_model, opt_pointnet,
                                 opt_segmen, accuracy, batch_size, learning_rate, number_of_points):
    state = {
        'base_pointnet': base_pointnet.state_dict(),
        'segmen_net': segmen_model.state_dict(),
        'opt_pointnet': opt_pointnet.state_dict(),
        'opt_segmen': opt_segmen.state_dict(),
        'task': task,
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': number_of_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
    }
    filename = 'model_' + name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def save_checkpoint_without_classifier_layer(name, model, optimizer, batch_size, learning_rate, n_points, epoch):
    
    state_dict=model.state_dict()
    # Remove the last layer from the state dictionary
    del state_dict['classifier.weight']
    del state_dict['classifier.bias']
    
    state = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch':epoch
    }
    filename = name + '.pt'
    torch.save(state, filename)


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                    learning_rate, n_points, weighing_method=None, weights=[], label_smoothing=None,
                    color_dropout=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method,
        'weights': weights,
        'label_smoothing': label_smoothing,
        'color_dropour': color_dropout,
    }
    filename = name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def adjust_learning_rate(optimizer, shrink_factor=0.1):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("\nDECAYING learning rate. The new lr is %f" % (optimizer.param_groups[0]['lr'],))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def get_labels(pc):
    """
    Get labels for segmentation

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    2 -> cables
    3 -> low vegetation
    4 -> high vegetation
    """

    segment_labels = pc[:, 3]
    segment_labels[segment_labels == 15] = 100
    segment_labels[segment_labels == 14] = 200
    segment_labels[segment_labels == 3] = 300  # low veg
    segment_labels[segment_labels == 4] = 400  # med veg
    segment_labels[segment_labels == 5] = 400
    segment_labels[segment_labels < 100] = 0
    segment_labels = (segment_labels / 100)

    labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
    return labels


def rotate_point_cloud_z_numpy(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """
    Randomly rotate the point clouds around the Z-axis to augment the dataset.
    Rotation is per shape based along up (Z) direction.
    Use input angle if given.
    
    Input:
      batch_data: BxNx3 tensor, original batch of point clouds
    Return:
      BxNx3 tensor, rotated batch of point clouds
    """
    batch_data = batch_data.to(torch.float32)

    if rotation_angle is None:
        rotation_angle = torch.rand(1).item() * 2 * math.pi

    cosval = math.cos(rotation_angle)
    sinval = math.sin(rotation_angle)
    rotation_matrix = torch.tensor([
        [cosval, sinval, 0],
        [-sinval, cosval, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=batch_data.device)

     # Right-multiply: (B, N, 3) @ (3, 3) -> (B, N, 3)
    rotated_data = batch_data @ rotation_matrix

    return rotated_data



def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: numpy array [b, n_samples, dims]
          label: numpy array [b, n_samples]
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[1])
    np.random.shuffle(idx)
    return data[:, idx, :], labels[:, idx], idx


def shuffle_clusters(data, labels):
    """ Shuffle data and labels.
        Input:
            # segmentation shapes : [b, n_samples, dims, w_len]
            # targets segmen: [b, n_points, w_len]
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[2])
    np.random.shuffle(idx)
    return data[:, :, :, idx], labels[:, :, idx]


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotatePoint(angle, x, y):
    a = np.radians(angle)
    cosa = np.cos(a)
    sina = np.sin(a)
    x_rot = x * cosa - y * sina
    y_rot = x * sina + y * cosa
    return x_rot, y_rot


def get_max(files_path):
    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)
        hag = data_f.HeightAboveGround
        if hag.max() > max_z:
            max_z = hag.max()


def sliding_window_coords(point_cloud, stepSize_x=10, stepSize_y=10, windowSize=[20, 20], min_points=10,
                          show_prints=False):
    """
    Slide a window across the coords of the point cloud to segment objects.

    :param point_cloud:
    :param stepSize_x:
    :param stepSize_y:
    :param windowSize:
    :param min_points:
    :param show_prints:

    :return: (dict towers, dict center_w)

    Example of return:
    For each window we get the center and the points of the tower
    dict center_w = {'0': {0: [2.9919000000227243, 3.0731000006198883]},...}
    dict towers = {'0': {0: array([[4.88606837e+05, 4.88607085e+05, 4.88606880e+05, ...,]])}...}
    """
    i_w = 0
    last_w_i = 0
    towers = {}
    center_w = {}
    point_cloud = np.array(point_cloud)
    x_min, y_min, z_min = point_cloud[0].min(), point_cloud[1].min(), point_cloud[2].min()
    x_max, y_max, z_max = point_cloud[0].max(), point_cloud[1].max(), point_cloud[2].max()

    # if window is larger than actual point cloud it means that in the point cloud there is only one tower
    if windowSize[0] > (x_max - x_min) and windowSize[1] > (y_max - y_min):
        if show_prints:
            print('Window larger than point cloud')
        if point_cloud.shape[1] >= min_points:
            towers[0] = point_cloud
            # get center of window
            center_w[0] = [point_cloud[0].mean(), point_cloud[1].mean()]
            return towers, center_w
        else:
            return None, None
    else:
        for y in range(round(y_min), round(y_max), stepSize_y):
            # check if there are points in this range of y
            bool_w_y = np.logical_and(point_cloud[1] < (y + windowSize[1]), point_cloud[1] > y)
            if not any(bool_w_y):
                continue
            if y + stepSize_y > y_max:
                continue

            for x in range(round(x_min), round(x_max), stepSize_x):
                i_w += 1
                # check points i window
                bool_w_x = np.logical_and(point_cloud[0] < (x + windowSize[0]), point_cloud[0] > x)
                if not any(bool_w_x):
                    continue
                bool_w = np.logical_and(bool_w_x, bool_w_y)
                if not any(bool_w):
                    continue
                # get coords of points in window
                window = point_cloud[:, bool_w]

                if window.shape[1] >= min_points:
                    # if not first item in dict
                    if len(towers) > 0:
                        # if consecutive windows overlap
                        if last_w_i == i_w - 1:  # or last_w_i == i_w - 2:
                            # if more points in new window -> store w, otherwise do not store
                            if window.shape[1] > towers[list(towers)[-1]].shape[1]:
                                towers[list(towers)[-1]] = window
                                center_w[list(center_w)[-1]] = [window[0].mean(), window[1].mean()]

                                last_w_i = i_w
                                if show_prints:
                                    print('Overlap window %i key %i --> %s points' % (
                                        i_w, list(towers)[-1], str(window.shape)))
                        else:
                            towers[len(towers)] = window
                            center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                            last_w_i = i_w
                            if show_prints:
                                print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

                    else:
                        towers[len(towers)] = window
                        center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                        last_w_i = i_w
                        if show_prints:
                            print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

        return towers, center_w


def remove_outliers(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'data_without_outliers')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                    data_f.intensity,
                                    data_f.return_number,
                                    data_f.red,
                                    data_f.green,
                                    data_f.blue
                                    ))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= max_z]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > max_z:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_LAS_data(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'dataset_input_model')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:
                # normalize axes
                data_f.x = (data_f.x - data_f.x.min()) / (data_f.x.max() - data_f.x.min())
                data_f.y = (data_f.y - data_f.y.min()) / (data_f.y.max() - data_f.y.min())
                data_f.HeightAboveGround = data_f.HeightAboveGround / max_z

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= 1]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > 1:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_pickle_data(files_path, max_z=100.0, max_intensity=5000, dir_name=''):
    dir_path = os.path.dirname(files_path)
    path_out_dir = os.path.join(dir_path, dir_name)
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)

    files = glob.glob(os.path.join(files_path, '*.pkl'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f)
        # print(pc.shape)  # [1000,4]
        # try:
        # check file is not empty
        if pc.shape[0] > 0:
            # normalize axes
            pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
            pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
            pc[:, 2] = pc[:, 2] / max_z

            # normalize intensity
            pc[:, 4] = pc[:, 4] / max_intensity
            pc[:, 4] = np.clip(pc[:, 4], 0, max_intensity)

            # return number
            # number of returns

            # normalize color
            pc[:, 7] = pc[:, 7] / 65536.0
            pc[:, 8] = pc[:, 8] / 65536.0
            pc[:, 9] = pc[:, 9] / 65536.0

            # todo add nir and ndv

            # Remove outliers (points above max_z)
            pc = pc[pc[:, 2] <= 1]
            # Remove points z < 0
            pc = pc[pc[:, 2] >= 0]

            if pc[:, 2].max() > 1:
                print('Outliers not removed correctly!!')

            if pc.shape[0] > 0:
                f_path = os.path.join(path_out_dir, fileName)
                with open(f_path + '.pkl', 'wb') as f:
                    pickle.dump(pc, f)
        else:
            print(f'File {fileName} is empty')
        # except Exception as e:
        #     print(f'Error {e} in file {fileName}')


def fps(pc, n_samples):
    """
    points: [N, D]  array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = pc[:, :3]
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return pc[sample_inds]


def get_ndvi(nir, red):
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c


######################################### augmentations #########################################


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


###################################################### samplings #######################################################


def get_sampled_sequence(pc, ids, n_points):
    """
    reshape tensor into sequence of n_points

    :param pc: Input tensor float32 of shape: [points, dims]
    :param ids: point ids int
    :param n_points: Number of points in each sequence
    :return: Sampled sequences pc tensor, ids and indices
    """

    # Shuffle
    indices = torch.randperm(pc.shape[0])

    dims = pc.shape[1]
    pc = pc[indices]
    ids = ids[indices]

    # Get needed points for sampling 
    if pc.shape[0] > n_points and  pc.shape[0] % n_points !=0:
        remain = n_points - (pc.shape[0] % n_points)
        pc3 = torch.cat((pc, pc[:remain, :]), dim=0)
        ids = torch.cat((ids, ids[:remain]), dim=0)
    elif pc.shape[0] < n_points:
        points_needed = n_points - pc.shape[0]
        # duplicate points 
        rdm_list = np.random.randint(0, pc.shape[0], points_needed)
        extra_points = pc[rdm_list, :]
        pc3 = torch.cat([pc, extra_points], dim=0)
        ids = torch.cat((ids, ids[rdm_list]), dim=0)
    else:
        pc3=pc

    try:
        # Add dimension with sequence
        pc3 = torch.unsqueeze(pc3, dim=0)
        pc3 = pc3.view(-1, n_points, dims)
        
    except Exception as e:
        print(f"Error during reshaping: {e}")
        print(f"Shape pc: {pc.shape}")
        print(f"Shape pc3: {pc3.shape}")

    return pc3, ids, indices


def get_sampled_sequence_np(pc, n_points):
    """

    :param pc:
    :param n_points:
    :return:
    """
    # shuffle
    np.random.shuffle(pc)

    # get needed points for sampling partition
    remain = n_points - (pc.shape[0] % n_points)
    if remain != 0:
        # Initialize the resulting tensor
        pad_tensor = pc[:remain, :]
        pc_samp = np.concatenate((pc, pad_tensor), axis=0)
    else:
        pc_samp = pc

    # add dimension with sequence
    pc_samp = np.expand_dims(pc_samp, axis=0)
    pc_samp = np.reshape(pc_samp, (-1, n_points, pc.shape[1]))

    return pc_samp


def knn_exp_prob(ref, query_mask, n_points=8000, num_samples=8, use_z=False):
    """
    Efficient KNN on x,y with exponential probability computation.

    Parameters:
        ref (torch.Tensor): Reference points tensor of shape (n_ref, 2).
        query_mask (torch.Tensor): bool.
        n_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Sampled indices tensor of shape (n_points,).
    """
    # query (torch.Tensor): Query points tensor of shape (n_query, 2).
    # Compute the squared Euclidean distances using torch.cdist
    if use_z:
        ref = ref[:, 2:3] # use z
        query = ref[query_mask]
        distances_sq = torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=1).squeeze(0) ** 2
    else:
        ref = ref[:, :2] # use x and y
        query = ref[query_mask]
        distances_sq = torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=2).squeeze(0) ** 2

    # Apply exponential function
    probs = torch.exp(-5 * distances_sq) #-5
    probs /= torch.max(probs, dim=1, keepdim=True)[0]  # Normalize probabilities

    points2sample = int(np.ceil(n_points / query.shape[0]))

    # Use torch.multinomial to sample indices based on the given probabilities
    knn_indices = torch.multinomial(probs, num_samples * points2sample, replacement=True).view(num_samples, -1)

    # Concatenate uncertain indices to the sampled indices for the first half of samples
    # if uncertain_ind.shape[0] < n_points/2:
    #     knn_indices[:3, :uncertain_ind.shape[0]] = uncertain_ind.unsqueeze(0).expand(3, -1)

    return knn_indices[:, :n_points]


def knn_indices_topk(ref, query, k):
    """
    KNN using PyTorch

    Parameters:
    - ref: Reference tensor of shape [n_ref, 2]
    - query: Query tensor of shape [n_query, 2]
    - k: Number of nearest neighbors to find

    Returns:
    - indices: Indices of the nearest neighbors (shape: [n_query, k])
    """
    # Expand dimensions for broadcasting
    ref_expanded = ref.unsqueeze(0)  # [1, n_ref, 2]
    query_expanded = query.unsqueeze(1)  # [n_query, 1, 2]

    # Compute the delta (difference) between each query and reference point
    delta = query_expanded - ref_expanded  # [n_query, n_ref, 2]

    # Calculate the squared Euclidean distances
    distances_sq = torch.pow(delta, 2).sum(dim=-1)

    # Find the top k distances and their corresponding indices
    sorted_dist_sq, indices = torch.topk(-distances_sq, k, dim=-1)  # Use topk with negated distances for sorting

    return indices


def recalibrate_batchnorm(model, dataloader, num_batches=10, device='cuda'):
    """Recalculates running mean & variance for BatchNorm layers using test data."""

    # check if model is in device
    model.to(device)
    model.train()  # Temporarily set model to train mode to update BN stats

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_batches:  # Limit the number of batches for recalibration
                break
            pointcloud = data[0]  # Extract only the input
            pc = pointcloud.squeeze(0)

            # convert into float32
            pc = torch.Tensor(pc).to(torch.float32).to(device)  # Move to device

            pc = pc.transpose(2, 1)  # Ensure correct shape for PointNet++
            _ = model(pc[:20, :8, :])  # Forward pass to update BN stats

    model.eval()  # Switch back to evaluation mode


def store_las(array, output_dir, f_name='predictions.las'):
    """
    Store LAS file
    To open with Terrassolid -> version = "1.2" point_format=3

    :param array: points of tile with predicted classes, atributes: id, x, y, z, class
    :param output_dir: path to store las file
    :param f_name: name of file
    :return: None

    ***DO NOT CHANGE THIS CODE TO STORE POINTS CORRECTLY!***
    """
    # Modify class values to standard LAS values
    classes = array[:, 3].astype('int')
    classes[classes==1]=15
    classes[classes==2]=14
    classes[classes==4]=18

    # Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(array[:, :3], axis=0)
    header.scales = np.array([0.001, 0.001, 0.001]) # Scale to preserve 3 decimal places
    
    # Create a Las
    las = laspy.LasData(header)
    las.x = array[:, 0] 
    las.y = array[:, 1] 
    las.z = array[:, 2] 
    las.classification = classes

    # Define the output file path
    las.write(os.path.join(output_dir, f_name))


def remove_isolated_points(points, min_neighbors=3, radius=0.1):
    """
    Removes isolated points based on the number of neighbors within a radius.
    
    :param points: array (id, x, y, z, class) -> Nx3 array (x, y, z)
    :param min_neighbors: Minimum number of neighbors a point must have to be kept
    :param radius: Distance threshold for neighbors
    :return: Filtered points
    """
    xyz = points[:, 1:4]  # Extract x, y, z coordinates

    # Build KDTree (faster than BallTree for lower dimensions like 3D)
    tree = KDTree(xyz, leaf_size=40)

    # Count neighbors within the given radius
    neighbors_count = tree.query_radius(xyz, r=radius, count_only=True)

    # Keep points with at least `min_neighbors`
    mask = neighbors_count >= min_neighbors

    print(f'Num discarded points:{len(neighbors_count < min_neighbors)}')
    
    return points[mask]


class DynamicLabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, smoothing_matrix):
        super(DynamicLabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.smoothing_matrix = smoothing_matrix
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        # Apply label smoothing based on the confusion matrix
        smoothed_targets = torch.zeros_like(output)
        for i in range(self.num_classes):
            smoothed_targets[:, i] = (1 - self.smoothing_matrix[i, i]) * target[:, i] + self.smoothing_matrix[i, i] / (
                    self.num_classes - 1)

        return self.ce(output, smoothed_targets)
