import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from progressbar import progressbar
import itertools
import os
import pickle
import laspy
from k_means_constrained import KMeansConstrained


def kmeans_clustering(in_pc, n_points=2048, get_centroids=True, max_clusters=18, ix_features=[0, 1, 8], out_path='',
                      file_name=''):
    """
    K-means constrained

    :param in_pc: torch.Tensor [points, dim]
    :param n_points: int
    :param get_centroids: bool
    :param max_clusters: int
    :param ix_features: list of indices of features to be used for k-means clustering
    :param out_path: str, if not '' path for saving tensor clusters
    :param file_name: str

    :return: cluster_lists [], centroids Torch.Tensor []
    """
    # in_pc [n_p, dim]
    MAX_CLUSTERS = max_clusters
    cluster_lists = []
    in_pc = in_pc.squeeze(0)
    centroids = torch.FloatTensor()

    # if point cloud is larger than n_points we cluster them with k-means
    if in_pc.shape[0] >= 2 * n_points:

        # K-means clustering
        k_clusters = int(np.floor(in_pc.shape[0] / n_points))

        if k_clusters > MAX_CLUSTERS:
            k_clusters = MAX_CLUSTERS

        if k_clusters * n_points > in_pc.shape[0]:
            print('debug error')

        clf = KMeansConstrained(n_clusters=k_clusters, size_min=n_points,
                                n_init=3, max_iter=10, tol=0.01,
                                verbose=False, random_state=None, copy_x=True, n_jobs=-1
                                )
        i_f = ix_features  # x,y, NDVI
        i_cluster = clf.fit_predict(in_pc[:, i_f].numpy())  # array of ints -> indices to each of the windows

        # get tuple cluster points
        tuple_cluster_points = list(zip(i_cluster, in_pc))
        cluster_list_tuples = [list(item[1]) for item in
                               itertools.groupby(sorted(tuple_cluster_points, key=lambda x: x[0]), key=lambda x: x[0])]

        for cluster in cluster_list_tuples:
            pc_cluster_tensor = torch.stack([feat for (i_c, feat) in cluster])  # [2048, 11]
            cluster_lists.append(pc_cluster_tensor)
            if get_centroids:
                centroid = get_cluster_centroid(pc_cluster_tensor).unsqueeze(0)
                centroids = torch.cat([centroids, centroid], dim=0)

    else:
        cluster_lists.append(in_pc)
        # get centroids
        if get_centroids:
            centroids = get_cluster_centroid(in_pc)
            centroids = centroids.unsqueeze(0)

    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, file_name + '_clusters') + '.pt', 'wb') as f:
            torch.save(cluster_lists, f)

        # with open(os.path.join(out_path, file_name + '_centroids') + '.pt', 'wb') as f:
        #     torch.save(centroids, f)

    return cluster_lists, centroids


def split4classif_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                              targets=[], task='classification', device='cuda'):
    """ split point cloud in windows of fixed size (n_points)
        and padd with 0 needed points to fill the window

    :param lengths:
    :param filenames:
    :param task:
    :param targets:
    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot:
    :param writer_tensorboard:

    :return pc_w: point cloud in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    count_p = 0
    j = 0
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]

        else:
            # padd with zeros to fill the window -> només aplica a la última finestra del batch
            points_needed = end_batch - points.shape[1]
            in_points = points[:, j * n_points:, :]
            if points_needed != n_points:
                # padd with zeros
                padd_points = torch.zeros(points.shape[0], points_needed, points.shape[2]).to(device)
                in_points = torch.cat((in_points, padd_points), dim=1)
                if task == 'segmentation':
                    extra_targets = torch.full((targets.shape[0], points_needed), -1).to(device)
                    targets = torch.cat((targets, extra_targets), dim=1)

        if plot:
            # write figure to tensorboard
            ax = plt.axes(projection='3d')
            pc_plot = in_points.cpu()
            sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10, marker='o',
                            cmap='Spectral')
            plt.colorbar(sc)
            tag = filenames[0].split('/')[-1]
            plt.title(
                'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                    targets[0].cpu().numpy()))
            writer_tensorboard.add_figure(tag, plt.gcf(), j)

        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points into tensor w
        pc_w = torch.cat([pc_w, in_points], dim=3)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets



def get_cluster_centroid(pc):
    mean_x = pc[:, 0].mean(0)  # [1, n_clusters]
    mean_y = pc[:, 1].mean(0)  # [1, n_clusters]

    centroids = torch.stack([mean_x, mean_y], dim=0)
    return centroids



def split4segmen_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                             targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets: [batch, n_samples]
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        # get padded points (padding target value = -1)
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]

            # transform targets into Long Tensor
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

            # write figure to tensorboard
            if plot:
                ax = plt.axes(projection='3d')
                pc_plot = in_points.cpu()
                sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                marker='o',
                                cmap='Spectral')
                plt.colorbar(sc)
                tag = filenames[0].split('/')[-1]
                plt.title(
                    'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                        in_targets[0].cpu().numpy()))
                writer_tensorboard.add_figure(tag, plt.gcf(), j)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4segmen_test(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                      targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets:
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)

    count_p = 0
    j = 0
    # loop over windows
    while j < 4:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]
        else:
            # get duplicated points from previous windows
            rdm_list = np.random.randint(0, points.shape[1], n_points)
            in_points = points[:, rdm_list, :]  # [2048, 11]
            in_targets = targets[:, rdm_list]  # [2048]

        # transform targets into Long Tensor
        in_targets = torch.LongTensor(in_targets.cpu()).to(device)
        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points and targets into tensor w
        pc_w = torch.cat((pc_w, in_points), dim=3)
        targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4cls_kmeans(o_points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[],
                     targets=torch.Tensor(), duplicate=True):
    """ split point cloud in windows of fixed size (n_points) with k-means
        Fill empty windows with duplicate points of previous windows
        Number of points must be multiple of n_points, so points left over are removed

        :param o_points: input point cloud [batch, n_samples, dims]
        :param n_points: number of points
        :param plot: bool set to True for plotting windows
        :param writer_tensorboard:
        :param filenames: []
        :param targets: [batch, w_len]
        :param duplicate: bool

        :return pc_w: tensor containing point cloud in windows of fixed size [b, 2048, dims, w_len]
        :return targets_w: tensor of targets [b, w_len]

    """

    # o_points = o_points.to('cpu')

    # if point cloud is larger than n_points we cluster them with k-means
    if o_points.shape[1] > n_points:

        pc_batch = torch.FloatTensor()
        targets_batch = torch.LongTensor()

        if o_points.shape[1] % n_points != 0:
            # Number of points must be multiple of n_points, so points left over are removed
            o_points = o_points[:, :n_points * (o_points.shape[1] // n_points), :]

        K_clusters = int(np.floor(o_points.shape[1] / n_points))
        clf = KMeansConstrained(n_clusters=K_clusters, size_min=n_points, size_max=n_points, random_state=0)

        # loop over batches
        for b in progressbar(range(o_points.shape[0]), redirect_stdout=True):
            # tensor for points per window
            pc_w = torch.FloatTensor()

            # todo decide how many features get for clustering
            i_f = [4, 5, 6, 7, 8, 9]  # x,y,z,label,I,R,G,B,NIR,NDVI
            clusters = clf.fit_predict(o_points[b, :, i_f].numpy())  # array of ints -> indices to each of the windows

            # loop over clusters
            for c in range(K_clusters):
                ix_cluster = np.where(clusters == c)
                # sample and get all features again
                in_points = o_points[b, ix_cluster, :]  # [batch, 2048, 11]

                # get position of in_points where all features are 0
                i_bool = torch.all(in_points == 0, dim=2).view(-1)
                # if there are padding points in the cluster
                if True in i_bool:
                    added_p = True
                    points_needed = int(sum(i_bool))
                    if duplicate:
                        # get duplicated random points
                        first_points = in_points[:, ~i_bool, :]
                        rdm_list = np.random.randint(0, n_points, points_needed)

                        in_points = o_points[b, rdm_list, :].view(1, points_needed, 11)
                        # concat points if not all points are padding points
                        if first_points.shape[1] > 0:
                            in_points = torch.cat([first_points, in_points], dim=1)
                else:
                    added_p = False

                in_points = torch.unsqueeze(in_points, dim=3)  # [1, 2048, 11, 1]
                # concat points of cluster
                pc_w = torch.cat((pc_w, in_points), dim=3)

                if int(targets[b, 0]) == 1 or b == 0:  # if there is a tower
                    # write figure to tensorboard
                    if plot:
                        ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                        pc_plot = in_points
                        sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                        marker='o',
                                        cmap='Spectral')
                        plt.colorbar(sc)
                        tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                        plt.title('PC size: ' + str(o_points.shape[1]) + ' added P: ' + str(added_p))
                        writer_tensorboard.add_figure(tag, plt.gcf(), c)

                        if c == 4:
                            ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                            sc = ax.scatter(o_points[b, :, 0], o_points[b, :, 1], o_points[b, :, 2],
                                            c=o_points[b, :, 3],
                                            s=10,
                                            marker='o',
                                            cmap='Spectral')
                            plt.colorbar(sc)
                            tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                            plt.title('original PC size: ' + str(o_points.shape[1]))
                            writer_tensorboard.add_figure(tag, plt.gcf(), c)

            # concat batch
            pc_batch = torch.cat((pc_batch, pc_w), dim=0)
            targets_batch = torch.cat((targets_batch, targets[b, 0].unsqueeze(0)), dim=0)

        # broadcast targets_batch to shape [batch, w_len]
        targets_batch = targets_batch.unsqueeze(1)
        targets_batch = targets_batch.repeat(1, targets.shape[1])

    # if point cloud is equal n_points
    else:
        pc_batch = o_points
        targets_batch = targets

    return pc_batch, targets_batch


def split4cls_rdm(points, n_points=2048, targets=[], device='cuda', duplicate=True):
    """ Random split for classification
        split point cloud in windows of fixed size (n_points)
        check batches with padding (-1) and fill windows with duplicate points of previous windows

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points in window
    :param targets: [b, w_len]
    :param device: 'cpu' or 'cuda'
    :param duplicate: bool

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    points = points.cpu()

    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j].cpu()  # [batch, 1]

            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # get empty batches
                batches_null = (in_targets == -1).numpy()
                if duplicate:
                    # get duplicated points from previous window
                    rdm_list = np.random.randint(0, end_batch - n_points, n_points)
                    copy_points = points[:, rdm_list, :]
                    extra_points = copy_points[batches_null, :, :]
                    extra_points = extra_points.view(-1, n_points, 11)
                    in_points = torch.cat((in_points[~ batches_null, :, :], extra_points), dim=0)
                    extra_targets = targets[batches_null, 0]
                    in_targets = torch.cat((in_targets[~ batches_null].to(device), extra_targets), dim=0)
                else:
                    # padd with 0
                    in_points[batches_null, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)

            in_points = torch.unsqueeze(in_points, dim=3).to(device)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3).to(device)
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_targets = torch.unsqueeze(in_targets, dim=1)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w



def get_labels_clusters(cluster_lists):
    """
    Get labels for segmentation

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    2 -> cables
    3 -> low vegetation
    4 -> high vegetation
    """

    segment_labels_list = []

    for pointcloud in cluster_lists:
        pointcloud = pointcloud.squeeze(0)
        segment_labels = pointcloud[:, 9]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300
        segment_labels[segment_labels == 4] = 400
        segment_labels[segment_labels == 5] = 400
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        # segment_labels[segment_labels == 15] = 1
        # segment_labels[segment_labels != 15] = 0

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        segment_labels_list.append(labels)

    return segment_labels_list

# ##################################################### NOT USED #####################################################


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data
