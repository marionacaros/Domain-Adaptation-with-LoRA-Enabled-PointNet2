import torch
import random


class CollatorPadd(object):
    """
        Pads batch of variable length
        Replicate points up to desired length of windows
        Padds with -1 tensor of targets to mask values in loss during training

        :return: batch_data. Tensor [batch, 2048, 11, 20]
                 pad_targets. Tensor [batch, 2048, 20]
                 filenames. List

        """
    def __init__(self, n_points, max_clusters):
        self.n_points = n_points
        self.max_clusters = max_clusters

    def __call__(self, batch):

        b_data = [torch.FloatTensor(t[0]) for t in batch]
        targets = [torch.LongTensor(t[1]) for t in batch]
        filenames = [t[2] for t in batch]
        # centroids = [torch.FloatTensor(t[3]) for t in batch]

        # padding
        batch_data = []
        pad_targets = []
        pad_centroids = []
        i = 0

        for pc_w, target in zip(b_data, targets):
            # cent = centroids[i].unsqueeze(1)
            if pc_w.shape[0] < self.n_points:
                rdm_list = torch.randint(0, pc_w.shape[0], (self.n_points,))
                pc_w = pc_w[rdm_list, :, :]
                target = target[rdm_list, :]

            elif pc_w.shape[0] > self.n_points:
                ix = random.sample(range(pc_w.shape[0]), self.n_points)
                pc_w = pc_w[ix, :, :]
                target = target[ix, :]
            p1d = (0, self.max_clusters - pc_w.shape[2])  # pad last dim
            # p1d = (0,0)
            batch_data.append(torch.nn.functional.pad(pc_w, p1d, "replicate"))
            pad_targets.append(torch.nn.functional.pad(target, p1d, "constant", -1))
            # pad_centroids.append(torch.nn.functional.pad(cent, p1d, "replicate"))
            i += 1

        batch_data = torch.stack(batch_data, dim=0)
        pad_targets = torch.stack(pad_targets, dim=0)
        # pad_centroids = torch.stack(pad_centroids, dim=0)
        # pad_centroids = pad_centroids.view(-1,MAX_WINDOWS, 2)

        return batch_data, pad_targets, filenames  # , pad_centroids


def collate_seq_padd(batch):
    """
    Pads batch of variable length
    Replicate points up to desired length of windows
    Padds with -1 tensor of targets to mask values in loss during training

    :param batch: List of point clouds.
    :return: batch_data. Tensor [batch, 2048, 11, 20]
             pad_targets. Tensor [batch, 2048, 20]
             filenames. List

    """

    N_POINTS = 4096
    MAX_WINDOWS = 6

    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]
    # centroids = [torch.FloatTensor(t[3]) for t in batch]

    # padding
    batch_data = []
    pad_targets = []
    pad_centroids = []
    i = 0

    for pc_w, target in zip(b_data, targets):
        # cent = centroids[i].unsqueeze(1)
        if pc_w.shape[0] < N_POINTS:
            rdm_list = torch.randint(0, pc_w.shape[0], (N_POINTS,))
            pc_w = pc_w[rdm_list, :, :]
            target = target[rdm_list, :]

        elif pc_w.shape[0] > N_POINTS:
            ix = random.sample(range(pc_w.shape[0]), N_POINTS)
            pc_w = pc_w[ix, :, :]
            target = target[ix, :]
        p1d = (0, MAX_WINDOWS - pc_w.shape[2])  # pad last dim
        batch_data.append(torch.nn.functional.pad(pc_w, p1d, "replicate"))
        pad_targets.append(torch.nn.functional.pad(target, p1d, "constant", -1))
        # pad_centroids.append(torch.nn.functional.pad(cent, p1d, "replicate"))
        i += 1

    batch_data = torch.stack(batch_data, dim=0)
    pad_targets = torch.stack(pad_targets, dim=0)
    # pad_centroids = torch.stack(pad_centroids, dim=0)
    # pad_centroids = pad_centroids.view(-1,MAX_WINDOWS, 2)

    return batch_data, pad_targets, filenames #, pad_centroids


def collate_cls_padd(batch):
    """
    Pads batch of variable length
    Replicate points up to desired length of windows
    Padds with -1 tensor of targets to mask values in loss during training

    :param batch: List of point clouds.
    :return: batch_data. Tensor [batch, 2048, 11, 20]
             pad_targets. Tensor [batch, 2048, 20]
             filenames. List

    """
    N_POINTS = 2048
    MAX_WINDOWS = 9

    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]
    centroids = [torch.FloatTensor(t[3]) for t in batch]
    labels_segmen = [torch.LongTensor(t[4]) for t in batch]

    # padding
    batch_data = []
    pad_labels_segmen = []
    pad_centroids = []
    i = 0

    for pc_w, labels in zip(b_data, labels_segmen):
        cent = centroids[i].unsqueeze(1)
        if pc_w.shape[0] < N_POINTS:
            rdm_list = torch.randint(0, pc_w.shape[0], (N_POINTS,))
            pc_w = pc_w[rdm_list, :, :]
            labels = labels[rdm_list, :]
            # pc_w = torch.cat([pc_w, extra_points], dim=0)

        elif pc_w.shape[0] > N_POINTS:
            ix = random.sample(range(pc_w.shape[0]), N_POINTS)
            pc_w = pc_w[ix, :, :]
            labels = labels[ix, :]

        p1d = (0, MAX_WINDOWS - pc_w.shape[2])  # pad last dim
        batch_data.append(torch.nn.functional.pad(pc_w, p1d, "replicate"))
        pad_centroids.append(torch.nn.functional.pad(cent, p1d, "replicate"))
        pad_labels_segmen.append(torch.nn.functional.pad(labels, p1d, "constant", -1))

        i += 1

    batch_data = torch.stack(batch_data, dim=0)
    targets = torch.stack(targets, dim=0)
    pad_centroids = torch.stack(pad_centroids, dim=0)
    pad_centroids = pad_centroids.view(-1,MAX_WINDOWS, 2)
    pad_labels_segmen = torch.stack(pad_labels_segmen, dim=0)

    # batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=False, padding_value=0)  # [max_length,B,D]

    return batch_data, targets, filenames, pad_centroids, pad_labels_segmen
