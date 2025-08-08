import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import itertools

# colors
orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
blue = np.array([51 / 256, 153 / 256, 1, 1])
purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
green = np.array([50 / 256, 205 / 256, 50 / 256, 0.8])
light_green = np.array([200 / 256, 250 / 256, 90 / 256, 0.7])
red = np.array([256 / 256, 0 / 256, 50 / 256, 1])
black = np.array([0 / 256, 0 / 256, 0 / 256, 1])
dark_green = np.array([2 / 256, 128 / 256, 90 / 256, 0.7])
pink = np.array([256 / 256, 60 / 256, 200 / 256, 1])  # PINK
light_blue = np.array([0 / 256, 243 / 256, 256 / 256, 1])


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'bo', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    if save_to_file:
        fig.savefig('figures_notebooks/Loss.png', dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'bo', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_3d(points, name, d_color=3, directory = 'figures_notebooks/results_infer'):
    # points = points.numpy()
    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, d_color], s=1,
                    marker='o',
                    cmap="viridis_r",
                    alpha=0.8)
    plt.colorbar(sc, shrink=0.5, pad=0.05)
    # plt.title(name + ' classes: ' + str(set(points[:, 3].astype('int'))))
    plt.show()
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_legend(points, labels, name='', point_size=1, directory='figures_notebooks', set_figsize=[10, 10]):
    """
    3D plot with legend
    Labels range [0,4]
    Expects numpy array as input

    """
    fig = plt.figure(figsize=set_figsize)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10).reversed()
    newcolors = viridisBig(np.linspace(0, 0.9, len(set(labels))))

    cmap = ListedColormap(newcolors)

    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=labels,
                    s=point_size,
                    marker='o',
                    cmap=cmap,
                    alpha=0.5,
                    )

    # Add legend
    unique_labels = np.unique(labels)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Label {label}',
            markerfacecolor=cmap(i / len(unique_labels)), markersize=10)
        for i, label in enumerate(unique_labels)
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Categories")

    plt.title('Points: ' + str(points.shape[0]))

    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.gcf()
        plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()


def plot_3d_subplots(points_tNet, fileName, points_i):
    fig = plt.figure(figsize=[12, 6])
    #  First subplot
    # ===============
    # set up the axes for the first plot
    # print('points_input', points_i.shape)
    # print('points_tNet', points_tNet.shape)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.title.set_text('Input data: ' + fileName)
    sc = ax.scatter(points_i[0, :], points_i[1, :], points_i[2, :], c=points_i[2, :], s=10,
                    marker='o',
                    cmap="winter", alpha=0.5)
    # fig.colorbar(sc, ax=ax, shrink=0.5)  #
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax.scatter(points_tNet[0, :], points_tNet[1, :], points_tNet[2, :], c=points_tNet[2, :], s=10,
                     marker='o',
                     cmap="winter", alpha=0.5)
    ax.title.set_text('Output of tNet')
    plt.show()
    directory = 'figures_notebooks/plots_train/'
    name = 'tNetOut_' + str(fileName) + '.png'
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()


def plot_hist2D(points, name='hist'):
    n_bins = 50
    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0].hist(points[0, 0, :], bins=n_bins)
    # axs[1].hist(points[0, 1, :], bins=n_bins)
    # axs[0].title.set_text('x')
    # axs[1].title.set_text('y')
    # plt.show()

    # 2D histogram
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(points[0, :], points[1, :], bins=n_bins)
    fig.colorbar(hist[3], ax=ax)
    directory = 'figures_notebooks'
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_hist(points, name):
    n_bins = 50
    # fig = plt.figure(tight_layout=True, figsize=[10,10])
    plt.hist(points, bins=n_bins)
    directory = 'figures_notebooks'
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_pointcloud_with_labels(pc, labels, targets=None, name='', path_plot='', point_size=1):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> cables
    # 3 -> surrounding env
    # 4 -> wind turbines
    """

    # labels = labels.astype(int)
    fig = plt.figure(figsize=[20, 10])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 5))

    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = green
    newcolors[4:5, :] = gray

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1))  # zlim=(0, max(pc[:, 2]))
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=targets, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=4)
    plt.title(f'Ground truth - Points: {pc.shape[0]}')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1))
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=4)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    # plt.title('GT Tower pts: ' + str(len(targets[targets == 1])))
    plt.title('Predicted point cloud')

    # Title
    # xstr = lambda x: "None" if x is None else str(round(x, 2))
    # plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) +
    #              ' IoU: [pylon=' + xstr(ious[0]) + ', lines=' + xstr(ious[1]) + ', mIoU=' + xstr(ious[2]) + ']\n',
    #              fontsize=16)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tower', markerfacecolor=purple, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=light_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Wind turbine', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Surrounding', markerfacecolor=green, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Building', markerfacecolor=red, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ground', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(400)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_two_pointclouds_z(pc, pc_2, labels=None, labels_2=None, name='', path_plot='', point_size=1, xyz=None, target_class=4,
                           label_names=['Surrounding', 'Tower', 'Power lines', 'Surrounding', 'Wind turbine']):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> cables
    # 3 -> surrounding env
    # 4 -> wind turbines
    """

    # labels = labels.astype(int)
    fig = plt.figure(figsize=[20, 10])
    N_CLASSES = len(label_names)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, N_CLASSES))
    newcolors[:1, :] = green
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    # newcolors[3:4, :] = green
    # newcolors[4:5, :] = red

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot: Labels
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # xlim=(-1, 1), ylim=(-1, 1) zlim=(0, max(pc[:, 2]))
    # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2]*200, c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 3], c=pc[:, 3], s=point_size, marker='o', cmap='viridis')

    # plt.title(f'Labeled point cloud: {pc.shape[0]} pts - class {target_class}: {len(labels[labels==target_class])} pts')
    plt.title(f'Ground truth')

    # ==============
    # Second subplot: Predictions
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')# xlim=(-1, 1), ylim=(-1, 1))
    # ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2]*200, c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 3], c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    # plt.title(f'Prediction - class {target_class}: {len(labels_2[labels_2==target_class])} pts')
    plt.title(f'Prediction')

    name = 'z_color_' + name + '_' + str(len(labels_2[labels_2==target_class])) +'p' 
    if xyz is not None:
        coords= xyz[0, :]
    
    # Legend
    # ==============
    # Dynamically generate legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label_names[i], markerfacecolor=newcolors[i], markersize=10)
        for i in range(N_CLASSES)
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(300)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_two_pointclouds_hag(pc, pc_2, labels=None, labels_2=None, name='', path_plot='', point_size=1, xyz=None, target_class=4,
                           label_names=['Surrounding', 'Tower', 'Power lines', 'Surrounding', 'Wind turbine']):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> cables
    # 3 -> surrounding env
    # 4 -> wind turbines
    """

    # labels = labels.astype(int)
    fig = plt.figure(figsize=[20, 10])
    N_CLASSES = len(label_names)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, N_CLASSES))
    newcolors[:1, :] = green
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot: Labels
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # xlim=(-1, 1), ylim=(-1, 1) zlim=(0, max(pc[:, 2]))
    # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2]*200, c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 3], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)

    # plt.title(f'Labeled point cloud: {pc.shape[0]} pts - class {target_class}: {len(labels[labels==target_class])} pts')
    plt.title(f'Ground truth')

    # ==============
    # Second subplot: Predictions
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')# xlim=(-1, 1), ylim=(-1, 1))
    # ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2]*200, c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 3], c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)

    plt.title(f'Prediction')

    name = 'z_' + name + '_' + str(len(labels_2[labels_2==target_class])) +'p' 
    if xyz is not None:
        coords= xyz[0, :]
    
    # Legend
    # ==============
    # Dynamically generate legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label_names[i], markerfacecolor=newcolors[i], markersize=10)
        for i in range(N_CLASSES)
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(300)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_two_pointclouds(pc, pc_2, labels=None, labels_2=None, name='', path_plot='', point_size=1, xyz=None):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> cables
    # 3 -> surrounding env
    # 4 -> wind turbines
    """

    # labels = labels.astype(int)
    fig = plt.figure(figsize=[20, 10])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 5))
    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = green
    newcolors[4:5, :] = gray

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # xlim=(-1, 1), ylim=(-1, 1) zlim=(0, max(pc[:, 2]))
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=4)
    plt.title(f'Full point cloud {pc.shape[0]} points')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')# xlim=(-1, 1), ylim=(-1, 1))
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2], c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=4)

    if xyz is not None:
        coords= xyz[0, :]

    plt.title(f'Sample {pc_2.shape[0]} points')

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tower', markerfacecolor=purple, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=light_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Wind turbine', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Surrounding', markerfacecolor=green, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ground', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(300)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_pointcloud_with_labels_barlow(pc, labels, targets, miou, name, path_plot='', point_size=1):
    """# Segmentation labels:
    # 0 -> building
    # 1 -> tower
    # 2 -> lines
    # 3 -> low vegetation
    # 4 -> high vegetation
    # 5 -> roof
    """

    labels = labels.astype(int)
    fig = plt.figure(figsize=[16, 8])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))

    newcolors[:1, :] = light_blue
    newcolors[1:2, :] = blue
    newcolors[2:3, :] = purple
    newcolors[3:4, :] = dark_green
    newcolors[4:5, :] = light_green
    newcolors[5:6, :] = red
    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc, fraction=0.03, pad=0.1)
    plt.title('Predictions')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc2 = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=targets, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    plt.title('Ground truth')

    # Title
    xstr = lambda x: "None" if x is None else str(round(x, 2))
    plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) +
                 'mIoU=' + xstr(miou) + ']\n',
                 fontsize=16)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Building', markerfacecolor=light_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Pylon', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Lines', markerfacecolor=purple, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg', markerfacecolor=dark_green, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg', markerfacecolor=light_green, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Roof', markerfacecolor=red, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(200)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    plt.gcf()
    if path_plot:
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    fig.clear()
    plt.close(fig)


def plot_pointcloud_with_labels_DALES(pc, labels, preds, miou=None, name='plot', path_plot='', point_size=1, n_classes=9):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> poles
    # 3 -> vegetation
    # 4 -> fences-buildings
    # 5 -> cars-trucks
    """

    labels = labels.astype(int)
    fig = plt.figure(figsize=[14, 6])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 1, n_classes))

    newcolors[:1, :] = light_green
    newcolors[1:2, :] = blue
    newcolors[2:3, :] = purple
    newcolors[3:4, :] = green
    newcolors[4:5, :] = red
    newcolors[5:6, :] = pink

    class_labels = ['Ground', 'Tower', 'Poles', 'Vegetation', 'Fences-Buildings', 'Cars-Trucks']
    class_colors = [light_green, blue, purple, green, red, pink]

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc, fraction=0.03, pad=0.1)
    plt.title('Ground Truth')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc2 = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=preds, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    plt.title('Preds')

    # Title
    xstr = lambda x: "None" if x is None else str(round(x, 2))
    plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) ,
                #  'mIoU=' + xstr(miou) + ']\n',
                 fontsize=16)

   # ============
    # Add legend
    # ============
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], markersize=8, label=class_labels[i]) for i in range(len(class_labels))]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(200)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    plt.gcf()
    if path_plot:
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    fig.clear()
    plt.close(fig)


def plot_pc_tensorboard(pc, labels, writer_tensorboard, tag, step, classes=5, lim_z=None, title='', model_name=None):
    if lim_z is None:
        lim_z = [0, 1]
    plt.figure()
    ax = plt.axes(projection='3d' ) #zlim=(lim_z[0], lim_z[1])
    labels = labels.numpy().astype(int)

    # remove padding
    # Get indices where the array is equal to -1
    # ix = list(np.where(labels != -1))  # [batch, n_points]
    # labels = labels[ix]
    # pc = pc[ix, :].squeeze(0)

    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.75, classes))
    newcolors[0:1, :] = np.array([255 / 256, 165 / 256, 0 / 256, 1])  # orange
    newcolors[1:2, :] = np.array([0 / 256, 0 / 256, 1, 1])  # blue
    newcolors[2:3, :] = np.array([256 / 256, 60 / 256, 200 / 256, 1])  # pink
    newcolors[3:4, :] = np.array([0 / 256, 100 / 256, 10 / 256, 0.8])  # dark green
    newcolors[4:5, :] = np.array([256 / 256, 0 / 256, 50 / 256, 1])  # red
    # newcolors[5:6, :] =

    cmap = ListedColormap(newcolors)
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=1, marker='o', cmap=cmap, vmin=0, vmax=classes)
    plt.colorbar(sc, fraction=0.02, pad=0.1)
    plt.title(title)
    fig = plt.gcf()
    directory = '/home/m.caros/work/3DSemanticSegmentation/src/runs/figures/'
    if model_name:
        directory = directory + model_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig(directory + tag + '_' + str(step) + '.png', bbox_inches='tight', dpi=200)
    fig.set_dpi(100)
    if writer_tensorboard:
        writer_tensorboard.add_figure(tag, fig, global_step=step)
    plt.close()

def plot_2d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w):
    """
    Plot sequence of K-means clusters in Tensorboard

    :param pc: [2048, 11]
    :param writer_tensorboard:
    :param filename:
    :param i_w:
    """
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 3], s=10, marker='o', cmap='Spectral')
    plt.colorbar(sc)
    tag = 'k-means_2Dxy_' + filename.split('/')[-1]
    # plt.title('PC')
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)


def plot_3d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):
    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))

    segment_labels = pc[:, 3]
    segment_labels[segment_labels == 15] = 100
    segment_labels[segment_labels == 14] = 200
    segment_labels[segment_labels == 3] = 300  # low veg
    segment_labels[segment_labels == 4] = 300  # med veg
    segment_labels[segment_labels == 5] = 400
    # segment_labels[segment_labels == 18] = 500
    segment_labels[segment_labels < 100] = 0
    segment_labels = (segment_labels / 100)

    # convert array of booleans to array of integers
    labels = segment_labels.numpy().astype(int)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))
    orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
    blue = np.array([0 / 256, 0 / 256, 1, 1])
    purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
    gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = np.array([151 / 256, 188 / 256, 65 / 256, 1])  # green
    newcolors[4:5, :] = np.array([200 / 256, 250 / 256, 90 / 256, 1])  # light green
    cmap = ListedColormap(newcolors)

    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=3, marker='o', cmap=cmap, vmin=0, vmax=5)
    tag = str(n_clusters) + 'c-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pylon', markerfacecolor=purple, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg',
               markerfacecolor=np.array([200 / 256, 250 / 256, 90 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg',
               markerfacecolor=np.array([151 / 256, 188 / 256, 65 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.45, 0.5))  # , bbox_to_anchor=(1.04, 0.5)

    directory = '/home/m.caros/work/3DSemanticSegmentation/figures_notebooks/kmeans_seq/'
    name = filename + '_' + str(i_w) + '.png'
    plt.savefig(directory + name, bbox_inches='tight', dpi=100)

    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)
    plt.close()


def plot_3d_dales_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):
    ax = plt.axes(projection='3d')

    # convert array of booleans to array of integers
    labels = pc[:, 3].numpy().astype(int)

    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=3, marker='o', cmap='viridis_r')
    tag = str(n_clusters) + 'c-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)

    directory = '/home/m.caros/work/3DSemanticSegmentation/figures_notebooks/kmeans_seq/'
    name = filename + '_' + str(i_w) + '.png'
    plt.savefig(directory + name, bbox_inches='tight', dpi=100)

    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)
    plt.close()


def plot_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures_notebooks/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[20, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(selFile.x, selFile.y, selFile.z, c=selFile.z, s=point_size, marker='o', cmap="Spectral")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_2d_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures_notebooks/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[10, 5])
    sc = plt.scatter(selFile.x, selFile.y, c=selFile.z, s=point_size, marker='o', cmap="viridis")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_3d_coords(coords, fileName='', selClass=[], point_size=2, save_dir='',
                   cmap=None, feat_color=None, show=False, figsize=[10, 10], title='', bar=False, legend=False):
    """Plot of point cloud. Can be filtered by a specific class"""
    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 2))
    newcolors[:1, :] = blue
    newcolors[1:2, :] = red
    cmap = ListedColormap(newcolors)

    # plot
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=feat_color, s=point_size, marker='o', cmap=cmap)

    if bar:
        plt.colorbar(sc, shrink=0.5, pad=0.05)
    if legend:
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Uncertain points', markerfacecolor=red, markersize=10)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='x-large')  # , bbox_to_anchor=(1.04, 0.5)

    plt.title(title, fontsize=16)

    if save_dir:
        if selClass:
            name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        else:
            name = fileName + '.png'
        plt.savefig(os.path.join(save_dir, name), bbox_inches='tight', dpi=100)
    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_coords(coords, ax=[], save_plot=False, point_size=40, figsize=[10, 5], save_dir='figures_notebooks/'):
    if not ax:
        fig = plt.figure(figsize=figsize)
        sc = plt.scatter(coords[0], coords[2], c=coords[1], s=point_size, marker='o', cmap="viridis")
        plt.colorbar(sc)
    else:
        ax.scatter(coords[1], coords[2], c=coords[2], s=point_size, marker='o', cmap="viridis")
        ax.title.set_text('Points=%i' % (len(coords[1])))


def plot_confusion_matrix(preds, targets, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # getting the standard confusion matrix in text form
    cm = confusion_matrix(np.asarray(targets), np.asarray(preds))
    # using the matrix generated as means to plot a confusion matrix graphically
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, fontsize=14):

    # using the matrix generated as means to plot a confusion matrix graphically
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize+2)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize)
    plt.yticks(tick_marks, classes,fontsize=fontsize )

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')