import os
import random
from utils.utils import *
import torch.utils.data as data
import torch
import numpy as np
import logging
import csv 
from utils.utils_plot import *
import cv2
import os
from scipy.ndimage import gaussian_filter

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


############################################# TRAIN DATASETS  ###############################################


class CAT3Dataset(data.Dataset):

    def __init__(self,
                 task='segmentation',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 return_xyz=False,
                 use_z=False,
                 use_windturbine=True,
                 store_filtered_paths=None,
                 check_files=False,
                 is_prod=False,
                 add_noise_to_pc=False,
                 remove_classes=[],
                 max_z = None
                 ):

        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.return_xyz = return_xyz
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.paths_files = self.files
        self.use_z = use_z
        self.use_windturbine = use_windturbine
        self.is_prod = is_prod
        self.add_noise_to_pc=add_noise_to_pc
        self.remove_classes = remove_classes
        self.max_z = max_z

        if check_files:
            print(f'Num files before height filtering = {len(self.paths_files)}')
            # Filter out files with short height point clouds or not enough points
            self.paths_files = [
                file for file in self.paths_files
                if not self._is_short(file)
            ]
            print(f'Num files after height filtering = {len(self.paths_files)}')
            
            if store_filtered_paths:
                print(f'Length stored files: {len(self.paths_files)}')

                if not os.path.exists(os.path.dirname(store_filtered_paths)):
                    os.makedirs(os.path.dirname(store_filtered_paths))

                with open(store_filtered_paths, 'a') as fid:
                    writer = csv.writer(fid)
                    for file in self.paths_files:
                        writer.writerow([file],)

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
            input dims: x, y, z, class, I, R, G, NIR, NDVI, HAG, point_ID
        :return: pc: [n_points, 5], labels, filename
            if use_z: out dims: x, y, HAG, z, I, G, B, NDVI
            else: out dims: x, y, z, I, G, B, NDVI
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename)

        if self.is_prod:
            default_class = 3
        else:
            # Lora labels
            default_class = 0

        # get labels depending on task
        labels = self.get_labels_segmen(pc, self.is_prod, self.use_windturbine, default_class=default_class)

        if self.use_z:
            # pc = np.concatenate((pc[:, :2], pc[:, 10:11], pc[:, 2:3], pc[:, 12:], pc[:, 4:5], pc[:, 6:8], pc[:, 9:10]), axis=1)  # [n_p, 9]   
            pc = np.concatenate((pc[:, :2], pc[:, 10:11], pc[:, 2:3], pc[:, 4:5], pc[:, 6:8], pc[:, 9:10]), axis=1)  # [n_p, 8]   
        else:
            pc = np.concatenate((pc[:, :2], pc[:, 10:11], pc[:, 4:5], pc[:, 6:8], pc[:, 9:10]), axis=1)  # [n_p, 7]   

        # if self.return_xyz:
        #     return pc, labels, filename, xyz

        return pc, labels, filename


    def _is_short(self, filename):
        """
        Helper function to check if a file corresponds to a short sample.
        :param filename: Path to the file.
        :return: True if files needs to be filtered out, False otherwise.
        """
        remove_file = self.check_file(filename)
        return remove_file
    

    def check_file(self, point_file):
        """
        return: remove: boolean is True if max hag in point cloud < 5 meters or not enough points
        """
        remove = False

        with open(point_file, 'rb') as f:
            pc = torch.load(f)

        pc = pc[~np.isin(pc[:, 3], self.remove_classes)]

        if pc.shape[0] < self.n_points:
            remove = True
            return remove
        
        # Check if max height is short so no objects can be there
        if pc[:,10].max()*200 < 5:
            remove = True

        # if pc[:,10].max()*200 > 200:
            # print(f'Outlier at max height: {pc[:,10].max()*200} meters')

        return remove


    def prepare_data(self,
                     point_file):
        """
        dimensions of point cloud :
        0 - x
        1 - y
        2 - z
        3 - label
        4 - I
        5 - red
        6 - green
        7 - blue
        8 - nir
        9 - ndvi
        10 - hag
        11 - ix
        :param point_file: path of file

        :return: torch tensor [points, dims]
        """
        sample_pts=self.n_points
        xyz = []

        with open(point_file, 'rb') as f:
            pc = torch.load(f)

        pc = pc.to(torch.float32) # atenció si es volen agafar les coordenades, convertir a float32 despres
        pc = pc[~np.isin(pc[:, 3], self.remove_classes)]
        
        # store raw xyz
        # if self.return_xyz:
        #     xyz = pc[:, :3].copy()

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        
        # Normalize z between 0 and 1
        if self.max_z:
            # Local Z norm
            pc_localZ = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min())
            pc_localZ = pc_localZ.unsqueeze(1)

            # Global Z normalization
            pc[:, 2] = pc[:, 2]/ self.max_z
            pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

            pc = torch.cat((pc, pc_localZ), dim=1)
        else:
            pc[:, 2] = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min())
        
        # make sure intensity and color are in range (0,1)
        pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)  
        pc[:, 6] = np.clip(pc[:, 6], 0.0, 1.0)
        pc[:, 7] = np.clip(pc[:, 7], 0.0, 1.0)

        # get 2D views
        # if self.get_views:
            # views, pixel_ix = self.create_2d_cv_views(pc)

        # random sample points if fixed_num_points
        if self.fixed_num_points and pc.shape[0] > sample_pts:
            sampling_indices = random.sample(range(pc.shape[0]), sample_pts)
            pc = pc[sampling_indices, :]
            
        # duplicate points if needed
        elif self.fixed_num_points and pc.shape[0] < sample_pts:
            points_needed = sample_pts - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)
            
        return pc
    

    def get_sampled_pc_below2m(self, pc):
        # check number of points below 2 meters in hag
        mask_low_pts = pc[:, 10] < 2/200.0
        if pc.shape[0] - mask_low_pts.sum().item() > 4000:
            # set to false three quarters of the True values in the mask_low_pts
            n_false= int(mask_low_pts.sum()*0.75)

            if pc.shape[0] - n_false > 8000:
                # print(f'Removed points, {n_false}, total points, {pc.shape[0]}')
                mask_low_pts[np.random.choice(np.where(mask_low_pts)[0], n_false)] = False
                # concatenate points of mask_low_pts==True and points hag >= 2 meters
                pc = np.concatenate([pc[mask_low_pts], pc[pc[:, 10] >= 2]], axis=0)

        return pc
    

    def add_noise(self, points, num_points=3000):
        """
        Add synthetic noise to the point cloud
        Number of noise points (class=104) in original point clouds of 80x80m is in range 2000-5000 pts
        :param points: [n_points, 12]
        :param num_points: number of points to add noise
        :return: points with noise [n_points, 3]
        """
        # Generate noise for all features
        noise = np.random.rand(num_points, points.shape[1])  # Values between 0 and 1

        # Modify feature 4 and 10 to have noise between 0 and 0.1
        noise[:, 4] *= 0.1  
        noise[:, 10] *= 0.1  

        # Concatenate noise as new points
        points_with_noise = np.vstack((points, noise))  # Shape [original_n + num_points, 12]

        return points_with_noise


    def get_labels_segmen(self, pointcloud, is_prod, use_windturbine, default_class=0, is_train=True):
        """
        Get labels for segmentation

        Segmentation labels with ground:
        0 -> ground
        1 -> towers
        2 -> lines
        3 -> surrounding
        4 -> wind turbine

        Segmentation labels if no ground:
        0 -> surrounding 
        1 -> towers
        2 -> lines

        :param pointcloud: [num_points, dim]
        :return labels: classes of points [num_points]
        """

        segment_labels = pointcloud[:, 3]

        if is_prod:

            if is_train:
                # Remove small towers 
                mask = (segment_labels == 15)  
                if 0 < mask.sum() < 6:  
                    # plot_3d_legend(pointcloud, segment_labels, name=name, point_size=1, directory='figures', set_figsize=[10, 10])
                    segment_labels[mask] = 3  # class surr

                conditions = [
                    (segment_labels == 2),
                    (segment_labels == 15),
                    (segment_labels == 14),
                    (segment_labels == 29) |  (segment_labels == 19), # windturbine
                ]
                choices = [0, 1, 2, 4]
            
            else:
                # keep noise from sensor in class 12 for inference
                conditions = [
                (segment_labels == 2),
                (segment_labels == 15),
                (segment_labels == 14),
                (segment_labels == 29) |  (segment_labels == 19), # windturbine
                (segment_labels == 12),
                ]
                choices = [0, 1, 2, 4, 12]
        
        elif use_windturbine:
            conditions = [
                (segment_labels == 2),
                (segment_labels == 15),
                (segment_labels == 14),
                (segment_labels == 29) |  (segment_labels == 19) |  (segment_labels == 18), # windturbine
            ]
            choices = [0, 1, 2, 3]
            
        else: 
            conditions = [
                (segment_labels == 2),
                (segment_labels == 15), # |  (segment_labels == 18),
                (segment_labels == 14),
            ]
            choices = [0, 1, 2]

        # Default choice for other classes not interested other towers + vegetation + buildings
        segment_labels = np.select(conditions, choices, default=default_class)

        return segment_labels

    

class CAT3DatasetViews(CAT3Dataset):

    def __init__(self,
                 task='segmentation',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 return_xyz=False,
                 use_z=False,
                 use_windturbine=True,
                 store_filtered_paths=None,
                 check_files=False,
                 is_prod=False,
                 add_noise_to_pc=False,
                 remove_otherground=True,
                 remove_classes=[],
                 sample_below2m=False,
                 get_views=False,
                 max_z = None #277
                 ):

        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.return_xyz = return_xyz
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.paths_files = self.files
        self.use_z = use_z
        self.use_windturbine = use_windturbine
        self.remove_otherground = remove_otherground
        self.is_prod = is_prod
        self.add_noise_to_pc=add_noise_to_pc
        self.remove_classes = remove_classes
        self.sample_below2m =  sample_below2m
        self.get_views = get_views
        self.max_z = max_z

        if check_files:
            print(f'Num files before height filtering = {len(self.paths_files)}')
            # Filter out files with short height point clouds or not enough points
            self.paths_files = [
                file for file in self.paths_files
                if not self._is_short(file)
            ]
            print(f'Num files after height filtering = {len(self.paths_files)}')
            
            if store_filtered_paths:
                print(f'Length stored files: {len(self.paths_files)}')

                if not os.path.exists(os.path.dirname(store_filtered_paths)):
                    os.makedirs(os.path.dirname(store_filtered_paths))

                with open(store_filtered_paths, 'a') as fid:
                    writer = csv.writer(fid)
                    for file in self.paths_files:
                        writer.writerow([file],)

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
            input dims: x, y, z, class, I, R, G, NIR, NDVI, HAG, point_ID
        :return: pc: [n_points, 5], labels, filename
            if use_z: out dims: x, y, HAG, z, I, G, B, NDVI
            else: out dims: x, y, z, I, G, B, NDVI
        """
        filename = self.paths_files[index]
        pc, views, point2pixelmap = self.prepare_data(filename)

        if self.is_prod:
            default_class = 3
        else:
            # Lora labels
            default_class = 0

        # get labels depending on task
        labels = self.get_labels_segmen(pc, self.is_prod, self.use_windturbine, default_class=default_class)

        # if self.use_z:
        pc = np.concatenate((pc[:, :2], pc[:, 10:11], pc[:, 2:3], pc[:, 4:5], pc[:, 6:8], pc[:, 9:10]), axis=1)  # [n_p, 8]   

        return pc, labels, filename, views, point2pixelmap
    

    def prepare_data(self,
                     point_file):
        """
        dimensions of point cloud :
        0 - x
        1 - y
        2 - z
        3 - label
        4 - I
        5 - red
        6 - green
        7 - blue
        8 - nir
        9 - ndvi
        10 - hag
        11 - point id

        :param point_file: path of file
        :return: torch tensor [points, dims]
        """
        sample_pts=self.n_points
        xyz = []
        views=[]

        with open(point_file, 'rb') as f:
            pc = torch.load(f)

        pc = pc.to(torch.float32) # atenció si es volen agafar les coordenades, convertir a float32 despres
        pc = pc[~np.isin(pc[:, 3], self.remove_classes)]
        
        # store raw xyz
        # if self.return_xyz:
        #     xyz = pc[:, :3].copy()

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        # Normalize z between 0 and 1
        if self.max_z:
            # print(f'Normalization / max z: {self.max_z}')
            pc[:, 2] = pc[:, 2]/ self.max_z
            pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)  
        else:
            pc[:, 2] = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min())
        
        # make sure intensity and color are in range (0,1)
        pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)  
        pc[:, 6] = np.clip(pc[:, 6], 0.0, 1.0)
        pc[:, 7] = np.clip(pc[:, 7], 0.0, 1.0)

        # random sample points if fixed_num_points
        if self.fixed_num_points and pc.shape[0] > sample_pts:
            sampling_indices = random.sample(range(pc.shape[0]), sample_pts)
            pc_sampled = pc[sampling_indices, :]
            
        # duplicate points if needed
        elif self.fixed_num_points and pc.shape[0] < sample_pts:
            points_needed = sample_pts - pc.shape[0]
            sampling_indices = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[sampling_indices, :]
            pc_sampled = np.concatenate([pc, extra_points], axis=0)
            sampling_indices = np.arange(pc.shape[0]) + sampling_indices # DEBUG
        else:
            pc_sampled=pc
            sampling_indices = np.arange(pc.shape[0])

        # get 2D views
        if self.get_views:
            views, pixel_ix = self.create_views2D(pc, 256, sample_ix=sampling_indices)
            
        return pc_sampled, views, pixel_ix

    
    def create_views2D(self, pc, image_size=256, sample_ix=[], sample_size=None):
         
        # normalize axes between 0 and 1
        pc[:, 0] = (pc[:, 0] +1) / 2
        pc[:, 1] = (pc[:, 1] +1) / 2

        # sample pc for generating views with less points
        if sample_size and len(pc) > sample_size:
            pc = pc[np.random.choice(len(pc), sample_size, replace=False)]
 
        point_ids = pc[:, 11].long() # Store point IDs

        # Prepare coordinate projections
        coords = np.stack([
            (pc[:, [0, 1, 2]]),  # XY view -> use Z as value
            (pc[:, [0, 2, 1]]),  # XZ view -> use Y as value
            (pc[:, [1, 2, 0]])   # YZ view -> use X as value
        ], axis=0)  # Shape: (3, N, 3)

        views = np.zeros((3, image_size, image_size), dtype=np.float32)
        
        # Preallocate mapping: [3, N, 3] → for each view: [point_id, x, y]
        mappings = np.empty((3, coords.shape[1], 3), dtype=np.int32)  # [view, N, (id, x, y)]
        
        # get random number between 0 and 300
        # rdm = np.random.randint(0, 300)
        # o_path='/home/m.caros/work/3DSemanticSegmentation/figures/views2'

        for i in range(3):
            x = ((coords[i, :, 0]) * (image_size - 1)).astype(int)
            y = (((1 - coords[i, :, 1])) * (image_size  - 1)).astype(int)
            values = coords[i, :, 2] * 255

            # # Use np.maximum to preserve max depth value if overlapping
            np.maximum.at(views[i], (y, x), values)

            # Fill mapping tensor
            mappings[i, :, 0] = point_ids        # point ID
            mappings[i, :, 1] = x                # x
            mappings[i, :, 2] = y                # y

            # Plot - Convert to uint8 and save image
            # img_uint8 = views[i].astype(np.uint8)
            # save_path = os.path.join(o_path, f"{rdm}_{i}_Zview.png")
            # cv2.imwrite(save_path, img_uint8)

        # get only mappings of sampled points
        mappings = mappings[:,sample_ix,:]

        return views, mappings  # views: [3, H, W], mappings: [3, N, 3]



############################################# TEST & INFERENCE DATASETS  ###############################################


class CAT3SamplingDataset(CAT3Dataset):

    def __init__(self,
                 task='segmentation',
                 n_points=None,
                 files=None,
                 fixed_num_points=True,
                 return_coord=False,
                 use_z=True,
                 tile_ids=False,
                 keep_labels=False,
                 use_windturbine=False,
                 store_filtered_path='',
                 check_files=False,
                 is_prod=False,
                 remove_otherground=True,
                 remove_classes=[],
                 sample_below2m=False,
                 max_id=100_000_000,
                 max_z= None,
                 get_views=False):

        super().__init__(task, n_points, files, fixed_num_points)
        self.n_points = n_points
        self.return_xyz = return_coord
        self.paths_files = files
        self.use_tile_ids = tile_ids
        self.use_z = use_z
        self.remove_otherground = remove_otherground
        self.keep_labels = keep_labels # for easy ploting
        self.use_windturbine = use_windturbine
        self.is_prod=is_prod
        self.remove_classes = remove_classes
        self.sample_below2m = sample_below2m
        self.max_id = max_id
        self.max_z = max_z
        self.get_views = get_views

        if check_files:
            print(f'Num files before height filtering = {len(self.paths_files)}')

            # Filter out files with short height point clouds
            self.paths_files = [
                file for file in self.paths_files
                if not self._is_short(file)
            ]
            print(f'Num files after height filtering = {len(self.paths_files)}')

            if store_filtered_path:
                print(f'Length stored files: {len(self.paths_files)}')
                with open(store_filtered_path, 'a') as fid:
                    writer = csv.writer(fid)
                    for file in self.paths_files:
                        writer.writerow([file],)


    def __len__(self):
        return len(self.paths_files)
    

    def __getitem__(self, index):
        """
        :param index: index of the file
            input dims: x, y, z, class, I, R, G, NIR, NDVI, HAG, point_ID

        :return: 
        pc: [seq, n_points, dims]  out dims: x, y, HAG, z, I, G, B, NDVI
        labels
        filename
        n_unique_pts
           
        """
        filename = self.paths_files[index]

        if self.get_views:
            pc, xyz, ids, n_unique_pts, views, point2pixel = self.prepare_data(filename)
        else:
            pc, xyz, ids, n_unique_pts = self.prepare_data(filename)

        if self.is_prod:
            default_class = 3
        else:
            # LoRA labels
            default_class = 0

        # get labels depending on num classes and dataset
        # labels = self.get_labels_segmen(np.reshape(pc, (-1, pc.shape[2])), self.is_prod, self.use_windturbine, default_class=default_class)
        labels = self.get_labels_segmen(pc.view(-1, pc.shape[2]).numpy(), self.is_prod, self.use_windturbine, default_class=default_class, is_train=False)

        # for plotting
        if self.keep_labels:  
            pc[:, :, 3] =  torch.FloatTensor(labels).view(-1, pc.shape[1])
            pc = np.concatenate((pc[:, :, :2], pc[:,  :, 10:11], pc[:,  :, 2:3], pc[:,  :, 4:5], pc[:,  :,  6:8], pc[:,  :, 9:10], pc[:,:,3:4]), axis=2)  # [seq, n_p, 9] 
        
        # if not keep_labels
        else:
            # pc = np.concatenate((pc[:, :, :2], pc[:, :, 10:11], pc[:, :, 2:3], pc[:, :, 12:], pc[:, :, 4:5], pc[:, :, 6:8], pc[:, :, 9:10]), axis=2)  # [n_p, 9]   
            pc = np.concatenate((pc[:, :, :2], pc[:,  :, 10:11], pc[:,  :, 2:3], pc[:,  :, 4:5], pc[:,  :,  6:8], pc[:,  :, 9:10]), axis=2)  # [seq, n_p, 8] 

        if self.get_views:
            return pc, labels, filename, ids, n_unique_pts, xyz, views, point2pixel
        else:
            return pc, labels, filename, ids, n_unique_pts, xyz


    def prepare_data(self,
                     point_file):
        """
        dimensions of point cloud :
        0 - x
        1 - y
        2 - z
        3 - label
        4 - I
        5 - red
        6 - green
        7 - blue
        8 - nir
        9 - ndvi
        10 - hag (height above ground)
        11 - point ID

        :param point_file: path of file

        :return: 
        pc_samp: torch.Tensor [n_samp, points, dims]
        xyz: torch.Tensor [points, 3] real coordinates float64
        ids: torch.Tensor unique ids per point
        n_unique_points

        """
        xyz = torch.Tensor()

        with open(point_file, 'rb') as f:
            pc = torch.load(f)

        pc = pc[~np.isin(pc[:, 3], self.remove_classes)]

        if self.return_xyz:
            xyz = pc[:, :3].clone()
        
        if self.use_tile_ids:
            ids_ini = pc[:, 11]
            ids_ini = torch.Tensor.int(ids_ini)
        else:
            # generate id per point
            ids_ini = self.create_morton_hash(pc[:, :3].squeeze(0), self.max_id)  # Generate unique IDs for each point
            
        # Get number of unique points in subtile
        n_unique_points = len(ids_ini)

        # normalize x and y between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        
        # Normalize z between 0 and 1
        if self.max_z:
            # Local Z norm
            pc_localZ = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min())
            pc_localZ = pc_localZ.unsqueeze(1)

            # Global Z normalization
            pc[:, 2] = pc[:, 2]/ self.max_z
            pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

            pc = torch.cat((pc, pc_localZ), dim=1)
        else:
            pc[:, 2] = (pc[:, 2] - pc[:, 2].min()) / (pc[:, 2].max() - pc[:, 2].min())

        # make sure intensity and color are in range (0,1)
        pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)  
        pc[:, 6] = np.clip(pc[:, 6], 0.0, 1.0)
        pc[:, 7] = np.clip(pc[:, 7], 0.0, 1.0)

        # pc = np.float32(pc)

        # get 2D views
        if self.get_views:
            views, pixel_ix = self.create_2d_cv_views(pc)

        if n_unique_points < self.n_points:
            n_sample_points = n_unique_points

            if n_unique_points < self.n_points:
                n_sample_points = self.n_points
        else:
            n_sample_points=self.n_points

        # reformat point cloud into sampled sequence tensor
        pc_samp, ids, indices = get_sampled_sequence(torch.FloatTensor(pc), torch.Tensor.int(ids_ini), n_sample_points)
        
        if self.return_xyz:
            xyz=xyz[indices,:]
        
        if self.get_views:
            return pc_samp, xyz, ids, n_unique_points, views, pixel_ix

        return pc_samp, xyz, ids, n_unique_points


    def create_2d_cv_views(self, pc, image_size=256, sample_size=None):
         
        # normalize axes between 0 and 1
        pc[:, 0] = (pc[:, 0] +1) / 2
        pc[:, 1] = (pc[:, 1] +1) / 2

        # sample pc
        if sample_size and len(pc) > sample_size:
            pc = pc[np.random.choice(len(pc), sample_size, replace=False)]
 
        point_ids = pc[:, 11] # Store point IDs

        # Prepare coordinate projections
        coords = np.stack([
            (pc[:, [0, 1, 2]]),  # XY view -> use Z as value
            (pc[:, [0, 2, 1]]),  # XZ view -> use Y as value
            (pc[:, [1, 2, 0]])   # YZ view -> use X as value
        ], axis=0)  # Shape: (3, N, 3)

        views = np.zeros((3, image_size, image_size), dtype=np.float32)
        
        # Preallocate mapping: [3, N, 3] → for each view: [point_id, x, y]
        mappings = np.empty((3, coords.shape[1], 3), dtype=np.int32)  # [view, N, (id, x, y)]

        for i in range(3):
            x = ((coords[i, :, 0]) * (image_size - 1)).astype(int)
            y = (((1 - coords[i, :, 1])) * (image_size  - 1)).astype(int)
            values = coords[i, :, 2] * 255

            # # Use np.maximum to preserve max depth value if overlapping
            np.maximum.at(views[i], (y, x), values)

            # Fill mapping tensor
            mappings[i, :, 0] = point_ids        # point ID
            mappings[i, :, 1] = x                # x
            mappings[i, :, 2] = y                # y

        return views, mappings  # views: [3, H, W], mappings: [3, N, 3]
    
    def create_unique_ids(self, xyz):
        """
        Create unique IDs for each point based on its coordinates.
        Args:
            xyz (torch.Tensor): Tensor of shape [n_points, 3] containing the x, y, z coordinates of each point.
        Returns:
            torch.Tensor: Tensor of unique IDs for each point.
        """
        # Scale coordinates to integers to avoid floating-point precision issues
        scale_factor = 1e6
        scaled_xyz = (xyz * scale_factor).to(dtype=torch.int64)
        
        # Combine x, y, z into a single unique ID using hashing
        unique_ids = torch.stack([scaled_xyz[:, 0], scaled_xyz[:, 1], scaled_xyz[:, 2]], dim=1)
        unique_ids = torch.sum(unique_ids * torch.tensor([1, 2**20, 2**40], dtype=torch.int64, device=xyz.device), dim=1)
        return unique_ids
    

    def encode_xyz_uint32(x, y, z, scale=10):
        """Codifica (x, y, z) en un uint32 con precisión de 10 cm en XY y 25 cm en Z"""
        x_idx = int(x * scale) & 0x3FFF  # 14 bits
        y_idx = int(y * scale) & 0x3FFF  # 14 bits
        z_idx = int(z / 0.25) & 0xF      # 4 bits (0-15)

        return (x_idx << 18) | (y_idx << 4) | z_idx  # uint32
    




# ############################################ Barlow Twins Dataset #################################################


class BarlowTwinsDatasetWithGround(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=True):
        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, dims], labels:[n_points], filename:[n_points]
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)

        # pc size [2048,14]
        if self.task == 'segmentation':
            labels = self.get_labels_segmnetation(pc)
        else:  # elif self.task == 'classification':
            labels = self.get_labels_classification(self.classes_mapping[self.files[index]])
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)
        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]

        # remove not classified points
        # pc = pc[pc[:, 3] != 1]
        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 11] == 1]  # should be flag of position 11

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        # normalize height
        # pc[:, 2] = pc[:, 10] / 2  # HAG
        # pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmnetation(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> Other infrastructure
        1 -> power lines
        2 -> med-high veg
        3 -> low vegetation
        4 -> ground

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]

        segment_labels[segment_labels == 15] = 100  # tower
        segment_labels[segment_labels == 14] = 100  # lines
        segment_labels[segment_labels == 18] = 100  # other towers

        segment_labels[segment_labels == 4] = 200  # med veg
        segment_labels[segment_labels == 5] = 200  # high veg
        segment_labels[segment_labels == 1] = 200  # not classified
        segment_labels[segment_labels == 3] = 300  # low veg

        segment_labels[segment_labels == 2] = 400  # ground
        segment_labels[segment_labels == 8] = 400  # ground
        segment_labels[segment_labels == 7] = 400  # ground
        segment_labels[segment_labels == 13] = 400  # ground

        segment_labels[segment_labels < 100] = 0  # infrastructure
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        return labels

    @staticmethod
    def get_labels_classification(point_cloud_class):
        """
        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)

        :param point_cloud_class:
        :return:
        """
        labels = point_cloud_class

        return labels


class BarlowTwinsDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    # 0 -> no tower
    # 1 -> tower
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=False,
                 no_labels=False):

        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.no_labels = no_labels
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, dims], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)
        # pc size [2048,14]
        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:  # if self.task == 'classification':
            labels = self.get_labels_classification(pc)
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)  # pc size [2048,9]

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]
            # pc = pickle.load(f).astype(np.float32)  # [17434, 14]

        # remove not classified points
        pc = pc[pc[:, 3] != 1]

        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 11] == 1]  # should be flag of position 11

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        # Z normalization up to 50 m
        pc[:, 2] = pc[:, 10] / 2  # HAG
        pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> all infrastructure
        1 -> pylon
        2 -> power lines
        3 -> low veg
        4 -> med-high vegetation
        5 -> roofs and objects over roofs

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]

        segment_labels[segment_labels == 15] = 100  # tower
        segment_labels[segment_labels == 14] = 200  # lines
        segment_labels[segment_labels == 18] = 100  # other towers

        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 400  # med veg
        segment_labels[segment_labels == 5] = 400  # high veg
        segment_labels[segment_labels == 1] = 400  # undefined

        segment_labels[segment_labels == 6] = 500  # roof
        segment_labels[segment_labels == 17] = 500  # objects over roofs

        segment_labels[segment_labels < 100] = 0  # all infrastructure
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        return labels

    @staticmethod
    def get_labels_classification(pointcloud):
        """
        Classification labels:
        0 -> Only vegetation in point cloud
        1 -> Power lines and other towers
        2 -> Buildings and Infrastructure

        :param pointcloud:
        :return:
        """
        # if not self.no_labels:
        unique, counts = np.unique(pointcloud[:, 3].cpu().numpy().astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))

        # power lines and other towers
        if 15 in dic_counts.keys():
            labels = 1
        elif 14 in dic_counts.keys():
            labels = 1
        elif 18 in dic_counts.keys():
            labels = 1
        # buildings and infrastructures
        elif 6 in dic_counts.keys():
            labels = 2
        elif 16 in dic_counts.keys():
            labels = 2
        elif 17 in dic_counts.keys():
            labels = 2
        elif 19 in dic_counts.keys():
            labels = 2
        elif 22 in dic_counts.keys():
            labels = 2
        # vegetation
        else:
            labels = 0

        return labels


##### -------------------------------------------- DALES DATASET ---------------------------------------------------

class DalesDataset(data.Dataset):

    def __init__(self,
                 files,
                 task='segmentation',
                 number_of_points=None,
                 fixed_num_points=True,
                 get_centroids=False,
                 check_pts_files=False,
                 use_all_labels=False):
        
        self.files = files
        self.task = task
        self.number_of_points = number_of_points
        self.fixed_num_points = fixed_num_points
        self.get_centroids = get_centroids
        self.use_all_labels = use_all_labels

        if check_pts_files:
            self.files = self._check_files()
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return:
                pc: float Tensor [n_points, 5]
                labels: float Tensor
                filename: str
        """
        filename = self.files[index]
        pc = self.prepare_data(filename)

        if self.use_all_labels:
            labels = self.get_all_labels(pc[:,3])
        else:
            labels = self.get_labels(pc[:,3])

        pc = np.concatenate((pc[:, :3], pc[:, 4:6]), axis=1)

        # Get cluster centroids
        if self.get_centroids:
            centroids = self.get_cluster_centroid(pc)
            return pc, labels, filename, centroids

        return pc, labels, filename

    def _check_files(self):
        checked_paths = []
        counter = 0
        for point_file in self.files:
            with open(point_file, 'rb') as f:
                pc = torch.load(f).numpy()

            if pc.shape[0] > 1024:
                checked_paths.append(point_file)
            else:
                print(point_file)
                counter += 1
        print(f'Number of discarded files (<1024p): {counter}')
        return checked_paths

    def prepare_data(self,
                     point_file):
        """
        point cloud dims: x, y, z, classification, return_num, num_of_returns

        :param point_file: str path to file
        :return: pc tensor
        """

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]

        if self.fixed_num_points and pc.shape[0] > self.number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], self.number_of_points)
            pc = pc[sampling_indices, :]
        
        # duplicate points if needed
        elif self.fixed_num_points and pc.shape[0] < self.number_of_points:
            points_needed = self.number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # Normalize x,y between -1 and 1
        pc = self.pc_normalize_neg_one(pc)
        # normalize z between 0 1
        pc[:, 2] = pc[:, 2] / 200.
        pc[:, 2] = np.clip(pc[:, 2], 0., 1.0)
        # normalize return number between 0 1
        pc[:, 4] = pc[:, 4] / 7.
        # normalize number of returns between 0 1
        pc[:, 5] = pc[:, 5] / 7.

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1  # x
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1  # y
        return pc

    @staticmethod
    def get_cluster_centroid(pc):
        """
        :param pc: point cloud (n_p, dims, n_clusters) i.e.(2048,5,9)
        :return:
        """
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        centroids = np.stack([mean_x, mean_y], axis=0)
        return centroids

    @staticmethod
    def get_labels(seg_labels):
        """
        Inputs segmentation labels: categories: ground(1), vegetation(2), cars(3), trucks(4), power lines(5), fences(6), poles(7) and buildings(8).

        Output segmentation labels:
        0 -> other
        1 -> ground
        2 -> poles
        3 -> power lines

        :param pc: [n_points]
        :return labels: points with categories to segment or classify
        """

        conditions = [
            (seg_labels == 0) |  (seg_labels == 6) | (seg_labels == 8)  |  (seg_labels == 3) |  (seg_labels == 4) | (seg_labels == 2),
            (seg_labels == 1),
            (seg_labels == 7),
            (seg_labels == 5),
            
        ]
        choices = [0, 1, 2, 3]
        seg_labels = np.select(conditions, choices)

        return seg_labels 
    
    @staticmethod
    def get_all_labels(seg_labels):
        """
        Inputs segmentation labels: categories: ground(1), vegetation(2), cars(3), trucks(4), power lines(5), fences(6), poles(7) and buildings(8).

        Output segmentation labels:
        0 -> other
        1 -> ground
        2 -> poles
        3 -> power lines
        4 -> veg
        5 -> buildings
        6 -> cars and trucks

        :param pc: [n_points]
        :return labels: points with categories to segment or classify
        """

        conditions = [
            (seg_labels == 0) | (seg_labels == 6),
            (seg_labels == 1),
            (seg_labels == 7),
            (seg_labels == 5),
            (seg_labels == 2),
            (seg_labels == 8),
            (seg_labels == 3) | (seg_labels == 4),
        ]
        choices = [-1, 0, 1, 2, 3, 4, 5]
        seg_labels = np.select(conditions, choices)

        return seg_labels 


class DalesSamplingDataset(DalesDataset):

    def __init__(self,
                 files,
                 task='segmentation',
                 number_of_points=None,
                 fixed_num_points=True,
                 get_centroids=False,
                 check_pts_files=False,
                 keep_labels=False,
                 use_all_labels=False):
        
        self.files = files
        self.task = task
        self.n_points = number_of_points
        self.fixed_num_points = fixed_num_points
        self.get_centroids = get_centroids
        self.last_id = 0  # Inicializar el último ID usado
        self.keep_labels = keep_labels
        self.use_all_labels = use_all_labels

        if check_pts_files:
            self.files = self._check_files()
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return:
                pc: float Tensor [n_points, 5]
                labels: float Tensor
                filename: str
        """
        filename = self.files[index]
        pc, ids = self.prepare_data(filename)

        pc_labels = np.reshape(pc, (-1, pc.shape[2]))

        if self.use_all_labels:
            labels = self.get_all_labels(pc_labels[:,3])
        else:
            labels = self.get_labels(pc_labels[:,3])

        # for plotting
        if self.keep_labels:
            # Extract segment labels (the 3rd dimension slice) -> shape [6, 8000]
            ini_labels = pc[:, :, 3].view(-1).numpy() 
            if self.use_all_labels:
                seg_labels = self.get_all_labels(ini_labels)
            else:
                seg_labels = self.get_labels(ini_labels)
            seg_labels = torch.FloatTensor(seg_labels)

            # Assign the modified segment labels back to the original tensor
            pc[:, :, 3] = seg_labels.view(-1, pc.shape[1])
            pc = np.concatenate((pc[:, :, :3], pc[:, :, 4:6], pc[:, :, 3].unsqueeze(2)), axis=2)
        
        else:
            pc = np.concatenate((pc[:, :, :3], pc[:, :, 4:6]), axis=2)

        # Get cluster centroids
        if self.get_centroids:
            centroids = self.get_cluster_centroid(pc)
            return pc, labels, filename, centroids

        return pc, labels, filename, ids
    
    def prepare_data(self,
                     point_file):
        """
        point cloud dims: x, y, z, classification, return_num, num_of_returns

        :param point_file: str path to file
        :return: pc tensor
        """

        with open(point_file, 'rb') as f:
            pc_ = torch.load(f).numpy()  # [points, dims]

        # Normalize x,y between -1 and 1
        pc = self.pc_normalize_neg_one(pc_)
        # normalize z between 0 1
        pc[:, 2] = pc[:, 2] / 200.
        pc[:, 2] = np.clip(pc[:, 2], 0., 1.0)
        # normalize return number between 0 1
        pc[:, 4] = pc[:, 4] / 7.
        # normalize number of returns between 0 1
        pc[:, 5] = pc[:, 5] / 7.

        # generate unique id per point
        num_points = pc.shape[0]
        ids = torch.Tensor(np.arange(0, num_points))

        # reformat point cloud into sampled sequence tensor
        pc_samp, ids, indices = get_sampled_sequence(torch.FloatTensor(pc), ids, self.n_points)  # [batch, n_points, n_feat]

        # pc_samp = pc[np.newaxis,:,:]

        return pc_samp, ids