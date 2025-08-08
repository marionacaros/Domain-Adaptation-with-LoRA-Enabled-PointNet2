import argparse
from tqdm import tqdm
import sys
sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from utils.utils import *
import time
import random
import multiprocessing
import logging
import csv
from proc_adjacent_tiles import *

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    """
    Last version of preporcessing
    All LAS files must have Distance attribute
    1 - Remove ground (categories 2, 8) if size > n_points (i.e. 8000)
    2 - Split point cloud into windows of size W_SIZE. (i.e. 80 x 80)
    3 - Add HAG and NDVI
    4 - Remove points of target classes if number of points < 15
    
    Output is tensor containing: x,y,z,class,I,R,G,B,NIR,NDVI,HAG
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str, help='output directory where processed files are stored')
    parser.add_argument('--n_points', type=int, default=8000)
    parser.add_argument('--w_size', type=int, default=80)
    parser.add_argument('--stride', default=10)
    parser.add_argument('--max_height', type=float, default=200.0)
    parser.add_argument('--max_intensity', type=float, default=5000.0)

    args = parser.parse_args()
    start_time = time.time()

    global OUT_PATH, W_SIZE, MAX_Z, MAX_I, STORE_DOUBLE, DATASET_NAME, MAX_N_PTS, NUM_CPUS, N_POINTS, MAX_H, STRIDE
    NUM_CPUS = 16

    STORE_DOUBLE = True
    OUT_PATH = args.out_path
    DATASET_NAME = 'Z31' #OUT_PATH.split('/')[-1].split('_')[0]
    N_POINTS = args.n_points
    W_SIZE = args.w_size
    MAX_H = args.max_height
    MAX_I = args.max_intensity
    STRIDE = W_SIZE - args.stride
    MAX_N_PTS = 400000
    files = []
    filter_tiles = [] 

    if filter_tiles:
        for tile in filter_tiles:
            files.append(os.path.join(args.in_path, '*' + str(tile) + '*.las'))
    else:
        files = glob.glob(os.path.join(args.in_path, '*.las')) 

    logging.info(f'DATASET_NAME: {DATASET_NAME}')
    logging.info(f'Stride: {args.stride} m. moving {STRIDE} m.')
    logging.info(f'Output path: {args.out_path}')
    logging.info(f'Num of files: {len(files)}')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    files = sorted(files)

    # Multiprocessing
    parallel_proc(files, num_cpus=NUM_CPUS)
    # for file in tqdm(files):
        # split_pointcloud(file)

    print("--- Dataset preprocessing time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))

    ####################################################################################################################################


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)
    # Use tqdm with imap_unordered
    with tqdm(total=len(files_list)) as pbar:
        for _ in p.imap_unordered(split_pointcloud, files_list):
            pbar.update(1)  # Update progress bar for each completed task
    p.close()
    p.join()


def parallel_adj_pairs(pairs, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)
    # Use tqdm with imap_unordered
    with tqdm(total=len(pairs)) as pbar:
        for _ in p.imap_unordered(split_overlap, pairs):
            pbar.update(1)  # Update progress bar for each completed task
    p.close()
    p.join()


def get_ndvi(nir, red):
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c


def split_pointcloud(f):
    """
    1 - Remove ground (categories 2, 8, 13)
    2 - Split point cloud into windows of size W_SIZE.
    3 - Add HAG and NDVI

    :param f: file path
    """

    f_name = f.split('/')[-1].split('.')[0]  # /mnt/Lidar_M/DEMO_Productes_LIDARCAT3/LAS_def/505679.las

    data_f = laspy.read(f)

    # check file is not empty
    if len(data_f.x) > 0:
        if DATASET_NAME != 'EMP':
            try:
                pc = np.vstack((data_f.x,
                                data_f.y,
                                data_f.z,
                                data_f.classification,  # 3
                                data_f.intensity / MAX_I,  # 4
                                data_f.red / 65536.0,  # 5
                                data_f.green / 65536.0,  # 6
                                data_f.blue / 65536.0,  # 7
                                data_f.nir / 65536.0,  # 8
                                np.zeros(len(data_f.x)),  # 9  NDVI
                                data_f.Distance / 1000,  # 10  HAG
                                np.arange(len(data_f.x)) # 11  ID
                                ))
            except AttributeError as e:
                print(e)
                print(f)
                return
        else:
            # Alt Emporda data
            try:
                pc = np.vstack((data_f.x,
                                data_f.y,
                                data_f.z,
                                data_f.classification,  # 3
                                data_f.intensity / MAX_I,  # 4
                                data_f.red / 65536.0,  # 5
                                data_f.green / 65536.0,  # 6
                                data_f.blue / 65536.0,  # 7
                                data_f.nir / 65536.0,  # 8
                                ))
                data_f = laspy.read('/dades/LIDAR/towers_detection/LAS_CAT3_HAG/' + f_name + '.las')
                pc2 = np.vstack((np.zeros(len(data_f.x)),  # 9  NDVI
                                 data_f.HeightAboveGround / MAX_H,  # 10  HAG
                                np.arange(len(data_f.x)) # 11  ID
                                 ))
                pc = np.concatenate((pc, pc2), axis=0)
            except Exception as e:
                print(e)
                print(f)
                return
        
        pc = pc.transpose()

        if DATASET_NAME=='COSTA':
            # Reduce to 60% of original points
            fraction = 0.6
            sampled_indices = np.random.choice(pc.shape[0], int(pc.shape[0] * fraction), replace=False)
            pc = pc[sampled_indices,:]

        # Filter unwanted classes
        invalid_classes = {7, 11, 24, 13, 30, 31, 99, 102, 103, 104, 105, 106, 135}
        pc = pc[~np.isin(pc[:, 3], list(invalid_classes))]

        # Filter out invalid distance values
        pc = pc[pc[:, 10] >= 0]

        # add column with id based on index
        pc = np.column_stack((pc, np.arange(pc.shape[0])))  # New column as the last one

        if pc.shape[0] > 0:  # Verifica que pc tenga datos y la columna 10 exista
            max_distance = pc[:, 10].max()
            # min_distance = pc[:, 10].min()
        else:
            print(f"Warning: No valid points found in {f_name}, skipping max/min computation.")
            max_distance = None
            # min_distance = None
            return

        # Guardar en CSV
        # OUTPUT_CSV = "/dades/LIDAR/towers_detection/Z31_distance_values.csv"
        # with open(OUTPUT_CSV, mode="a", newline="") as file:
            # writer = csv.writer(file)
            # writer.writerow([f_name, min_distance, max_distance, pc[:, 0].min(), pc[:, 0].max(), pc[:, 1].min(), pc[:, 1].max(), pc[:, 2].min(), pc[:, 2].max()])

        # Check max height is above 5 meters
        if max_distance > 5:

            # key DEM points to ground
            indices = np.where(pc[:, 3] == 8.)
            pc[indices, 3] = 2.

            # if B29 change category of wind turbine from other towers
            if DATASET_NAME == 'B29':
                indices = np.where(pc[:, 3] == 18.) # other towers -> wind turbines
                pc[indices, 3] = 29.

            pc = pc.transpose()

            i_w = 0
            x_min, y_min = pc[0].min(), pc[1].min()
            x_max, y_max = pc[0].max(), pc[1].max()

            for y in range(round(y_min), round(y_max - (STRIDE-1)), STRIDE):
                bool_w_y = np.logical_and(pc[1] < (y + W_SIZE), pc[1] > y)

                for x in range(round(x_min), round(x_max - (STRIDE-1)), STRIDE):
                    bool_w_x = np.logical_and(pc[0] < (x + W_SIZE), pc[0] > x)
                    bool_w = np.logical_and(bool_w_x, bool_w_y)
                    i_w += 1

                    if any(bool_w):
                        if pc[:, bool_w].shape[1] >= N_POINTS:  # check size (ground + other)
                            pc_w = pc[:, bool_w]
                            pc_w = pc_w.transpose()

                            # Check max height is above 5 meters
                            if pc[:, 10].max() > 5:

                                # store torch file
                                fileName = 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)

                                # count number of points per class
                                unique, counts = np.unique(pc_w[:, 3].astype(int), return_counts=True)
                                dic_counts = dict(zip(unique, counts))

                                if 35 in dic_counts.keys():
                                    fileName = 'crane_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)

                                elif 29 in dic_counts.keys():
                                    if dic_counts[29] >= 15:  # check that number of points is >= 15
                                        fileName = 'windturbine_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                                    else:
                                        pc = pc[pc[:,3]!=29]

                                elif 15 in dic_counts.keys():
                                    if dic_counts[15] >= 15: 
                                        fileName = 'tower_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                                    else:
                                        pc = pc[pc[:,3]!=15]

                                elif 14 in dic_counts.keys():
                                    if dic_counts[14] >= 15:
                                        fileName = 'lines_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                                    else:
                                        pc = pc[pc[:,3]!=14]

                                elif 18 in dic_counts.keys():
                                    if dic_counts[18] >= 15:
                                        fileName = 'othertower_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                                    else:
                                        pc = pc[pc[:,3]!=18]

                                # Check how many points are not ground
                                len_pc = pc_w[pc_w[:, 3] != 2].shape[0]
                                if len_pc > 1000:

                                    pc_w = preprocessing(pc_w, max_h=MAX_H, n_points=N_POINTS, max_points=MAX_N_PTS)  # [points, 12]

                                    if pc_w.shape[0] >= N_POINTS:
                                        out_fileName = os.path.join(OUT_PATH, fileName)

                                        if STORE_DOUBLE:
                                            torch.save(torch.DoubleTensor(pc_w), out_fileName + '.pt')
                                        else:
                                            torch.save(torch.FloatTensor(pc_w), out_fileName + '.pt')


if __name__ == '__main__':
    main()

    
