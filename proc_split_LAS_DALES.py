import argparse
from utils.utils import *
import logging
import time
import multiprocessing

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

global SAVE_PATH, W_SIZE, LAS_files_path


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(split_dataset_windows, files_list, 1),
                         max_value=len(files_list)):  # redirect_stdout=True)
        pass
    p.close()
    p.join()


def split_dataset_windows(file):
    i_w = 0
    name_f = file.split('/')[-1].split('.')[0]
    las_pc = laspy.read(file)

    pc = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                    las_pc.return_number, las_pc.number_of_returns))
    # remove undefined
    pc = pc[:, pc[3] != 0]
    # get coords
    x_min, y_min, z_min = pc[0].min(), pc[1].min(), pc[2].min()
    x_max, y_max, z_max = pc[0].max(), pc[1].max(), pc[2].max()

    # split point cloud
    for y in range(round(y_min), round(y_max), W_SIZE[1]):
        bool_w_y = np.logical_and(pc[1] < (y + W_SIZE[1]), pc[1] > y)

        for x in range(round(x_min), round(x_max), W_SIZE[0]):
            bool_w_x = np.logical_and(pc[0] < (x + W_SIZE[0]), pc[0] > x)
            bool_w = np.logical_and(bool_w_x, bool_w_y)
            i_w += 1

            if any(bool_w):
                pc_w = pc[:, bool_w]
                if pc_w.shape[1] > 0:

                    # store torch file
                    fileName = name_f + '_w' + str(i_w)
                    torch_file = os.path.join(SAVE_PATH, fileName) + '.pt'
                    torch.save(torch.FloatTensor(pc_w.transpose()), torch_file)

    print(f'Stored windows of block {name_f}: {i_w}')

    # ------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/home/m.caros/work/DALES/dales_25x25',
                        help='output folder where processed files are stored')
    parser.add_argument('--LAS_files_path', type=str, default='/home/m.caros/work/DALES/dales_las')
    parser.add_argument('--w_size', default=[25, 25])
    start_time = time.time()

    args = parser.parse_args()

    W_SIZE = args.w_size

    for dir in ['/train', '/test']:

        LAS_files_path = args.LAS_files_path + dir
        files = glob.glob(LAS_files_path + '/*.las')
        SAVE_PATH = args.out_path + dir

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        parallel_proc(files, num_cpus=16)
        print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
