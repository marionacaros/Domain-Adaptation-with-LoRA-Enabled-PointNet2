import glob
import numpy as np
import torch
from progressbar import progressbar
import os
import laspy
import argparse
import pylas


def store_las(array, output_dir, f_name='predictions.las'):
    # Store LAS
    # 1. Create a new header
    las = pylas.create(point_format_id=8, file_version="1.4")

    # header = laspy.LasHeader(point_format=3, version="1.4")
    # header.offsets = np.min(array, axis=0)
    # # print(f'Offset: {np.min(array, axis=0)}')
    # header.scales = np.array([0.001, 0.001, 0.001])
    # header.add_extra_dim(laspy.ExtraBytesParams(name="ndvi", type=np.int32))

    # 2. Create a Las
    # las = laspy.LasData(header)
    las.x = array[:, 0]
    las.y = array[:, 1]
    las.z = array[:, 10]
    las.classification = array[:, 3].astype(np.uint8)
    las.intensity = array[:, 4] * 5000.0
    las.red = array[:, 5] * 65536.0
    las.green = array[:, 6] * 65536.0
    las.blue = array[:, 7] * 65536.0
    las.nir = array[:, 8] * 65536.0

    # Define the output file path
    las.write(os.path.join(output_dir, f_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/QPcotLIDev01/LiDAR/DL_preproc/LAS_filtered_ground/train',
                        help='output directory')
    parser.add_argument('--input_dir', type=str,
                        default='/mnt/QPcotLIDev01/LiDAR/DL_preproc/100x100_s50_p8k/train')
    args = parser.parse_args()

    # list of tiles
    pt_files = glob.glob(os.path.join(args.input_dir, '*.pt'))
    tiles = set([path_f.split('_')[-2].split('.')[0] for path_f in pt_files])

    output_dir = args.output_dir

    for tile_name in progressbar(tiles):
        print('---------------------------------------------------------')
        print(f"Tile: {tile_name}")
        tile = np.empty((0, 12))

        pt_files = glob.glob(os.path.join(args.input_dir, '*' + tile_name + '*.pt'))

        for point_file in pt_files:
            with open(point_file, 'rb') as f:
                pc = torch.load(f).numpy()

            tile = np.concatenate((tile, pc), axis=0)

        # store LAS
        store_las(tile, output_dir, f_name=tile_name + '.las')

