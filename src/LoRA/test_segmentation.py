import argparse
import os.path
import time
import torch
import sys
import logging
import json
sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import CAT3SamplingDataset
from src.LoRA.models.pointnet2_ss import *
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
import torch.nn.functional as F
from src.config import *
from sklearn.metrics import confusion_matrix
import csv
from src.LoRA.models.pointnet2_msg import *


N_SAMPLES = 16
global DATASET, DEVICE


def load_model(model_checkpoint, n_classes):
    checkpoint = torch.load(model_checkpoint)

    # model
    model = PointNet2(num_classes=n_classes, num_feat=8).to(DEVICE)
    # model = PointNet2msg(num_classes=n_classes, num_feat=8).to(DEVICE)

    model = model.apply(weights_init)
    model = model.apply(inplace_relu)

    model.load_state_dict(checkpoint['model'])
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']
    logging.info('---------- Checkpoint loaded ----------')

    model_name = model_checkpoint.split('/')[-1].split('.')[0].split(':')[0]
    logging.info(f'Model name: {model_name} ')
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    logging.info(f"Total Trainable Params: {total_params}")

    return model


# @track_emissions()
def test(output_dir,
         number_of_workers,
         model,
         list_files,
         n_points,
         targets_arr,
         preds_arr, 
         plot_objects=False,
         n_classes=5):
    
    # net to eval mode
    model = model.eval()
    use_windturbine=False

    if n_classes>3:
        use_windturbine=True

    # Initialize dataset
    test_dataset = CAT3SamplingDataset(task='segmentation',
                                       n_points=n_points,
                                       files=list_files,
                                       return_coord=False,
                                       tile_ids=True,
                                       use_z=True,
                                       keep_labels=False,
                                       use_windturbine=use_windturbine,
                                       check_files=True,
                                       max_z = None)
    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    preds_tile = torch.zeros((100000000, n_classes), dtype=torch.int64)
    labels_tile = np.zeros((100000000, n_classes), dtype=int)

    with torch.no_grad():
        for i, data in enumerate(progressbar(test_dataloader, )):

            pointcloud, targets, file_name, ids, n_o_points, _ = data  # [batch, n_samp, n_sampled_p, dims]

            ids_ini = ids.squeeze(0).cpu().to(torch.int64)
            file_name = file_name[0].split('/')[-1].split('.')[0]  # pc_B23_ETehpt421601_w212
            # tile = file_name.split('_')[-2]

            targets = targets.view(-1).cpu()
            targets_1hot = torch.nn.functional.one_hot(targets, num_classes=n_classes).numpy()
            # add labels to labels_tile
            labels_tile[ids_ini.numpy()] = targets_1hot

            pc = pointcloud.squeeze(0).to(DEVICE)  # [n_samp, n_points, dims]
            dims = pc.shape[2]

            if n_points < pc.shape[0] * pc.shape[1] < N_SAMPLES * n_points:
                # duplicate points with different samplings
                pc2, ids2, _ = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
                pc = torch.cat((pc, pc2), dim=0)
                ids = torch.cat((ids_ini, ids2), dim=0)
                dupli = True
            else:
                ids = ids_ini

            # -----------------------------------------------------------
            # transpose for model
            pc = pc.transpose(2, 1)

            # get logits from model
            preds, _ = model(pc[:,:,:])  # [batch, n_class, n_points]
            preds = preds.contiguous().view(-1, n_classes)

            # get predictions
            preds = F.softmax(preds, dim=1).data.max(1)[1]
            preds = preds.view(-1).to(dtype=torch.int64)  # [n_points]

            pc = pc.transpose(1, 2)
            pc = pc.view(-1, dims)  # [n_points, dims] [32000, 7]

            # one hot encoding
            preds_1hot = torch.nn.functional.one_hot(preds.cpu(), num_classes=n_classes).to(torch.int64)  # [points, classes]

            # --- Update the preds_tile matrix ---
            preds_tile.scatter_add_(0, ids.unsqueeze(1).expand(-1, n_classes), preds_1hot)  # [6x10⁹, N_CLASSES]

            # debug and plot
            preds_all = preds.cpu().numpy()
            if plot_objects and (4 in set(preds_all)): # or 1 in set(preds_all)):
                if 4 in set(preds_all):
                    obj_class=4
                else:
                    obj_class=1
                # pc_2=pc[:n_points, :].cpu().numpy(),
                # preds_2 = preds_all[:n_points]
                pc_np=pc.cpu().numpy()
                plot_two_pointclouds_z(pc_np,
                                        pc_np,
                                        labels=pc_np[:,-1],
                                        labels_2=preds_all,
                                        name=file_name + '_labels_preds',
                                        path_plot=os.path.join(output_dir, 'plots/predictions_windturbine'),
                                        point_size=2,
                                        target_class=obj_class)

            # ---------------------------------- end voting loop --------------------------------------------------------
            del preds, pc
            torch.cuda.empty_cache()

    final_class_tile = torch.argmax(preds_tile, dim=1).numpy()
    labels_tile = np.argmax(labels_tile, axis=1)

    # Find the indices of rows with non-zero values
    mask = (preds_tile != 0).any(dim=1)
    non_zero_ix = torch.where(mask)[0]

    if len(non_zero_ix) > 0:
        # Get only points that went through the model
        final_class_tile=final_class_tile[mask]
        labels_tile=labels_tile[mask]

    return labels_tile, final_class_tile, non_zero_ix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='src/LoRA/metrics',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points per point cloud')
    parser.add_argument('--model_checkpoint', type=str,
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-10-16:49e2.pt',
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-14-13:19B29msge2.pt',
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-21-10:08B29e2.pt',
                        default='src/LoRA/checkpoints_lidarcat/seg_02-24-15:52B29.pt',
                        # default='src/LoRA/checkpoints_lidarcat/seg_04-16-11:14B29CE.pt',
                        help='models checkpoint path')
    parser.add_argument('--in_path', type=str,
                        # default='train_test_files/B29_80x80')
                        default='train_test_files/RIB_smallLoRA_80x80')

    parser.add_argument('--device', type=str, default='cuda', help='device to be used, cuda or cpu')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    start_time = time.time()
    
    DEVICE = args.device
    logging.info(f'DEVICE: {DEVICE}')

    DATASET =  'RIB' #args.in_path.split('/')[-1]
    logging.info(f'DATASET: {DATASET}')
        
    n_classes = args.num_classes
    logging.info(f'Number of classes: {n_classes}')

    # load model
    model = load_model(args.model_checkpoint, n_classes)
    MODEL_NAME = args.model_checkpoint.split('/')[-1].split('.')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Data B29
    with open(os.path.join(args.in_path, 'test_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Define file path
    file_path = os.path.join(args.output_dir, 'IoU-results-%s.csv' % MODEL_NAME)

    # store in csv file
    with open(file_path, 'a') as fid:
        writer = csv.writer(fid)

        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            # Write the header if the file is empty
            # writer.writerow(["Tile", "Surrounding IoU", "Tower IoU", "Lines IoU", "Wind IoU", "Ground IoU", "Mean IoU", "points surr", "points tower", "points lines", "points wind","points ground"])
            writer.writerow(["Tile", "Surrounding IoU", "Tower IoU", "Lines IoU", "Wind IoU", "Mean IoU", "points surr", "points tower", "points lines", "points wind"])

    # list of tiles
    # tiles = set([path_f.split('_')[-2].split('.')[0] for path_f in test_files])
    # tiles = list(tiles)

    tiles=["pt438656", "pt438652", "pt438658","pt440652", "pt436662", "pt436656"] # RIB

    # SELEC TILES
    # tiles=['ETOHc000023', 'ETOHc000026', 'ETOHc000028', 'ETOHc000029', 'ETOHc000030', 'ETOHc000031', 'ETOHc000044', 'ETOHc000045', 'ETOHc000046', 
    #        'ETOHc000049', 'ETOHc000024','ETOHc000033', 'ETOHc000156',"ETOHc000169","ETOHc000244","ETOHc000239", "ETOHc000174", "ETOHc000252"]
    tiles.sort()

    logging.info(f"Number of tiles: {len(tiles)}")

    dataset = DATASET + '_8kp_' + MODEL_NAME
    print(f"Dataset config: {dataset}")
    print(tiles)

    path = os.path.dirname(test_files[0]) 

    for tile in tiles:
        print(f'--------- TILE: {tile} --------- ')  # Example: tile='300546' 282553
        test_files = glob.glob(os.path.join(path, '*' + tile + '*.pt'))

        # Initialize empty arrays to avoid filling RAM, IoU computed for each tile.
        preds_arr = np.empty(0, dtype=int)
        targets_arr = np.empty(0, dtype=int)

        # Get predictions and targets for the tile
        targets_arr, preds_arr, n_data = test(args.output_dir,
                                            args.n_workers,
                                            model,
                                            test_files,
                                            args.n_points,
                                            targets_arr,
                                            preds_arr,
                                            plot_objects=False,
                                            n_classes=n_classes
                                            )
        if len(n_data)== 0:
            continue

        # Count the samples for each category
        unique_labels, counts = np.unique(targets_arr, return_counts=True)
        category_counts = dict(zip(unique_labels, counts))

        print('SHAPE: ', targets_arr.shape) 

        # Define the categories and compute IoUs
        categories = ['surr', 'tower', 'lines', 'wind_turbine']#, 'ground']
        iou = {cat: get_iou_np(targets_arr, preds_arr, idx) for idx, cat in enumerate(categories)}

        # Compute the mean IoU across relevant classes (ignoring NaNs)
        mean_iou = np.nanmean([iou[cat] for cat in categories])

        # Prepare category counts, defaulting to 0 for any missing categories
        counts_arr = [category_counts.get(i, 0) for i in range(len(categories))]

        # Write the IoU and category counts for this tile into the CSV
        with open(file_path, 'a') as fid:
            writer = csv.writer(fid)
            writer.writerow([tile] + [iou[cat] for cat in categories] + [mean_iou] + counts_arr)
            
        # confusion matrix
        cm = confusion_matrix(targets_arr, preds_arr, labels=[i for i in range(n_classes)])
        out_dir_cm = os.path.join(args.output_dir, 'confusion_matrices')
        if not os.path.exists(out_dir_cm):
            os.makedirs(out_dir_cm)
        with open(os.path.join(out_dir_cm, 'CM_' + dataset + '_' + tile +'.txt'), 'w') as filehandle:
            json.dump(cm.tolist(), filehandle)
            
    time_min = round((time.time() - start_time) / 60, 3)
    print("--- TOTAL TIME: %s min ---" % str(time_min))
