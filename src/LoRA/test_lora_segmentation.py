import argparse
import os.path
import torch
import sys
import logging
import json
sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import CAT3SamplingDataset
from src.LoRA.models.lora_pointnet2_params import *
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
import torch.nn.functional as F
from src.config import *
from sklearn.metrics import confusion_matrix
import csv

N_SAMPLES = 14
global DATASET, DEVICE


def load_model(model_checkpoint, n_classes, num_features=8, lora_max_rank=64, lora_min_rank=8, lora_alpha=1):

    checkpoint = torch.load(model_checkpoint)

    # model
    model = LoraPointNet2(num_classes=n_classes,
                          num_feat=num_features,
                          lora_max_rank=lora_max_rank,
                          lora_min_rank=lora_min_rank,
                          alpha=lora_alpha).to(DEVICE)

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


def test(output_dir,
         number_of_workers,
         model,
         list_files,
         n_points,
         targets_arr,
         preds_arr, 
         plot_objects=False,
         n_classes=5,
         tile=''):
    
    # net to eval mode
    model = model.eval()

    # Initialize dataset
    test_dataset = CAT3SamplingDataset(task='segmentation',
                                       n_points=n_points,
                                       files=list_files,
                                       return_coord=False,
                                       tile_ids=True,
                                       use_z=True,
                                       keep_labels=plot_objects,
                                       use_windturbine=False,
                                       max_z=None, 
                                       check_files=True,
                                       )

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    preds_tile = torch.zeros((int(1e8), n_classes), dtype=torch.int64)
    labels_tile = np.zeros((int(1e8), n_classes), dtype=int)
            
    with torch.no_grad():
        for i, data in enumerate(progressbar(test_dataloader, )):

            pointcloud, targets, file_name, ids, n_unique_pts, _ = data  # [batch, n_samp, n_sampled_p, dims]
            
            ids_ini = ids.squeeze(0).cpu().to(torch.int64)
            file_name = file_name[0].split('/')[-1].split('.')[0]  # pc_B23_ETehpt421601_w212
            # tile = file_name.split('_')[-2]

            targets = targets.view(-1).cpu()
            targets_1hot = torch.nn.functional.one_hot(targets, num_classes=n_classes).numpy()
            # add labels to labels_tile
            labels_tile[ids_ini.numpy()] = targets_1hot

            pc = pointcloud.squeeze(0).to(DEVICE)  # [n_samp, n_points, dims] torch.Size([3, 24000, 8])
            dims = pc.shape[2]

            # if multiple samplings, duplicate points to use different samplings for different inputs into the model
            if pc.shape[1] >= n_points:
                if  2 * pc.shape[0] <= N_SAMPLES :
                    # duplicate points with different samplings
                    pc2, ids2, _ = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
                    pc = torch.cat((pc, pc2), dim=0)
                    ids = torch.cat((ids_ini, ids2), dim=0)
                else:
                    ids = ids_ini
            else:
                ids = ids_ini

            # -----------------------------------------------------------
            # transpose for model
            pc = pc.transpose(2, 1)

            # get logits from model
            preds = model(pc[:,:8,:])  # [batch, n_class, n_points]
            preds = preds.contiguous().view(-1, n_classes)

            # get predictions
            preds = F.softmax(preds, dim=1).data.max(1)[1]
            preds = preds.view(-1).to(dtype=torch.int64)  # [n_points]

            pc = pc.transpose(1, 2)
            pc = pc.view(-1, dims)  # [n_points, dims] [32000, 7]

            # one hot encoding
            preds_1hot = torch.nn.functional.one_hot(preds.cpu(), num_classes=n_classes).to(torch.int64)  # [points, classes]

            # --- Update the preds_tile matrix ---
            preds_tile.scatter_add_(0, ids.unsqueeze(1).expand(-1, n_classes), preds_1hot)  # [6x10⁹, n_classes]

            if plot_objects:
                print(f'Plotting predictions for {file_name}...')
                preds_all = preds.cpu().numpy()

                if 1 in set(preds_all): # or 2 in set(preds_all)):
                    obj_class=1
                    # if 1 in set(preds_all):
                    #     obj_class=1
                    # else:
                    #     obj_class=2

                    pc_np=pc.cpu().numpy()
                    plot_two_pointclouds_z(pc_np,
                                            pc_np,
                                            labels=pc_np[:,-1], # labels
                                            labels_2=preds_all,
                                            name=file_name,
                                            path_plot=os.path.join(output_dir, 'plots', tile),
                                            point_size=2,
                                            label_names=['Surrounding', 'Tower', 'Power lines'],
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
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points per point cloud')
    parser.add_argument('--max_rank', type=int, default=0, help='Lora maximum rank')
    parser.add_argument('--rank', type=int, default=32, help='Lora fixed rank')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/LoRA/checkpoints_lidarcat/loraPN2_07-23-12:12_32R32alph16.pt', # best
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-25-12:37_64R64alph16.pt', 
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-26-16:24_64R64alph32.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-27-20:19_64R64.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-28-09:52_4R4alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-28-09:49_8R8alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-26-16:22_64R64alph16.pt', 
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-28-09:49_8R8alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-31-10:13_32R32alph32.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-31-11:15_2R2alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_08-01-10:07_32R32alph64.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-27-20:19_64R64.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_08-01-20:34_64R64alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_08-01-19:59_1R1alph16.pt',
                        help='model checkpoint path')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--num_features', type=int, default=8, help='number of features to use')
    parser.add_argument('--device', type=str, default='cuda', help='device to be used, cuda or cpu')
    parser.add_argument('--dataset', type=str, default='RIB', help='dataset for inference. Options: RIB, B29 or COSTA')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRa alpha')
    parser.add_argument('--plot_preds', type=bool, default=False, help='plot predictions')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    start_time = time()
    
    DEVICE = args.device
    logging.info(f'DEVICE: {DEVICE}')

    DATASET = args.dataset
    logging.info(f'DATASET: {DATASET}')

    logging.info(f'Plot predictions: {args.plot_preds}')

    n_classes = args.num_classes
    n_feat = args.num_features
    logging.info(f'NUM CLASSES: {n_classes}')

    if DATASET == 'RIB':
        args.in_path = 'train_test_files/RIB_smallLoRA_80x80'
        args.output_dir = 'src/LoRA/metrics/results_RIB'

        with open(os.path.join(args.in_path, 'test_files_filtered.txt'), 'r') as f:
            test_files = f.read().splitlines()
        # tiles = list(set([path_f.split('_')[-2].split('.')[0] for path_f in test_files]))
        path = os.path.dirname(test_files[0]) 
        # test tiles
        tiles=["pt438656", "pt438652","pt438658","pt440652"] 

    elif DATASET == 'B29':
        args.in_path = 'train_test_files/B29_80x80'
        args.output_dir = 'src/LoRA/metrics/results_B29_trainedRIB'

        with open(os.path.join(args.in_path, 'test_files_filtered.txt'), 'r') as f:
            test_files = f.read().splitlines()
        path = os.path.dirname(test_files[0]) 
        # test tiles
        tiles=['279553', '279554', '282553', '282555', '282557', '283553','283555', '300546']

    elif DATASET == 'COSTA':
        path = '/mnt/QPcotLIDev01/LiDAR/DL_preproc/COSTA_80x80_s40_p8k_id_f32'
        args.output_dir = 'src/LoRA/metrics/results_COSTA'
        # test tiles
        tiles=["ETOHc000244", "ETOHc000174", 'ETOHc000031', 'ETOHc000033', 'ETOHc000044','ETOHc000024']

    lora_max_rank = args.max_rank
    if lora_max_rank == 0:
        lora_min_rank = args.rank
        lora_max_rank = lora_min_rank

    # load model
    model = load_model(args.model_checkpoint, n_classes, n_feat, lora_max_rank, lora_min_rank, lora_alpha=args.lora_alpha)
    MODEL_NAME = args.model_checkpoint.split('/')[-1].split('.')[0]

    dataset = MODEL_NAME + str(args.n_points) + DATASET
    print(f"Dataset config: {dataset}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Define file path for storing results
    file_path = os.path.join(args.output_dir, 'IoU-' + dataset + '.csv')

    # store in csv file
    with open(file_path, 'a') as fid:
        writer = csv.writer(fid)
        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            # Write the header if the file is empty
            writer.writerow(["Tile", "Surrounding IoU", "Tower IoU", "Lines IoU",  "Mean IoU", "points surr", "points tower", "points lines"])

    tiles.sort()
    print(tiles)

    # Initialize empty arrays here if IoU needs to be computed for the whole dataset. Result is more rigorous
    # preds_arr = np.empty(0, dtype=int)
    # targets_arr = np.empty(0, dtype=int)

    for tile in tiles:
        print(f'--------- TILE: {tile} --------- ')  # Example: tile='300546' 282553
        test_files = glob.glob(os.path.join(path, '*' + tile + '*.pt')) 
        
        if len(test_files)== 0:
            continue

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
                                            plot_objects=args.plot_preds,
                                            n_classes=n_classes,
                                            tile=tile
                                            )
        if len(n_data)== 0:
            continue
        
        # Count the samples for each category
        unique_labels, counts = np.unique(targets_arr, return_counts=True)
        category_counts = dict(zip(unique_labels, counts))

        # Define the categories and compute IoUs
        categories = ['surr', 'tower', 'lines']
        iou = {cat: get_iou_np(targets_arr, preds_arr, idx) for idx, cat in enumerate(categories)}

        # Compute the mean IoU across relevant classes (ignoring NaNs)
        mean_iou = np.nanmean([iou[cat] for cat in categories])

        # Prepare category counts, defaulting to 0 for any missing categories
        counts_arr = [category_counts.get(i, 0) for i in range(len(categories))]

        # Write the IoU and category counts for this tile into the CSV
        with open(file_path, 'a') as fid:
            writer = csv.writer(fid)
            writer.writerow([tile] + [iou[cat] for cat in categories] + [mean_iou] + counts_arr)
            print(f'{([tile] + [iou[cat] for cat in categories])}')

        # ------ confusion matrix -------
        cm = confusion_matrix(targets_arr, preds_arr,  labels=[i for i in range(n_classes)])

        out_dir_cm = os.path.join(args.output_dir, 'confusion_matrices')
        if not os.path.exists(out_dir_cm):
            os.makedirs(out_dir_cm)
        with open(os.path.join(out_dir_cm, 'CM_' + dataset + '_' + tile +'.txt'), 'w') as filehandle:
            json.dump(cm.tolist(), filehandle)

    time_min = round((time() - start_time) / 60, 3)
    print("--- TOTAL TIME: %s min ---" % str(time_min))

