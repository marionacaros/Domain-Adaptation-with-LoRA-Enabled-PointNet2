import argparse
import os.path
import torch
import sys
import logging
import json
sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import DalesSamplingDataset
from src.LoRA.models.lora_pointnet2_old import *
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
import torch.nn.functional as F
from src.config import *
from sklearn.metrics import confusion_matrix
import csv


N_SAMPLES = 16
global DATASET, DEVICE


def load_model(model_checkpoint, n_classes, n_feat=5, lora_max_rank=64, lora_fix_rank=0):
    checkpoint = torch.load(model_checkpoint)

    # model
    model = LoraPointNet2(num_classes=n_classes, 
                      num_feat=n_feat, 
                      radius=[0.1, 0.2, 0.4, 0.8],
                      lora_fix_rank=lora_fix_rank,
                        lora_max_rank=lora_max_rank).to(DEVICE)
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
         n_classes=5,
         n_feat=5):
    
    # net to eval mode
    model = model.eval()

    # Initialize dataset
    test_dataset = DalesSamplingDataset(
                                task='segmentation',
                                number_of_points=n_points,
                                files=list_files,
                                fixed_num_points=True,
                                keep_labels=True,
                                use_all_labels=True
                                )
    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    preds_tile = torch.zeros((100000000, n_classes), dtype=torch.int64)
    labels_tile = np.zeros((100000000, n_classes), dtype=int)
    last_max_id = 0

    with torch.no_grad():
        for i, data in enumerate(progressbar(test_dataloader, )):

            pointcloud, targets, file_name, ids = data  # [batch, n_samp, n_sampled_p, dims]

            ids = ids + last_max_id
            last_max_id = ids.numpy().max()

            ids_ini = ids.squeeze(0).cpu().to(torch.int64)
            file_name = file_name[0].split('/')[-1].split('.')[0]  # pc_B23_ETehpt421601_w212
            # tile = file_name.split('_')[-2]

            targets = targets.view(-1).cpu()
            


            pc = pointcloud.squeeze(0).to(DEVICE)  # [n_samp, n_points, dims]
            dims = pc.shape[2]

            # if n_points < pc.shape[0] * pc.shape[1] < N_SAMPLES * n_points:
            # ------------- duplicate points with different samplings ----------------
            pc2, ids2, _ = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
            pc3, ids3, _ = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
            pc = torch.cat((pc, pc2, pc3), dim=0)
            ids = torch.cat((ids_ini, ids2, ids3), dim=0)
            # ------------------------------------------------------------------------

            # transpose for model
            pc = pc.transpose(2, 1)

            # get logits from model
            preds = model(pc[:,:n_feat,:])  # [batch, n_points, n_class]
            preds = preds.contiguous().view(-1, n_classes)

            # get predictions
            preds = F.softmax(preds, dim=1).data.max(1)[1]
            preds = preds.view(-1).to(dtype=torch.int64)  # [n_points]

            pc = pc.transpose(1, 2)
            pc = pc.view(-1, dims)  # [n_points, dims] [32000, 7]

            # Create a mask for valid labels (excluding -1)
            mask = targets != -1

            targets_1hot = torch.nn.functional.one_hot(targets[mask], num_classes=n_classes).numpy()
            # add labels to labels_tile
            labels_tile[ids_ini[mask].numpy()] = targets_1hot

            valid_ids=ids_ini[mask]
            # Create a mask where each element in ids is checked against valid_ids
            ids_mask = torch.isin(ids, valid_ids)  # Returns a boolean mask of the same shape as ids        
            # one hot encoding
            preds_1hot = torch.nn.functional.one_hot(preds[ids_mask].cpu(), num_classes=n_classes).to(torch.int64)  # [points, classes]

            # --- Update the preds_tile matrix ---
            preds_ids = ids[ids_mask].unsqueeze(1)
            preds_ids = preds_ids.expand(-1, n_classes)
            preds_tile.scatter_add_(0, preds_ids, preds_1hot)  # [6x10⁹, N_CLASSES]

            # debug and plot
            if plot_objects:
                preds_all = preds.cpu().numpy()
                pc_np=pc.cpu().numpy()
                plot_pointcloud_with_labels_DALES(pc_np,
                                                    labels=pc_np[:,-1],
                                                    preds=preds_all,
                                                    name=file_name + '_labels_preds',
                                                    path_plot=os.path.join(output_dir, 'plots/predictions_dales'),
                                                    point_size=2,
                                                    n_classes=n_classes)

            # ---------------------------------- end voting loop --------------------------------------------------------
            del preds, pc
            torch.cuda.empty_cache()

    print('Last max id:', last_max_id)
    final_class_tile = torch.argmax(preds_tile, dim=1).numpy()
    labels_tile = np.argmax(labels_tile, axis=1)

    # Find the indices of rows with non-zero values
    mask = (preds_tile != 0).any(dim=1)
    non_zero_ix = torch.where(mask)[0]

    if len(non_zero_ix) > 0:
    # Get the last index with non-zero values if any are found
        # Get only points that went through the model
        final_class_tile=final_class_tile[mask]
        labels_tile=labels_tile[mask]

    return labels_tile, final_class_tile, non_zero_ix

    # Concatenate with the existing arrays
    preds_arr = np.concatenate((preds_arr, final_class_tile))
    targets_arr = np.concatenate((targets_arr, labels_tile))
    return targets_arr, preds_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='src/LoRA/metrics/results_DALES',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points per point cloud')
    parser.add_argument('--model_checkpoint', type=str,
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-12-15:27_lora_64R_64.pt', # trained 2 blocks
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-12-15:23_lora_32R_32.pt', # trained 2 blocks
                        default='src/LoRA/checkpoints_lidarcat/seg_02-13-10:31_lora_32R_32T4.pt', # trained 4 blocks
                        help='models checkpoint path')
    parser.add_argument('--in_path', type=str, default='/dades/LIDAR/towers_detection/datasets/DALES/dales_25x25/test')
    parser.add_argument('--device', type=str, default='cuda', help='device to be used, cuda or cpu')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--num_feat', type=int, default=5, help='number of classes')
    parser.add_argument('--max_rank', type=int, default=32, help='Lora maximum rank')
    parser.add_argument('--fix_rank', type=int, default=32, help='Lora fixed rank')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    start_time = time()
    
    DEVICE = args.device
    logging.info(f'DEVICE: {DEVICE}')
    DATASET = 'DALES'
    logging.info(f'DATASET: {DATASET}')
    logging.info(f'Number of classes: {args.num_classes}')

    lora_max_rank = args.max_rank
    lora_fix_rank = args.fix_rank

    # load model
    model = load_model(args.model_checkpoint, args.num_classes, args.num_feat,lora_max_rank, lora_fix_rank)
    MODEL_NAME = args.model_checkpoint.split('/')[-1].split('.')[0]

    dataset = DATASET + '_' + str(args.n_points) + 'p_' + MODEL_NAME
    print(f"Dataset config: {dataset}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Data 
    test_files = glob.glob(os.path.join(args.in_path, '*.pt'))
    logging.info(f'Number of files: {len(test_files)}')

    # Define output file path
    file_path = os.path.join(args.output_dir, 'IoU-results-%s.csv' % dataset)

    # store in csv file
    with open(file_path, 'a') as fid:
        writer = csv.writer(fid)
        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            # Write the header if the file is empty
            writer.writerow(["Tile", 'ground iou', 'poles iou', 'powerline iou', 'veg iou', 'buildings iou','cars iou', 'mean iou', 'pts. ground',  'pts. poles','pts. powerline', 'pts. veg','pts. buildings', 'pts. cars'])

    # list of tiles
    tiles = set([path_f.split('_')[-2].split('.')[0] for path_f in test_files])
    tiles = list(tiles)
    tiles.sort()
    # tiles=['54325', '54335', '54390', '54395', '54400',  '54435', '54440', '54445', '54470', '54490'] #'54430',
    # tiles =['54445', '54470', '54490','54440']

    logging.info(f"Number of tiles: {len(tiles)}")
    print(tiles)

    # Initialize empty arrays here if IoU needs to be computed for the whole dataset. Result is more rigorous
    # preds_arr = np.empty(0, dtype=int)
    # targets_arr = np.empty(0, dtype=int)

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
                                            n_classes=args.num_classes,
                                            n_feat=args.num_feat
                                            )
        if len(n_data)== 0:
            continue

        # Count the samples for each category
        unique_labels, counts = np.unique(targets_arr, return_counts=True)
        category_counts = dict(zip(unique_labels, counts))

        print('SHAPE: ', targets_arr.shape) 

        # Define the categories and compute IoUs
        categories = ['ground', 'poles', 'powerlines', 'veg', 'buildings', 'cars']
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
        cm = confusion_matrix(targets_arr, preds_arr, labels=[i for i in range(args.num_classes)])
        out_dir_cm = os.path.join(args.output_dir, 'confusion_matrices')
        if not os.path.exists(out_dir_cm):
            os.makedirs(out_dir_cm)
        with open(os.path.join(out_dir_cm, 'CM_' + dataset + '_' + tile +'.txt'), 'w') as filehandle:
            json.dump(cm.tolist(), filehandle)
            
    time_min = round((time() - start_time) / 60, 3)
    print("--- TOTAL TIME: %s min ---" % str(time_min))
