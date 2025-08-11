import argparse
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import datetime
from prettytable import PrettyTable
import random
import sys
sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import CAT3Dataset
# from src.LoRA.models.pointnet2_msg import *
from src.LoRA.models.pointnet2_ss import *
from src.config import *
from utils.utils import *
from utils.get_metrics import *
from utils.utils_plot import plot_pc_tensorboard
from src.LoRA.models.utils import *
from utils.utils_loss import *
from utils.utils_plot import *

random.seed(4)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def train(
        path_files,
        n_points,
        batch_size,
        epochs,
        learning_rate,
        number_of_workers,
        model_checkpoint,
        num_feat=5,
        num_classes=5,
        device='cuda'):
    
    start_time = time.time()

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'src/runs/lora/'
    NAME = 'seg_' + now.strftime("%m-%d-%H:%M") +'B29' + 'LoGlZ' 

    if num_classes>3:
        use_windturbine=True
    else:
        use_windturbine=False

    writer_train = SummaryWriter(location + NAME + '_train')
    writer_val = SummaryWriter(location + NAME + '_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    with open(os.path.join(path_files, 'train_files.txt'), 'r') as f: # data augm obj x2
        train_files = f.read().splitlines()
    with open(os.path.join(path_files, 'val_files.txt'), 'r') as f:
        val_files = f.read().splitlines()
    

    ####################### Filter data ##########################################################

    wind_files = [file for file in train_files if file.split('/')[-1].startswith('wind')]
    towers_files = [file for file in train_files if file.split('/')[-1].startswith('tower')]
    lines_files = [file for file in train_files if file.split('/')[-1].startswith('lines')]
    cranes_files = [file for file in train_files if file.split('/')[-1].startswith('crane')]

    # Output the filtered list
    print(f'Files with wind turbines: {len(wind_files)}')
    print(f'Files with towers: {len(towers_files)}')
    print(f'Files with lines: {len(lines_files)}')
    print(f'Files with cranes: {len(cranes_files)}')

    # data augmentation
    # train_files = train_files + towers_files + lines_files + wind_files 
    random.shuffle(train_files)

    print(f'len train files: {len(train_files)}')
    print(f'len val files: {len(val_files)}')
    ###############################################################################################

    # Initialize datasets
    train_dataset = CAT3Dataset(
                                task='segmentation',
                                number_of_points=n_points,
                                files=train_files,
                                fixed_num_points=True,
                                use_z=True,
                                use_windturbine=use_windturbine,
                                check_files=False,
                                store_filtered_paths=None,
                                max_z = 752
                                )
    val_dataset = CAT3Dataset(
                                task='segmentation',
                                number_of_points=n_points,
                                files=val_files,
                                fixed_num_points=True,
                                use_z=True,
                                use_windturbine=use_windturbine,
                                check_files=False,
                                store_filtered_paths=None,
                                max_z = 752
                                )

    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')
    logging.info(f'Num classes: {num_classes}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)
    # models
    pointnet = PointNet2(num_classes=num_classes, num_feat=num_feat)
    # pointnet = PointNet2msg(num_classes=num_classes, num_feat=num_feat)
    pointnet.to(device)
    pointnet = pointnet.apply(weights_init)
    pointnet = pointnet.apply(inplace_relu)

    # print models and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in pointnet.named_parameters():
        # if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    # print(table)

    optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate, eps=1e-08, weight_decay=1e-4)

    if num_classes == 3:    
            c_weights = [0.1, 0.2,  0.2]
    elif num_classes == 4:    
        c_weights = [0.1, 0.2,  0.2,  0.2]
    elif num_classes == 5:    
        c_weights = [0.1, 0.2,  0.2,  0.2, 0.1]

    c_weights = torch.tensor(c_weights).float().to(device)

    # samples x class for weight computation
    # c_weights = get_weights4class(WEIGHING_METHOD,
    #                               n_classes=N_CLASSES,
    #                               samples_per_cls=SAMPLES_X_CLASS_CAT3,
    #                               beta=0.999).to(device)
    print(f'Weights: {c_weights}')

    # loss
    loss = CELoss(weight=c_weights, reduction='mean')
    # loss = FocalLoss(gamma=2, weight=c_weights)
    # loss= DiceLoss()

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint,  map_location=device)
        print('Fine-tunning')
        pointnet.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']) # DEBUG FINE-TUNING
        optimizer.param_groups[0]['lr'] = learning_rate

    # scheduler
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-10)  

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs)):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        iou = {
            'tower_train': [],
            'tower_val': [],
            'veg_train': [],
            'veg_val': [],
            'building_train': [],
            'building_val': [],
            'ground_train': [],
            'ground_val': [],
            'surr_train': [],
            'surr_val': [],
            'wires_train': [],
            'wires_val': [],
            'turbine_train': [],
            'turbine_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
        }

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds = train_loop(data, optimizer, loss, pointnet, writer_train,
                                                 True, epoch, device=device, num_feat=num_feat, num_classes=num_classes, n_points=n_points)
            targets = targets.view(-1)
            preds = preds.view(-1)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics)

            # Segmentation labels:
            iou['surr_train'].append(get_iou_obj(targets, preds, 0))
            iou['tower_train'].append(get_iou_obj(targets, preds, 1))
            iou['wires_train'].append(get_iou_obj(targets, preds, 2))
            iou['turbine_train'].append(get_iou_obj(targets, preds, 3))
            # iou['ground_train'].append(get_iou_obj(targets, preds, 4))

            # tensorboard
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        scheduler_cosine.step()
        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():

            for data in val_dataloader:
                metrics, targets, preds = train_loop(data, optimizer, loss, pointnet, writer_val,
                                                     False, 1, device=device, num_feat=num_feat, num_classes=num_classes, n_points=n_points)
                targets = targets.view(-1)
                preds = preds.view(-1)
                metrics = get_accuracy(preds, targets, metrics)

                iou['surr_val'].append(get_iou_obj(targets, preds, 0))
                iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                iou['wires_val'].append(get_iou_obj(targets, preds, 2))
                iou['turbine_val'].append(get_iou_obj(targets, preds, 3))
                # iou['ground_val'].append(get_iou_obj(targets, preds, 4))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])

        # ------------------------------------------------------------------------------------------------------
        # Save checkpoint
        if np.mean(epoch_val_loss) < best_vloss:
            name = 'src/LoRA/checkpoints_lidarcat/'+ NAME 
            
            save_checkpoint_without_classifier_layer( name + '_NOclassifier', pointnet, optimizer, 
                                                     batch_size, learning_rate, n_points, epoch)
            save_checkpoint(name,  pointnet, optimizer, batch_size, learning_rate, n_points, epoch)

            print(f'model {name} saved at epoch {epoch}')
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1
      
        # Tensorboard
        writer_train.add_scalar('loss', np.nanmean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.nanmean(epoch_val_loss), epoch)
        writer_train.add_scalar('accuracy', np.nanmean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.nanmean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('_iou_tower', np.nanmean(iou['tower_train']), epoch)
        writer_val.add_scalar('_iou_tower', np.nanmean(iou['tower_val']), epoch)
        writer_train.add_scalar('_iou_surr', np.nanmean(iou['surr_train']), epoch)
        writer_val.add_scalar('_iou_surr', np.nanmean(iou['surr_val']), epoch)
        writer_train.add_scalar('_iou_wires', np.nanmean(iou['wires_train']), epoch)
        writer_val.add_scalar('_iou_wires', np.nanmean(iou['wires_val']), epoch)
        writer_train.add_scalar('_iou_wind_turbine', np.nanmean(iou['turbine_train']), epoch)
        writer_val.add_scalar('_iou_wind_turbine', np.nanmean(iou['turbine_val']), epoch)
        writer_train.add_scalar('_iou_ground', np.nanmean(iou['ground_train']), epoch)
        writer_val.add_scalar('_iou_ground', np.nanmean(iou['ground_val']), epoch)
        writer_train.flush()
        writer_val.flush()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, device='cuda', num_feat=5, num_classes=5, n_points=8000):
    """
    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    # fname_0 = filenames[0].split('/')[-1].split('.')[0]

    # Sample. Generate random indices without repetition
    random_indices = random.sample(range(0, n_points), 4096) 
    pc = pc[:, random_indices, :]
    targets = targets[:, random_indices]

    # point cloud rotation
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])

    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]
    pc = pc.transpose(2, 1)  # [16,5,4096]

    # color dropout -> Fill the specified positions with zeros
    if train == True:
        if random.random() < COLOR_DROPOUT: # 20%
            pc[:, -3:, :] = 0

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # PointNet model
    log_prob, feat_transform = pointnet(pc) # [batcn, 4096, 5]

    # Loss
    metrics['loss'] = loss(log_prob, targets)
    targets_pc = targets.detach().cpu()

    log_prob = log_prob.contiguous().view(-1, num_classes)
    targets = targets.contiguous().view(-1) # [batch, 4096]

    # get predictions
    # probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    probs = log_prob.detach().to('cpu')
    preds = torch.LongTensor(probs.data.max(1)[1])

    if train:
        metrics['loss'].backward()
        optimizer.step()

    return metrics, targets_pc, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_paths', type=str,default='train_test_files/B29_80x80_QPcot')  
    parser.add_argument('--num_points', type=int, default=8000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the dataloader')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model_checkpoint', 
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-10-16:49e2.pt',
                        # default='src/LoRA/checkpoints_lidarcat/seg_04-15-12:14B29normz.pt',
                        default='src/LoRA/checkpoints_lidarcat/seg_04-18-10:48B29LoGlZ.pt',
                        type=str, help='models checkpoint path')

    sys.path.append('/home/m.caros/work/3DSemanticSegmentation/')

    args = parser.parse_args()

    train(
        args.in_paths,
        args.num_points,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.num_workers,
        args.model_checkpoint,
        num_feat=9,
        num_classes=args.num_classes,
        device=args.device)
