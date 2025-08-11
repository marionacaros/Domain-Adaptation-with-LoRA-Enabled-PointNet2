import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import datetime
from prettytable import PrettyTable
import random
import sys

sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import DalesDataset
from src.LoRA.models.lora_pointnet2_old import *
from src.config import *
from utils.utils import *
from utils.get_metrics import *
from utils.utils_plot import plot_pc_tensorboard
from src.LoRA.models.utils import *

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
        lora_fix_rank=None,
        lora_max_rank=64):
    
    start_time = time()

    if torch.cuda.is_available():
        logging.info(f"cuda available")
        device = 'cuda'
    else:
        logging.info(f"cuda not available")
        device = 'cpu'

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'src/runs/lora/'
    NAME = 'seg_' + now.strftime("%m-%d-%H:%M") + '_lora'+ f'_{lora_fix_rank}R'+ f'_{lora_max_rank}'+'T4'

    writer_train = SummaryWriter(location + NAME + '_train')
    writer_val = SummaryWriter(location + NAME + '_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # train_files = glob.glob(os.path.join(path_files, '*.pt'))
    
    train_files=[]
    tiles=['54320', '54395', '54340', '54455'] # last 2 tiles have powerlines
    for tile in tiles:
        train_files.extend(glob.glob(os.path.join(path_files, '*' + tile + '*.pt')))


    val_files = train_files[int(0.80*len(train_files)):]
    train_files = train_files[:int(0.80*len(train_files))]

    print(f'len train files: {len(train_files)}')
    print(f'len val files: {len(val_files)}')

    # Initialize datasets
    train_dataset = DalesDataset(
                                task='segmentation',
                                number_of_points=n_points,
                                files=train_files,
                                fixed_num_points=True,
                                use_all_labels=True
                                )
    val_dataset = DalesDataset(
                                task='segmentation',
                                number_of_points=n_points,
                                files=val_files,
                                fixed_num_points=True,
                                use_all_labels=True
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
    # model
    pointnet = LoraPointNet2(num_classes, 
                             num_feat, 
                             lora_fix_rank=lora_fix_rank, 
                             lora_max_rank=lora_max_rank,
                             lora_alpha=1,
                             radius=[0.1, 0.2, 0.4, 0.8])

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

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pointnet.parameters()), lr=learning_rate, eps=1e-08, weight_decay=1e-4)

    # samples x class for weight computation
    c_weights = get_weights4class(WEIGHING_METHOD,
                                  n_classes=N_CLASSES,
                                  samples_per_cls=SAMPLES_X_CLASS_DALES_ALL, # change for FT
                                  beta=0.9999).to(device)
    print(f'Weights: {c_weights}')

    # loss
    LS = 0.0
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', label_smoothing=LS, ignore_index=-1)

    # Load checkpoint
    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint, map_location='cuda')
        
        # Need to copy weights as layer names are different
        chkp_model_kvpairs = list(checkpoint['model'].items())
        my_model_kvpair = pointnet.state_dict()
        count=0
        myKeys = list(my_model_kvpair.keys())[44:-2] # remove lora A & B matrices weights and classification layer weights and bias
        for key in myKeys:
            layer_name, weights = chkp_model_kvpairs[count]      
            my_model_kvpair[key]=weights
            count+=1
        pointnet.load_state_dict(my_model_kvpair)
    else:
        pointnet = pointnet.apply(weights_init)
        pointnet = pointnet.apply(inplace_relu)

    pointnet.to(device)

    # schedulers
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-10)  

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs)):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        
        iou = {
            'ground_train': [],
            'ground_val': [],
            'powerlines_train': [],
            'powerlines_val': [],
            'poles_train': [],
            'poles_val': [],
            'veg_train': [],
            'veg_val': [],
            'fences_val': [],
            'buildings_val': [],
            'fences_train': [],
            'buildings_train': [],
            'cars_val': [],
            'cars_train': [],
            'surr_val': [],
            'surr_train': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
        }

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_train,
                                                True, epoch, device=device, num_feat=num_feat, num_classes=num_classes, n_points=n_points)
            targets = targets.view(-1)
            preds = preds.view(-1)
            # Compute metrics
            metrics = get_accuracy(preds, targets, metrics)

            # Add IoU for all categories
            # iou['surr_train'].append(get_iou_obj(targets, preds, 0))
            iou['ground_train'].append(get_iou_obj(targets, preds, 0))
            iou['poles_train'].append(get_iou_obj(targets, preds, 1))
            iou['powerlines_train'].append(get_iou_obj(targets, preds, 2))
            iou['veg_train'].append(get_iou_obj(targets, preds, 3))
            iou['buildings_train'].append(get_iou_obj(targets, preds, 4))
            iou['cars_train'].append(get_iou_obj(targets, preds, 5))
            # Tensorboard
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        scheduler_cosine.step()

        # --------------------------------------------- val loop ---------------------------------------------
        with torch.no_grad():
            for data in val_dataloader:
                metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_val,
                                                    False, 1, device=device, num_feat=num_feat, num_classes=num_classes, n_points=n_points)
                targets = targets.view(-1)
                preds = preds.view(-1)
                metrics = get_accuracy(preds, targets, metrics)

                # iou['surr_val'].append(get_iou_obj(targets, preds, 0))
                iou['ground_val'].append(get_iou_obj(targets, preds, 0))
                iou['poles_val'].append(get_iou_obj(targets, preds, 1))
                iou['powerlines_val'].append(get_iou_obj(targets, preds, 2))
                iou['veg_val'].append(get_iou_obj(targets, preds, 3))
                iou['buildings_val'].append(get_iou_obj(targets, preds, 4))
                iou['cars_val'].append(get_iou_obj(targets, preds, 5))

                # Tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())
                epoch_val_acc.append(metrics['accuracy'])

        # ------------------------------------------------------------------------------------------------------
        # Save checkpoint
        if np.mean(epoch_val_loss) < best_vloss:
            name = 'src/LoRA/checkpoints_lidarcat/' + NAME
            save_checkpoint(name, pointnet, optimizer, batch_size, learning_rate, n_points, epoch)

            print(f'model {name} saved at epoch {epoch}')
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)
        else:
            epochs_since_improvement += 1

        # Tensorboard Logging
        writer_train.add_scalar('loss', np.nanmean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.nanmean(epoch_val_loss), epoch)
        writer_train.add_scalar('accuracy', np.nanmean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.nanmean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Log IoU for all categories
        for category in ['ground', 'powerlines', 'poles', 'veg', 'buildings', 'cars']: #'fences_buildings', 'cars_trucks'
            writer_train.add_scalar(f'_iou_{category}', np.nanmean(iou[f'{category}_train']), epoch)
            writer_val.add_scalar(f'_iou_{category}', np.nanmean(iou[f'{category}_val']), epoch)

        writer_train.flush()
        writer_val.flush()

    print("--- TOTAL TIME: %s h ---" % (round((time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, device='cuda',num_feat=5, num_classes=5, n_points=8000):
    """
    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    pc = pc.data.numpy()
    # fname_0 = filenames[0].split('/')[-1].split('.')[0]

    # Sample. Generate random indices without repetition
    random_indices = random.sample(range(0, n_points), 4096) 
    pc = pc[:, random_indices, :]
    targets = targets[:, random_indices]

    # point cloud rotation
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.Tensor(pc)

    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]
    pc = pc.transpose(2, 1)  # [16,5,4096]

    # color dropout -> Fill the specified positions with zeros
    if train == True:
        if random.random() < COLOR_DROPOUT: # 20%
            pc[:, num_feat:, :] = 0

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # PointNet model
    logits = pointnet(pc) # [64, 4096, 5]
    logits = logits.contiguous().view(-1, num_classes)
    targets = targets.contiguous().view(-1)

    # CrossEntropy loss
    metrics['loss'] = ce_loss(logits, targets)
    targets_pc = targets.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])

    # plot predictions in Tensorboard
    # if epoch % 10 == 0 and epoch > 0 and random.random() < 0.05 and 1 in set(targets_pc[:4096].numpy()):
    #     # preds_plot, targets_plot, mask = rm_padding(preds[0, :].cpu(), targets_pc[0, :])
    #     preds_plot, targets_plot = preds[:4096], targets_pc[:4096]
    #     # Tensorboard
    #     plot_pc_tensorboard(pc[0, :, :].T.cpu(), targets_plot, None, fname_0 +'_targets', step=epoch,
    #                         classes=N_CLASSES, model_name=model_name)
    #     plot_pc_tensorboard(pc[0, :, :].T.cpu(), preds_plot, None, fname_0 + '_predictions', step=epoch,
    #                         classes=N_CLASSES, model_name=model_name)

    if train:
        metrics['loss'].backward()
        optimizer.step()

    return metrics, targets_pc, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_paths', type=str,default='/dades/LIDAR/towers_detection/datasets/DALES/dales_25x25/train')  
    parser.add_argument('--num_points', type=int, default=8000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the dataloader')
    parser.add_argument('--fix_rank', type=int, default=32, help='Fixed matrix rank to be used when training LoRA. If set to 0 lora rank variable between 4 and max_rank')
    parser.add_argument('--max_rank', type=int, default=32, help='Maximum matrix rank to be used when training LoRA.')
    parser.add_argument('--num_classes', type=int, default=6, help='number of workers for the dataloader')
    parser.add_argument('--num_features', type=int, default=5, help='number of features')
    parser.add_argument('--model_checkpoint', 
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-06-18:22base_NOclassifier.pt',
                        # default='src/LoRA/checkpoints_lidarcat/seg_02-04-11:52e2_NOclassifier.pt',
                        default='src/LoRA/checkpoints_lidarcat/seg_02-10-11:21_base_NOclassifier.pt',
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
        num_feat=args.num_features,
        num_classes=args.num_classes, 
        lora_fix_rank=args.fix_rank,
        lora_max_rank=args.max_rank)
