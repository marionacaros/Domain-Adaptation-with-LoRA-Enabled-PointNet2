import argparse
import torch.optim as optim
from time import time
from torch.utils.tensorboard import SummaryWriter
import logging
import datetime
from prettytable import PrettyTable
import random
import sys

sys.path.append('/home/m.caros/work/3DSemanticSegmentation')
from src.datasets import CAT3Dataset
from src.LoRA.models.lora_pointnet2_params import *
from src.config import *
from utils.utils import *
from utils.get_metrics import *
from utils.utils_plot import plot_pc_tensorboard
from src.LoRA.models.utils import *

random.seed(5)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def train(
        path_list_files,
        n_points,
        batch_size,
        epochs,
        learning_rate,
        number_of_workers,
        model_checkpoint,
        n_feat=5,
        n_classes=5, 
        lora_fix_rank=False,
        lora_min_rank=16,
        lora_max_rank=64,
        lora_alpha=1):
    
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
    NAME = 'loraPN2_'+ now.strftime("%m-%d-%H:%M") + f'_{lora_min_rank}R'+ f'{lora_max_rank}'+'alph'+str(lora_alpha)

    writer_train = SummaryWriter(location + NAME + '_train')
    writer_val = SummaryWriter(location + NAME + '_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    with open(os.path.join(path_list_files, 'train_files.txt'), 'r') as f:
        train_files = f.read().splitlines()
    # train_files = list(set(train_files))

    val_files=[]

    train_files.sort()
    random.shuffle(train_files)

    val_files = train_files[int(0.80*len(train_files)):]
    train_files = train_files[:int(0.80*len(train_files))]
    
    # ####################### Filter data ##########################################################

    # train files
    print('---train files---')
    towers_files = [file for file in train_files if file.split('/')[-1].startswith('tower')]
    lines_files = [file for file in train_files if file.split('/')[-1].startswith('line')]
    print(f'Files with towers: {len(towers_files)}')
    print(f'Files with lines: {len(lines_files)}')

    train_files = train_files + towers_files + lines_files + towers_files + lines_files 
    random.shuffle(train_files)
   
    print('---val files---')
    towers_files = [file for file in val_files if file.split('/')[-1].startswith('tower')]
    lines_files = [file for file in val_files if file.split('/')[-1].startswith('line')]
    print(f'Files with towers: {len(towers_files)}')
    print(f'Files with lines: {len(lines_files)}')

    print(f'len train files: {len(train_files)}')
    print(f'len val files: {len(val_files)}')

    ###############################################################################################

    # Initialize datasets
    train_dataset = CAT3Dataset(
                                number_of_points=n_points,
                                files=train_files,
                                fixed_num_points=True,
                                use_z=True,
                                use_windturbine=False,
                                check_files=False,
                                is_prod=False,
                                max_z=None)
    val_dataset = CAT3Dataset(
                                number_of_points=n_points,
                                files=val_files,
                                fixed_num_points=True,
                                use_z=True,
                                use_windturbine=False,
                                check_files=False,
                                is_prod=False,
                                max_z=None)
    
    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True,
                                                 pin_memory=False)
    # model
    pointnet = LoraPointNet2(n_classes, 
                             n_feat, 
                             lora_fix_rank=lora_fix_rank, 
                             lora_max_rank=lora_max_rank,
                             lora_min_rank=lora_min_rank,
                             alpha=lora_alpha)
    
    # Check if non-LoRA parameters are indeed excluded from the optimizer
    for p in pointnet.parameters():
        if not p.requires_grad:
            assert p.grad is None, "Non-LoRA parameters are still being updated."

    # print models and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in pointnet.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Total Params: {total_params}")
    # print(table)

    # Export table to a text file
    with open("src/LoRA/logs/" + NAME + ".txt", "w") as f:
        f.write(f"Total Params: {total_params}\n")
        f.write(f"Min rank: {pointnet.min_rank}\n")
        f.write(f"Max rank: {pointnet.max_rank}\n")
        f.write(str(table))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pointnet.parameters()), lr=learning_rate, eps=1e-08, weight_decay=1e-4)

    if n_classes == 3:    
        c_weights = [0.1, 0.2,  0.2]
    elif n_classes == 4:    
        c_weights = [0.1, 0.2,  0.2,  0.2]
    
    c_weights = torch.tensor(c_weights).float().to(device)

    print(f'Weights: {c_weights}')

    # loss
    LS = 0.0
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1, label_smoothing=LS)

    # Load stopped training of LoRA model
    # if model_checkpoint:
    #     print('Loading checkpoint of stopped training')
    #     checkpoint = torch.load(model_checkpoint,  map_location=device)
    #     print('Fine-tunning')
    #     pointnet.load_state_dict(checkpoint['model'], strict=True)
    #     pointnet.to(device)

    #     optimizer.load_state_dict(checkpoint['optimizer']) # DEBUG FINE-TUNING
    #     optimizer.param_groups[0]['lr'] = learning_rate

    # Load baseline B29 trained model
    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint,  map_location=device)
        
        # Need to copy weights as layer names are different
        chkp_model_kvpairs = list(checkpoint['model'].items())
        my_model_kvpair = pointnet.state_dict()
        count=0
        myKeys = list(my_model_kvpair.keys())[44:-2] # remove lora A & B matrices weights and classification layer weights and bias
        # myKeys = list(my_model_kvpair.keys())[44:]
        for key in myKeys:
            layer_name, weights = chkp_model_kvpairs[count]      
            my_model_kvpair[key]=weights
            count+=1
        pointnet.load_state_dict(my_model_kvpair)

        pointnet.to(device)

    # count_parameters(pointnet)

    print(f'Memory allocated to CUDA: {torch.cuda.memory_allocated()/1e6} GB')

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
            'tower_train': [],
            'tower_val': [],
            'surr_train': [],
            'surr_val': [],
            'wires_train': [],
            'wires_val': [],
            'building_train': [],
            'building_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
        }

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:

            metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_train,
                                                True, epoch, device=device, num_feat=n_feat, n_classes=n_classes, 
                                                n_points=n_points)
            targets = targets.view(-1)
            preds = preds.view(-1)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics)

            # Segmentation labels:
            # 0 -> other classes we're not interested and rest of ground
            # 1 -> tower
            # 2 -> lines
            iou['surr_train'].append(get_iou_obj(targets, preds, 0))
            iou['tower_train'].append(get_iou_obj(targets, preds, 1))
            iou['wires_train'].append(get_iou_obj(targets, preds, 2))

            # tensorboard
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        scheduler_cosine.step()
        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():

            for data in val_dataloader:
                metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_val,
                                                     False, 1, device=device, num_feat=n_feat, n_classes = n_classes, 
                                                     n_points=n_points)
                targets = targets.view(-1)
                preds = preds.view(-1)
                metrics = get_accuracy(preds, targets, metrics)
                
                iou['surr_val'].append(get_iou_obj(targets, preds, 0))
                iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                iou['wires_val'].append(get_iou_obj(targets, preds, 2))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])

        # ------------------------------------------------------------------------------------------------------
        # Save checkpoint
        if np.mean(epoch_val_loss) < best_vloss:
            name = 'src/LoRA/checkpoints_lidarcat/'+ NAME 
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
        writer_train.flush()
        writer_val.flush()

    print("--- TOTAL TIME: %s h ---" % (round((time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, device='cuda',num_feat=5, n_classes=5, n_points=8000):
    """
    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    # pc = pc.data.numpy()
    # fname_0 = filenames[0].split('/')[-1].split('.')[0]

    # Sample. Generate random indices without repetition
    random_indices = random.sample(range(0, n_points), 4096)  
    pc = pc[:, random_indices, :]
    targets = targets[:, random_indices]

    # point cloud rotation
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    # pc = torch.FloatTensor(pc)

    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]
    pc = pc.transpose(2, 1)  # [16,5,4096]

    # color dropout -> Fill the specified positions with zeros
    if train == True:
        if random.random() < COLOR_DROPOUT: # 20%
            pc[:, -3:, :] = 0

    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    logits = pointnet(pc)

    logits = logits.contiguous().view(-1, n_classes)
    targets = targets.contiguous().view(-1)

    # CrossEntropy loss
    metrics['loss'] = ce_loss(logits, targets)
    targets_pc = targets.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])

    if train:
        metrics['loss'].backward()
        optimizer.step()

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()

    return metrics, targets_pc, preds


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"🔍 Total de parámetros:     {total:,}")
    print(f"✅ Parámetros entrenables:  {trainable:,}")
    print(f"❄️  Parámetros congelados:   {frozen:,}")
    print(f"📉 Porcentaje entrenable:   {trainable / total * 100:.2f}%")
    
    # También podrías imprimir qué capas están entrenables, por si quieres más detalle
    print("\n🔧 Capas con parámetros entrenables:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}: {param.numel():,} parámetros")


# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_paths', type=str, 
                        default='train_test_files/RIB_smallLoRA_80x80')
    parser.add_argument('--num_points', type=int, default=8000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--min_rank', type=int, default=64, help='Fixed matrix rank to be used when training LoRA. If set to 0 lora rank variable between 4 and max_rank')
    parser.add_argument('--max_rank', type=int, default=64, help='Maximum matrix rank to be used when training LoRA.')
    parser.add_argument('--num_classes', type=int, default=3, help='number of workers for the dataloader')
    parser.add_argument('--num_features', type=int, default=8, help='number of features')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/LoRA/checkpoints_lidarcat/seg_02-24-15:52B29_NOclassifier.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-23-12:12_32R32alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-25-12:37_64R64alph16.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-29-19:36_32R32alph32.pt',
                        # default='src/LoRA/checkpoints_lidarcat/loraPN2_07-29-19:35_32R32alph64.pt',
                        #  default='src/LoRA/checkpoints_lidarcat/loraPN2_07-29-10:05_2R2alph16.pt',
                        help='models checkpoint path')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRa alpha')
    

    sys.path.append('/home/m.caros/work/3DSemanticSegmentation/')

    args = parser.parse_args()

    train(
        args.in_paths,
        args.num_points,
        args.batch_size,
        args.epochs,
        args.lr,
        args.num_workers,
        args.model_checkpoint,
        n_feat=args.num_features,
        n_classes=args.num_classes, 
        lora_max_rank=args.max_rank,
        lora_min_rank=args.min_rank,
        lora_alpha=args.lora_alpha)
