import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np

from pytorch_lightning import Callback
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics import Accuracy
import torch.nn.functional as F
import math

from src.models_pointnet import *
from src.utils import *

# set precision to what lightning suggests for this gpu
torch.set_float32_matmul_precision('high')
# make results reproducible
L.seed_everything(42)


class BaseLoraPointNet(nn.Module):
    def __init__(self, lora_rank, num_classes, channels_in, point_dimension, device, dropout, train_feat_transf=False, lora_mm=False):
        super(BaseLoraPointNet, self).__init__()

        self.channels_in = channels_in
        self.num_classes = num_classes
        self.point_dimension = point_dimension
        self.device = device
        self.train_feat_transf = train_feat_transf
        self.lora_mm = lora_mm
        
        # Transformation-Nets
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension, device=device)
        
#         if self.train_feat_transf:
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)

        # Convolutional layers
        self.conv_1 = nn.Conv1d(self.channels_in, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, 1024, 1, bias=False)

        # Batch normalization layers
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

        # Classification layers
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

        self.bn_21 = nn.BatchNorm1d(512)
        self.bn_22 = nn.BatchNorm1d(256)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = 1

        # Define LoRA layers
        self.define_lora_layers()

        # Initialize LoRA layers
        self.init_lora_layers()
        
        if self.train_feat_transf:
            self.freeze_non_lora_non_ft()
        else:
            # Freeze non-LoRA weights
            self.freeze_non_lora_weights()

    def define_lora_layers(self):
        """Define all LoRA layers."""
        self.lora_layers = []
        lora_sizes = [
            (self.channels_in, 64),
            (64, 64), (64, 64), (64, 128), (128, 1024),
            (1024, 512), (512, 256), (256, self.num_classes)
        ]
        for i, (in_size, out_size) in enumerate(lora_sizes, start=1):
            lora_A = nn.Parameter(torch.empty(in_size, self.lora_rank))
            lora_B = nn.Parameter(torch.empty(self.lora_rank, out_size))
            setattr(self, f'l{i}_lora_A', lora_A)
            setattr(self, f'l{i}_lora_B', lora_B)
            self.lora_layers.append((lora_A, lora_B))

#         if not self.train_feat_transf:
#             lora_ft_sizes = [
#                 (64, 64), (64, 128), (128, 1024),
#                 (1024, 512), (512, 256), (256, 64 * 64)
#             ]
#             for j, (in_size, out_size) in enumerate(lora_ft_sizes, start=1):
#                 lora_A_ft = nn.Parameter(torch.empty(in_size, self.lora_rank))
#                 lora_B_ft = nn.Parameter(torch.empty(self.lora_rank, out_size))
#                 setattr(self, f'lft{j}_lora_A', lora_A_ft)
#                 setattr(self, f'lft{j}_lora_B', lora_B_ft)
#                 self.lora_layers.append((lora_A_ft, lora_B_ft))

    def init_lora_layers(self):
        """Initialize LoRA layers."""
#         for lora_A, lora_B in self.lora_layers:
#             nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
#             nn.init.zeros_(lora_B)
        for n,p in self.named_parameters():
            if 'lora' in n:
                if n[-1]=='A':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif n[-1]=='B':
                    nn.init.zeros_(p)

    def freeze_non_lora_weights(self):
        """Freeze weights of non-LoRA layers."""
        for n, p in self.named_parameters():
            if 'lora' not in n:
                p.requires_grad = False
                
    def freeze_non_lora_non_ft(self):
        """Freeze weights of non-LoRA layers."""
        for n, p in self.named_parameters():
            if 'lora' not in n and 'feature_transform' not in n:
                p.requires_grad = False

    def lora_bmm(self, x, layer, lora_A, lora_B):
        """Apply Low-Rank Adaptation (LoRA) to the output of a given layer using batch matrix multiplication."""
        h = layer(x)
        batch, dims, points = x.shape
        AB = lora_A @ lora_B
        x_t = x.transpose(1, 2)
        AB_expanded = AB.unsqueeze(0)
        lora_res = torch.bmm(x_t, AB_expanded.repeat(batch, 1, 1)) * self.lora_alpha
        h += lora_res.transpose(1, 2)
        return h

    def forward(self, x):
        
        num_points = x.shape[1]  # torch.Size([BATCH, N_POINTS, DIMS])

        x_tnet = x[:, :, :self.point_dimension]  # [32, n_points, 3]
        input_transform = self.input_transform(x_tnet)  # [32, 3, 3]
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product

        x_tnet = torch.cat([x_tnet, x[:, :, 3:]], dim=2)  # concat other features if any
        x = x_tnet.transpose(2, 1)  # [batch, dims, n_points]

        # x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        # layer 1 (input size, hidden size)
        x = self.lora_bmm(x, self.conv_1, self.l1_lora_A, self.l1_lora_B)
        x = self.relu(self.bn_1(x))

        # x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 64, N_POINTS] 
        x = self.lora_bmm(x, self.conv_2, self.l2_lora_A, self.l2_lora_B)
        x = self.relu(self.bn_2(x))
        
        # ------------------- Feature Transform T-Net --------------------------
        
        if self.train_feat_transf:
            x = x.transpose(2, 1)  # [batch, n_points, dims]
            feature_transform = self.feature_transform(x)  # [batch, 64, 64]

            x = torch.bmm(x, feature_transform)
            local_point_features = x  # [batch, n_points, 64]
            
        else:
            # Lora to matrix multiplication of feature transform
            
            x_ft = self.lora_bmm(x, self.feature_transform, self.lft1_lora_A, self.lft1_lora_B)
            x_ft = self.relu(self.bn_ft1(x_ft))
            
#             # apply lora to T-Net for Feature Transform
#             # x = F.relu(self.bn_ft1(self.conv_ft1(x)))
#             x_ft = self.lora_bmm(x, self.conv_ft1, self.lft1_lora_A, self.lft1_lora_B)
#             x_ft = self.bn_ft1(x_ft)
#             x_ft = self.relu(x_ft)

#             # x_ft = F.relu(self.bn_ft2(self.conv_ft2(x_ft)))
#             x_ft = self.lora_bmm(x_ft, self.conv_ft2, self.lft2_lora_A, self.lft2_lora_B)
#             x_ft = self.bn_ft2(x_ft)
#             x_ft = self.relu(x_ft)

#             # x_ft = F.relu(self.bn_ft3(self.conv_ft3(x_ft)))  # [batch, 1024, 4096]
#             x_ft = self.lora_bmm(x_ft, self.conv_ft3, self.lft3_lora_A, self.lft3_lora_B)
#             x_ft = self.bn_ft3(x_ft)
#             x_ft = self.relu(x_ft)

#             x_ft = nn.MaxPool1d(num_points)(x_ft)  # [batch, 1024, 1] kernel_size = num_points
#             x_ft = x_ft.view(-1, 1024)

#             # x_ft = F.relu(self.bn_ft4(self.fc_ft1(x_ft)))
#             x_ft = self.lora_linear(x_ft, self.fc_ft1, self.lft4_lora_A, self.lft4_lora_B)
#             x_ft = self.bn_ft4(x_ft)
#             x_ft = self.relu(x_ft)

#             # x_ft = F.relu(self.bn_ft5(self.fc_ft2(x_ft)))
#             x_ft = self.lora_linear(x_ft, self.fc_ft2, self.lft5_lora_A, self.lft5_lora_B)
#             x_ft = self.bn_ft5(x_ft)
#             x_ft = self.relu(x_ft)

#             # x_ft = self.fc_ft3(x_ft)
#             x_ft = self.lora_linear(x_ft, self.fc_ft3, self.lft6_lora_A, self.lft6_lora_B)

#             identity_matrix = torch.eye(64)
#             identity_matrix = identity_matrix.to(self.device)

#             feature_transform = x_ft.view(-1, 64, 64) + identity_matrix
            
#             x = x.transpose(2, 1)  # [batch, n_points, dims]
        
#             x = torch.bmm(x, feature_transform)
#             local_point_features = x  # [batch, n_points, 64]

        # ----------------------------------------------------------------------
        x = x.transpose(2, 1)

        # x = F.relu(self.bn_3(self.conv_3(x)))
        x = self.lora_bmm(x, self.conv_3, self.l3_lora_A, self.l3_lora_B)
        x = self.bn_3(x)
        x = self.relu(x)

        # x = F.relu(self.bn_4(self.conv_4(x)))
        x = self.lora_bmm(x, self.conv_4, self.l4_lora_A, self.l4_lora_B)
        x = self.bn_4(x)
        x = self.relu(x)

        # x = F.relu(self.bn_5(self.conv_5(x)))
        x = self.lora_bmm(x, self.conv_5, self.l5_lora_A, self.l5_lora_B)
        x = self.bn_5(x)
        x = self.relu(x)

        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, 1024)  # [ batch, 1024, 1]

        # FC PointNet classification
        x = self.lora_linear(global_feature, self.fc_1, self.l6_lora_A, self.l6_lora_B)
        x = self.bn_21(x)
        x = self.relu(x)
        x = self.dropout_1(x)

        x = self.lora_linear(x, self.fc_2, self.l7_lora_A, self.l7_lora_B)
        x = self.bn_22(x)
        x = self.relu(x)

        x = self.lora_linear(x, self.classifier, self.l8_lora_A, self.l8_lora_B)

        return F.log_softmax(x, dim=1), feature_transform

        return x


class LoraPointNet(BaseLoraPointNet):
    """
    LoRA-enhanced PointNet without feature transform.
    """
    def __init__(self, lora_rank, num_classes, channels_in=3, point_dimension=3, device='cuda', dropout=0.3):
        super().__init__(lora_rank, num_classes, channels_in, point_dimension, device, dropout, train_feat_transf=False)

class LoraPointNet_ft(BaseLoraPointNet):
    """
    LoRA-enhanced PointNet with feature transform (T-Net).
    """
    def __init__(self, lora_rank, num_classes, channels_in=3, point_dimension=3, device='cuda', dropout=0.3):
        super().__init__(lora_rank, num_classes, channels_in, point_dimension, device, dropout, train_feat_transf=True)
        
        
class LoraPointNet_mm(BaseLoraPointNet):
    """
    LoRA-enhanced PointNet with feature transform (T-Net).
    """
    def __init__(self, lora_rank, num_classes, channels_in=3, point_dimension=3, device='cuda', dropout=0.3):
        super().__init__(lora_rank, num_classes, channels_in, point_dimension, device, dropout, lora_mm=True)
   
     
# Example instantiation
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = BaseLoraPointNet(
#     lora_rank=4,
#     num_classes=40,
#     channels_in=3,
#     point_dimension=1024,
#     device=device,
#     dropout=0.3,
#     train_feat_transf=False
# )
# model.to(device)
