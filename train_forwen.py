import torch
import torchvision
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import os

from utils import select_idx

from Trainer import Trainer
from Trainer import TrainMode

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train encoder decoder to learn moment matrix.')
    parser.add_argument('--model', help="Available models:\n 1. Q_Bilinear (learns M_inv directly using torch.nn.Bilinear)\n 2. Q (Learns M_inv = A) \n 3.  Q_PSD (Learns M_inv = A.T*A so M is PSD)\n 4. Q_M_Batches")
    parser.add_argument('--writer', help="Name of the session that will be opened by tensorboard X")
    parser.add_argument('--idx_inliers', help="Digit considered as inlier.")
    parser.add_argument('--device', help="cuda device")
    parser.add_argument('--weights', default=None, help="Path to where the weights will be saved.")
    args = parser.parse_args()
    
    model_type = str(args.model)
    writer = SummaryWriter('runs/'+str(args.writer)) # TensorboardX
    idx_inliers = int(args.idx_inliers)
    device = args.device
    
    # DATASETS & DATALOADERS
    mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    mnist = select_idx(mnist, idx_inliers)
    mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    bs = 64
    train_dataloader = torch.utils.data.DataLoader(mnist, batch_size=bs, drop_last=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=100, drop_last=True, shuffle=False)
    
    # TRAINING PARAMS
    n_epochs = 4
    train_mode = TrainMode.rec_and_empirical_M
    trainer = Trainer(train_mode=train_mode, Q_option=model_type)
    trainer.train(n_epochs, train_dataloader, val_dataloader, writer, idx_inliers, device)