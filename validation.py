import torch
import torchvision
from torchvision import transforms
import numpy as np
from models.Q_mnist import QMNIST
from models.Q_mnist import QMNIST_PSD
from tensorboardX import SummaryWriter
import argparse
import os


def select_idx(mnist, idx):
    # Select the digit we are considering as inlier
    idx_inliers = idx
    idxs = mnist.train_labels == idx_inliers
    mnist.train_labels = mnist.train_labels[idxs]
    mnist.train_data = mnist.train_data[idxs]
    return mnist

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train encoder decoder to learn moment matrix.')
    parser.add_argument('--model', help="Available models:\n 1. Q (learns M_inv directly)\n 2. Q_PSD (Learns M_inv = A.T*A so M is PSD)")
    parser.add_argument('--weights', help="Path to weights")
    parser.add_argument('--idx_inliers', help="Digit considered as inlier.")
    parser.add_argument('--device', help="cuda device")
    parser.add_argument('--writer', help="writer tensorboardX")
    args = parser.parse_args()
    
    # DATASETS & DATALOADERS
    mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    idx_inliers = int(args.idx_inliers)
    mnist = select_idx(mnist, idx_inliers)
    mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    bs = 16
    val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=bs, drop_last=True, shuffle=False)
    
    writer = SummaryWriter('runs/'+str(args.writer))


    # MODEL
    if(args.model == 'Q'):
        model = QMNIST((1,28,28), 64, 1)   
    elif(args.model == 'Q_PSD'):
        model = QMNIST_PSD((1,28,28), 64,1)
    else:
        model = None

    device = args.device
    
    # LOAD MODEL
    with torch.no_grad():
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        model = model.cuda('cuda:'+str(device))
        # VALIDATION
        # Loss function
        mse = torch.nn.MSELoss()
        lambda_reconstruction = torch.tensor([0.001]).cuda('cuda:'+str(device))
        lambda_q = torch.tensor([1.0]).cuda('cuda:'+str(device))
   
    with torch.no_grad():
        for batch_idx, (sample, label) in enumerate(val_dataloader):
            # Separate between inliers and outliers
            inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
            z, q, rec = model(inputs)
            # compute loss function
            inliers = label == idx_inliers
            outliers = label != idx_inliers

            inputs_in = inputs[inliers, :, :, :]
            z_in = z[inliers]
            q_in = q[inliers]
            rec_in = rec[inliers]
            rec_loss_in = lambda_reconstruction * mse(inputs_in, rec_in)
            q_loss_in = lambda_q * torch.sum(torch.abs(q_in))/q_in.size()[0] # BUG SUPER BUG OSTIAAAA 
            
            inputs_out = inputs[outliers]
            z_out = z[outliers]
            q_out = q[outliers]
            rec_out = rec[outliers]
            rec_loss_out = lambda_reconstruction * mse(inputs_out, rec_out)
            q_loss_out = lambda_q * torch.sum(torch.abs(q_out))/q_out.size()[0] # SUPER BUG BUG BUG BUG BUG !! OSTIA BUG!!! COM POTS SER TANT INUTIL!!!!!!! 
            print("Q_IN = {:0.2f}".format(q_loss_in.item()))
            print("Q_OUT = {:0.2f}".format(q_loss_out.item()))
            print("Norm of A = " + str(model.Q.get_norm_of_B()))
            step = batch_idx
            # if(q_in.size()[0]>0):
            #     writer.add_image('inlier/'+str(step)+'_q_'+str(q_in[0]), inputs_in[0,0,:,:].cpu().numpy().reshape(1,28,28), step)
            # elif(q_out.size()[0]>0):
            #     writer.add_image('outlier/'+str(step)+'_q_'+str(q_out[0]), inputs_out[0].cpu().numpy().reshape(1,28,28), step)
            # writer.add_scalars('val_loss/rec_loss', {'inliers_rec_loss': rec_loss_in.item(),'outliers_rec_loss': rec_loss_out.item()}, step)
            writer.add_scalars('val_loss/q_loss', {'inliers_q_loss': q_loss_in.item(),'outliers_q_loss': q_loss_out.item()}, step)