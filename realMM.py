import torch
import torchvision
from torchvision import transforms
import numpy as np
from models.Q_mnist import QMNIST
from models.Q_mnist import QMNIST_PSD
from models.Q_mnist import QMNIST_real_M
from tensorboardX import SummaryWriter
import argparse
import os

def print_losses(rec, q, tot, numerador, denominador):
    # print_losses(float(reconstruction_loss.item()), float(q_loss.item()), float(total_loss.item()), batch_idx, number_of_batches_per_epoch)
    print("{:0.2f}%  REC = {:0.2f} | Q = {:0.2f}  | TOTAL = {:0.2f} ".format((numerador/denominador)*100.0,rec, q, tot))


def select_idx(mnist, idx):
    # Select the digit we are considering as inlier
    idx_inliers = idx
    idxs = mnist.train_labels == idx_inliers
    mnist.train_labels = mnist.train_labels[idxs]
    mnist.train_data = mnist.train_data[idxs]
    return mnist


def freeze_ENC_DEC(model):
    # Freeze encoder-decoder
    for name,param in model.encoder.named_parameters():
        param.requires_grad = False
    for name,param in model.decoder.named_parameters():
        param.requires_grad = False


def write_train_results(step, reconstruction_loss, q_loss, norm_A, norm_of_z, writer):
    writer.add_scalar('train_loss/norm_z', norm_of_z, step)
    writer.add_scalar('train_loss/rec_loss', reconstruction_loss, step)
    writer.add_scalar('train_loss/Q_loss', q_loss, step)
    writer.add_scalar('train_loss/norm_A', norm_A, step)


def create_M(model, train_dataloader, val_dataloader, wr, idx_inliers, device):
    
    print("Im going to gather samples to create M")

    bs_totals = len(train_dataloader)
    with torch.no_grad():
        # TRAINING
        for batch_idx, (sample, label) in enumerate(train_dataloader):
            inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
            z, q = model(inputs)
            print(batch_idx/bs_totals)
        model.Q.create_M()
    print("M created")
    # VALIDATION
    with torch.no_grad():
        for batch_idx, (sample, label) in enumerate(val_dataloader):
            # Separate between inliers and outliers
            inputs = sample.view(100,1,28,28).float().cuda('cuda:'+str(device))
            z, q = model(inputs)
            # compute loss function
            inliers = label == idx_inliers
            outliers = label != idx_inliers

            inputs_in = inputs[inliers, :, :, :]
            z_in = z[inliers]
            q_in = q[inliers]
            # rec_in = rec[inliers]
            # rec_loss_in = lambda_reconstruction * mse(inputs_in, rec_in)
            q_loss_in = torch.sum(torch.abs(q_in))/q_in.size()[0]
            
            inputs_out = inputs[outliers]
            z_out = z[outliers]
            q_out = q[outliers]
            # rec_out = rec[outliers]
            # rec_loss_out = lambda_reconstruction * mse(inputs_out, rec_out)
            q_loss_out = torch.sum(torch.abs(q_out))/q_out.size()[0]

            step = batch_idx
            if(q_in.size()[0]>0):
                writer.add_image('inlier/'+str(step)+'_q_'+str(q_in[0]), inputs_in[0,0,:,:].cpu().numpy().reshape(1,28,28), step)
            elif(q_out.size()[0]>0):
                writer.add_image('outlier/'+str(step)+'_q_'+str(q_out[0]), inputs_out[0].cpu().numpy().reshape(1,28,28), step)
            # writer.add_scalars('val_loss/rec_loss', {'inliers_rec_loss': rec_loss_in.item(),'outliers_rec_loss': rec_loss_out.item()}, step)
            writer.add_scalars('val_loss/q_loss', {'inliers_q_loss': q_loss_in.item(),'outliers_q_loss': q_loss_out.item()}, step)



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train encoder decoder to learn moment matrix.')
    parser.add_argument('--model', help="Available models:\n 1. Q (learns M_inv directly)\n 2. Q_PSD (Learns M_inv = A.T*A so M is PSD)")
    parser.add_argument('--writer', help="Name of the session that will be opened by tensorboard X")
    parser.add_argument('--idx_inliers', help="Digit considered as inlier.")
    parser.add_argument('--device', help="cuda device")
    parser.add_argument('--weights', help="Path to weights")
    args = parser.parse_args()

    # DATASETS & DATALOADERS
    mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    idx_inliers = int(args.idx_inliers)
    mnist = select_idx(mnist, idx_inliers)
    mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    bs = 512
    train_dataloader = torch.utils.data.DataLoader(mnist, batch_size=bs, drop_last=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=100, drop_last=True, shuffle=False)

    # MODEL
    if(args.model == 'Q'):
        model = QMNIST((1,28,28), 64, 1)   
    elif(args.model == 'Q_PSD'):
        model = QMNIST_PSD((1,28,28), 64,1)
    elif(args.model == 'Q_real_M'):
        model = QMNIST_real_M((1,28,28), 64,1)
    else:
        model = None

    device = args.device
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model = model.cuda('cuda:'+str(device))
    
    # TensorboardX
    writer = SummaryWriter('runs/'+str(args.writer))
    # TRAINING PARAMS
    
    create_M(model, train_dataloader, val_dataloader, writer, idx_inliers, device)
    