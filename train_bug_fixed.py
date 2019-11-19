import torch
import torchvision
from torchvision import transforms
import numpy as np
from models.Q_mnist import QMNIST
from models.Q_mnist import QMNIST_PSD
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


def train_model(model, optimizer, epochs, train_dl, val_dl, wr, idx_inliers, device):
    number_of_batches_per_epoch = len(iter(train_dataloader))
    number_of_batches_per_epoch_validation = len(iter(val_dataloader))
    
    # Loss function
    mse = torch.nn.MSELoss()
    lambda_reconstruction = torch.tensor([0.001]).cuda('cuda:'+str(device))
    lambda_q = torch.tensor([1.0]).cuda('cuda:'+str(device))
   
    # TRAINING PROCESS
    for i in range(0, n_epochs):
        # TRAINING
        for batch_idx, (sample, label) in enumerate(train_dataloader):
            inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
            optimizer.zero_grad()
            z, q, rec = model(inputs)
            # compute loss function
            reconstruction_loss = lambda_reconstruction * mse(inputs, rec)
            q_loss = lambda_q * torch.sum(torch.abs(q))/bs 
            total_loss = torch.add(q_loss, reconstruction_loss)
            # Write results
            step = ((i*number_of_batches_per_epoch) + batch_idx)
            norm_of_z = torch.trace(torch.matmul(z,z.t()))
            # Q_0_0
            write_train_results(step, reconstruction_loss, q_loss, model.Q.get_norm_of_B(), norm_of_z, writer)
            # Q_1_0
            # write_train_results(step, reconstruction_loss, q_loss, model.Q.get_norm_of_ATA(), norm_of_z, writer)
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            break
            
        # if(i==0):
        #     freeze_ENC_DEC(model)

        # torch.save(model.state_dict(), os.path.join('/data/Ponc/Q_0_1_FIX_no_BN/'+str(i)))
        
        # VALIDATION
        with torch.no_grad():
            # model = model.eval()
            for batch_idx, (sample, label) in enumerate(val_dataloader):
                # Separate between inliers and outliers
                inputs = sample.view(100,1,28,28).float().cuda('cuda:'+str(device))
                z, q, rec = model(inputs)
                # compute loss function
                inliers = label == idx_inliers
                outliers = label != idx_inliers

                inputs_in = inputs[inliers, :, :, :]
                z_in = z[inliers]
                q_in = q[inliers]
                rec_in = rec[inliers]
                rec_loss_in = lambda_reconstruction * mse(inputs_in, rec_in)
                q_loss_in = lambda_q * torch.sum(torch.abs(q_in))/q_in.size()[0]
                
                inputs_out = inputs[outliers]
                z_out = z[outliers]
                q_out = q[outliers]
                rec_out = rec[outliers]
                rec_loss_out = lambda_reconstruction * mse(inputs_out, rec_out)
                q_loss_out = lambda_q * torch.sum(torch.abs(q_out))/q_out.size()[0]

                step = ((i*number_of_batches_per_epoch_validation)+batch_idx)
                number_inliers = q_in.size()[0]
                number_outliers = q_out.size()[0]
                if(q_in.size()[0]>0):
                    for i_q_in in range(number_inliers):
                        writer.add_image('inlier/'+str(step)+'_q_'+str(i_q_in), inputs_in[i_q_in,0,:,:].cpu().numpy().reshape(1,28,28), step+i_q_in)
                        writer.add_scalar('val_loss/q_loss_in_indep', q_loss_in[i_q_in].item(), step+i_q_in)
                elif(q_out.size()[0]>0):
                    for i_q_out in range(number_outliers):
                        writer.add_image('outlier/'+str(step)+'_q_'+str(i_q_out), inputs_out[i_q_out,0,:,:].cpu().numpy().reshape(1,28,28), step+i_q_out)
                        writer.add_scalar('val_loss/q_loss_ouy_indep', q_loss_out[i_q_out].item(), step+i_q_out)

                
                
                writer.add_scalars('val_loss/rec_loss', {'inliers_rec_loss': rec_loss_in.item(),'outliers_rec_loss': rec_loss_out.item()}, step)
                writer.add_scalars('val_loss/q_loss', {'inliers_q_loss': q_loss_in.item(),'outliers_q_loss': q_loss_out.item()}, step)



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train encoder decoder to learn moment matrix.')
    parser.add_argument('--model', help="Available models:\n 1. Q (learns M_inv directly)\n 2. Q_PSD (Learns M_inv = A.T*A so M is PSD)")
    parser.add_argument('--writer', help="Name of the session that will be opened by tensorboard X")
    parser.add_argument('--idx_inliers', help="Digit considered as inlier.")
    parser.add_argument('--device', help="cuda device")
    args = parser.parse_args()

    # DATASETS & DATALOADERS
    mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    idx_inliers = int(args.idx_inliers)
    mnist = select_idx(mnist, idx_inliers)
    mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    bs = 64
    train_dataloader = torch.utils.data.DataLoader(mnist, batch_size=bs, drop_last=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=100, drop_last=True, shuffle=False)

    # MODEL
    if(args.model == 'Q'):
        model = QMNIST((1,28,28), 64, 1)   
    elif(args.model == 'Q_PSD'):
        model = QMNIST_PSD((1,28,28), 64,1)
    else:
        model = None

    device = args.device
    model = model.cuda('cuda:'+str(device))

    # TensorboardX
    writer = SummaryWriter('runs/'+str(args.writer))
    # TRAINING PARAMS
    n_epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, optimizer, n_epochs, train_dataloader, val_dataloader, writer, idx_inliers, device)