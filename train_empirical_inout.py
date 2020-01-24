import torch
import torchvision
# torch.set_printoptions(profile="full")
from torchvision import transforms
import numpy as np
from models.Q_mnist import QMNIST
from SOS.Q import Q_hinge_loss
from tensorboardX import SummaryWriter
import argparse
import os
from datasets.my_mnist import MyMNIST_oneVSothers
"Q_OPTION_STAGE"
option = lambda w: w.logdir.split('_')[1]
stage = lambda w: w.logdir.split('_')[2]

# TODO: 1. Fer les SVDs d'una vegada
# TODO: 2. Mirar quins moments del veronese estan contribuint mes a aquells pics extranys en validacio
# TODO: 3. 

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


def train_model(model, optimizer, epochs, train_dl, val_dl, wr, idx_inliers, device, weights_path):
    number_of_batches_per_epoch = len(iter(train_dataloader))
    number_of_batches_per_epoch_validation = len(iter(val_dataloader))
    
    # Loss function
    mse = torch.nn.MSELoss()
    lambda_reconstruction = torch.tensor([0.1]).cuda('cuda:'+str(device))
    lambda_q = torch.tensor([1.0]).cuda('cuda:'+str(device))
    
    # TRAINING PROCESS
    count_inliers, count_outliers = 0, 0
    for i in range(0, n_epochs):
        # TRAINING
        for batch_idx, (sample, label) in enumerate(train_dataloader):
            
            inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
            
            idx_in = label==idx_inliers
            idx_out = label!= idx_inliers
            
            inliers = inputs[idx_in]
            outliers = inputs[idx_out]

            optimizer.zero_grad()
            # Set training for Q (Update Minv)
            model.Q.set_eval(False)
            z, q, rec = model(inliers)
            # compute loss function
            reconstruction_loss = lambda_reconstruction * mse(inliers, rec)
            q_loss_in = lambda_q * torch.sum(torch.abs(q))/idx_in.size()[0]
            # Set evaluation mode for Q
            model.Q.set_eval(True)
            z_out, q_out, rec_out = model(outliers)
            q_loss_out = lambda_q * torch.sum(torch.abs(q_out))
            total_loss = reconstruction_loss + q_loss_in - q_loss_out

            # Write results
            step = ((i*number_of_batches_per_epoch) + batch_idx)
            norm_of_z = torch.trace(torch.matmul(z,z.t()))
            if(option(wr)=='0'):
                # Q_0_X
                write_train_results(step, reconstruction_loss, q_loss_in, model.Q.get_norm_of_B(), norm_of_z, writer)
            elif(option(wr)=='1'):
                # Q_1_X
                write_train_results(step, reconstruction_loss, q_loss_in, model.Q.get_norm_of_ATA(), norm_of_z, writer)
                print("Writing Q_1_X")
            elif(option(wr)=='M'):
                write_train_results(step, reconstruction_loss, q_loss_in, torch.trace(torch.matmul(model.Q.M_inv_copy.t(), model.Q.M_inv_copy)), norm_of_z, writer)
                print("Writing")

            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
        if(i==2 and stage(wr)=='1'):
            freeze_ENC_DEC(model)
        if(weights_path is not None):
            torch.save(model.state_dict(), os.path.join(weights_path+str(i)))
        
        model.Q.set_eval(True)
        
        # VALIDATION
        with torch.no_grad():
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
                        # writer.add_image('inlier/'+str(count_inliers), inputs_in[i_q_in,0,:,:].cpu().numpy().reshape(1,28,28), count_inliers)
                        writer.add_scalar('val_loss/q_loss_in', q_in[i_q_in].item(), count_inliers)
                        count_inliers += 1

                if(q_out.size()[0]>0):
                    for i_q_out in range(number_outliers):
                        # writer.add_image('outlier/'+str(count_outliers), inputs_out[i_q_out,0,:,:].cpu().numpy().reshape(1,28,28), count_outliers)
                        writer.add_scalar('val_loss/q_loss_out', q_out[i_q_out].item(), count_outliers)
                        count_outliers += 1
                
                writer.add_scalars('val_loss/q_loss', {'inliers_q_loss': q_loss_in.item(),'outliers_q_loss': q_loss_out.item()}, step)
            model.Q.set_eval(False)



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train encoder decoder to learn moment matrix.')
    parser.add_argument('--model', help="Available models:\n \
                                            1. Q_Bilinear (learns M_inv = A directly using torch.nn.Bilinear)\n \
                                            2. Q (Learns M_inv = A) \n \
                                            3. Q_PSD (Learns M_inv = A.T*A so M is PSD)\n \
                                            4. Q_M_Batches")
    parser.add_argument('--writer', help="Name of the session that will be opened by tensorboard X")
    parser.add_argument('--idx_inliers', help="Digit considered as inlier.")
    parser.add_argument('--device', help="cuda device")
    parser.add_argument('--weights', default=None, help="Path to where the weights will be saved.")
    args = parser.parse_args()

    # DATASETS & DATALOADERS
    mnist = MyMNIST_oneVSothers(int(args.idx_inliers))
    mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
    bs = 64
    train_dataloader = torch.utils.data.DataLoader(mnist, batch_size=bs, drop_last=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=100, drop_last=True, shuffle=False)

    # MODEL
    model = QMNIST((1,28,28), 64, args.model)   
    device = args.device
    model = model.cuda('cuda:'+str(device))
    # TensorboardX
    writer = SummaryWriter('runs/'+str(args.writer))
    # TRAINING PARAMS
    n_epochs = 100
    optimizer = torch.optim.Adam([{'params': model.encoder.parameters()}, {'params': model.decoder.parameters()}, {'params': model.Q.parameters(), 'lr': 1e-2}], lr=1e-3)
    # bla = iter((train_dataloader))
    # fuet = bla.next()
    # print(fuet[1])
    # pernil = bla.next()
    # print(pernil[1])
    train_model(model, optimizer, n_epochs, train_dataloader, val_dataloader, writer, int(args.idx_inliers), device, args.weights)