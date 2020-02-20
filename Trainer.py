import torch
import torchvision
from torchvision import transforms
import numpy as np

from enum import Enum
# from models.Q_mnist import QMNIST
from models.vae_model_wen import ConvVAE
from utils import print_losses
from utils import freeze_ENC_DEC
from utils import write_train_results
from torch.nn import functional as F


def loss_function(recon_x, x, mu, logvar):
    # reconstruction loss
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

"Q_OPTION_STAGE"
option = lambda w: w.logdir.split('_')[1]
stage = lambda w: w.logdir.split('_')[2]


class TrainMode(Enum):
        
        rec_and_empirical_M = 0 # Train an autoencoder until it gets good reconstruction in MNIST, then 
                                # build a moment matrix with all the samples of the training set to see
                                # if it can discriminate between inliers and outliers.
                                    
        rec_and_Q = 1           # Here we train the AE with reconstruction error and Christoffel function 
                                # L = MSE(X,Xout) + v(z).T * inv(M) * v(z) , both terms at the same time

        rec_then_Q = 2          # Here we train the AE until it gets good reconstruction error, then we
                                # freeze its parameters and try to learn the inverse of the moment matrix, 
                                # different settings of the matrix are tested (Q, Q_PSD, etc...)


class Trainer:
    # The way we train the AE depends on the TrainMode we have selected
    def __init__(self, train_mode=TrainMode.rec_and_Q, Q_option='Q'):
        self.train_mode = train_mode
        self.Q_option = Q_option
        self.model = None
    
    # def train(self, n_epochs, train_dataloader, val_dataloader, writer, idx_inliers, device):
    def train(self, n_epochs, train_dataloader, val_dataloader, idx_inliers, device):
        
        if(self.train_mode == TrainMode.rec_and_empirical_M):
            # self.model = QMNIST((1,28,28), 64, 'Q_Real_M')
            self.model = ConvVAE((1,28,28), 64, 'Q_Real_M')
            self.model = self.model.cuda('cuda:'+str(device))
            optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},{'params': self.model.decoder.parameters()}, \
                                          {'params': self.model.Q.parameters(), 'lr': 1e-2}], lr=1e-3)
            # self.train_rec_and_empirical_M(self.model, optimizer, n_epochs, train_dataloader, val_dataloader, writer, idx_inliers, device)
            self.train_rec_and_empirical_M(self.model, optimizer, n_epochs, train_dataloader, val_dataloader, idx_inliers, device)

        elif(self.train_mode == TrainMode.rec_and_Q or \
             self.train_mode == TrainMode.rec_then_Q):
            
            if(Q_option != 'Q' or Q_option != 'Q_PSD' or Q_option != 'Q_Bilinear'):
                raise ValueError('Possible Qs: Q_Bilinear, Q, Q_PSD')
            self.model = QMNIST((1,28,28), 64, mode=Q_option)
            self.model = self.model.cuda('cuda:'+str(device))
            optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},{'params': self.model.decoder.parameters()}, \
                                          {'params': self.model.Q.parameters(), 'lr': 1e-2}], lr=1e-3)

            if(self.train_mode == TrainMode.rec_and_Q):
                self.train_rec_and_Q(self.model, optimizer, n_epochs, train_dataloader, val_dataloader, idx_inliers, device)  # have writer before idx_inliers
            elif(self.train_mode == TrainMode.rec_then_Q): 
                self.train_rec_then_Q(self.model, optimizer, n_epochs, train_dataloader, val_dataloader, idx_inliers, device) # have writer before idx_inliers


    def train_rec_and_empirical_M(self, model, optimizer, n_epochs, train_dataloader, val_dataloader, idx_inliers, device): # have writer before idx_inliers

        # Train an autoencoder until it gets good reconstruction in MNIST (n_epochs), then 
        # build a moment matrix with all the samples of the training set to see
        # if it can discriminate between inliers and outliers.
        number_of_batches_per_epoch = len(iter(train_dataloader))
        number_of_batches_per_epoch_validation = len(iter(val_dataloader))
        print("Number of batches per epoch in training = " + str(number_of_batches_per_epoch))
        print("Number of batches per epoch in validation = " + str(number_of_batches_per_epoch_validation))

        # Loss function
        mse = torch.nn.MSELoss()
        lambda_reconstruction = torch.tensor([0.1]).cuda('cuda:'+str(device))

        # TRAINING PROCESS
        bs = train_dataloader.batch_size
        count_inliers, count_outliers = 0, 0
        for i in range(0, n_epochs):
            print("Training begins")
            # TRAINING
            for batch_idx, (sample, label) in enumerate(train_dataloader):
                inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
            
                optimizer.zero_grad()
                z, q, rec ,mu, logvar= model(inputs)
                # compute loss function
                # reconstruction_loss = lambda_reconstruction * mse(inputs, rec)
                reconstruction_loss = loss_function(rec, inputs, mu, logvar)
                # train_loss += reconstruction_loss.item()
                
                # Write results to TENSORBOARD UNCOMMENT TO DISPLAY THEM
                # step = ((i*number_of_batches_per_epoch) + batch_idx)
                # norm_of_z = torch.trace(torch.matmul(z,z.t()))
                # if(option(wr)=='0'):
                #     # Q_0_X
                #     write_train_results(step, reconstruction_loss, q_loss, model.Q.get_norm_of_B(), norm_of_z, writer)
                # elif(option(wr)=='1'):
                #     # Q_1_X
                #     write_train_results(step, reconstruction_loss, q_loss, model.Q.get_norm_of_ATA(), norm_of_z, writer)
                #     print("Writing Q_1_X")
                # elif(option(wr)=='M'):
                #     write_train_results(step, reconstruction_loss, q_loss, torch.trace(torch.matmul(model.Q.M_inv_copy.t(), model.Q.M_inv_copy)), norm_of_z, writer)
                #     print("Writing")

                # Backpropagate & step
                reconstruction_loss.backward()
                optimizer.step()
                print(batch_idx/number_of_batches_per_epoch)


        # Now that it achieves good reconstruction, we freeze the model and create the empirical moment matrix
        freeze_ENC_DEC(model)
        print("Model frozen")
        # if(weights_path is not None):
        #     torch.save(model.state_dict(), os.path.join(weights_path+str(i)))
        
        # CREATE MOMENT MATRIX
        # 0. Set buil_M to True (to save veronese maps)
        model.Q.set_build_M()
        print("buil_M set to True")
        # 1. We go through all the training set
        with torch.no_grad():
            for batch_idx, (sample, label) in enumerate(train_dataloader):
                inputs = sample.view(bs,1,28,28).float().cuda('cuda:'+str(device))
                _, _, _,_,_ = model(inputs)
            # 2. Create empiricall moment matrix
            model.Q.create_M() 
            print("Moment matrix created")

            # Now that we have moment matrix, apply it!
            # VALIDATION
            for batch_idx, (sample, label) in enumerate(val_dataloader):
                # Separate between inliers and outliers
                inputs = sample.view(100,1,28,28).float().cuda('cuda:'+str(device))
                z, q, rec, mu, logvar = model(inputs)
                # compute loss function
                inliers = label == idx_inliers
                outliers = label != idx_inliers
                
                inputs_in = inputs[inliers, :, :, :]
                z_in = z[inliers]
                q_in = q[inliers]
                rec_in = rec[inliers]
                mu_in = mu[inliers]
                logvar_in = logvar[inliers]
                # rec_loss_in = lambda_reconstruction * mse(inputs_in, rec_in)
                rec_loss_in = loss_function(rec_in, inputs_in, mu_in, logvar_in)
                q_loss_in = torch.sum(torch.abs(q_in))/q_in.size()[0]
                
                inputs_out = inputs[outliers]
                z_out = z[outliers]
                q_out = q[outliers]
                rec_out = rec[outliers]
                mu_out = mu[outliers]
                logvar_out = logvar[outliers]
                # rec_loss_out = lambda_reconstruction * mse(inputs_out, rec_out)
                rec_loss_out = loss_function(rec_out, inputs_out, mu_out, logvar_out)
                q_loss_out = torch.sum(torch.abs(q_out))/q_out.size()[0]

                step = ((i*number_of_batches_per_epoch_validation)+batch_idx)
                number_inliers = q_in.size()[0]
                number_outliers = q_out.size()[0]
                
                '''
                # TENSORBOARD
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
                '''
