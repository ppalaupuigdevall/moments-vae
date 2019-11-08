import torch
import torchvision
from torchvision import transforms
import numpy as np
from models.Q_mnist import QMNIST
from models.Q_mnist import QMNIST_PSD
from tensorboardX import SummaryWriter
import argparse
import os


# DATASETS & DATALOADERS
mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
# Select the digit we are considering as inlier
idx_inliers = 1
idxs = mnist.train_labels == idx_inliers
mnist.train_labels = mnist.train_labels[idxs]
mnist.train_data = mnist.train_data[idxs]

mnist_test = torchvision.datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))

bs = 32
train_dataloader = torch.utils.data.DataLoader(mnist, batch_size=bs, drop_last=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=bs, drop_last=True)
model = QMNIST((1,28,28), 64, 1)
model.load_state_dict(torch.load('/data/Ponc/learning_M_inv_0'))
model = model.cuda()
number_of_batches_per_epoch_validation = len(iter(val_dataloader))
print("Number of batches per epoch = " + str(number_of_batches_per_epoch_validation))

print("Norma de A = " + str(model.Q.get_norm_of_B()))
A = model.Q.B.weight
print("La suma de la diagonal de la matriu al quadrat = ")
print(torch.sum(torch.diag(model.Q.B.weight.view(2145,2145))**2))
print("A.T*A")
print(torch.trace(torch.matmul(model.Q.B.weight.view(2145,2145).t(), model.Q.B.weight.view(2145,2145))))
# m_save = model.Q.B.weight.view(2145,2145).cpu().detach().numpy()
# np.save('/data/Ponc/A.npy', m_save)

print("Valor maxim de A")
print(torch.max(model.Q.B.weight))
with torch.no_grad():
    for batch_idx, (sample, label) in enumerate(val_dataloader):
        # Separate between inliers and outliers
        inputs = sample.view(bs,1,28,28).float().cuda(async=True)
        z, q, rec = model(inputs)
        # compute loss function
        inliers = label == idx_inliers
        outliers = label != idx_inliers

        inputs_in = inputs[inliers, :, :, :]
        z_in = z[inliers]
        q_in = q[inliers]
        rec_in = rec[inliers]
        
        
        inputs_out = inputs[outliers]
        z_out = z[outliers]
        q_out = q[outliers]
        rec_out = rec[outliers]
       
        if(q_in.size()[0]>0):
            # writer.add_image('inlier/'+str((i*number_of_batches_per_epoch_validation)+batch_idx)+'_q_'+str(q_in[0]), inputs_in[0,0,:,:].cpu().numpy().reshape(1,28,28), (i*number_of_batches_per_epoch_validation)+batch_idx)
            print("in")
        elif(q_out.size()[0]>0):
            # writer.add_image('outlier/'+str((i*number_of_batches_per_epoch_validation)+batch_idx)+'_q_'+str(q_out[0]), inputs_out[0].cpu().numpy().reshape(1,28,28), (i*number_of_batches_per_epoch_validation)+batch_idx)
            print("out")
        # writer.add_scalars('val_loss/rec_loss', {'inliers_rec_loss': rec_loss_in.item(),'outliers_rec_loss': rec_loss_out.item()}, (i*number_of_batches_per_epoch_validation)+batch_idx)
        # writer.add_scalars('val_loss/q_loss', {'inliers_q_loss': q_loss_in.item(),'outliers_q_loss': q_loss_out.item()}, (i*number_of_batches_per_epoch_validation)+batch_idx)


