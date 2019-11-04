import torch
import torchvision
import numpy as np
from models.Q_mnist import QMNIST
from tensorboardX import SummaryWriter

def print_losses(rec, q, tot):
    print("REC = {0:.2f} | Q = {0:.2f}  | TOTAL = {0:.2f} ".format(rec, q, tot))


# TensorboardX
writer = SummaryWriter('runs/learn_M')


# DATASETS
mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True)
# Select the digit we are considering as inlier
idx_inliers = 1
idxs = np.where(mnist.train_labels.numpy()==idx_inliers)[0]
training_samples = mnist.train_data[idxs]
number_of_samples = len(training_samples)
train_set = training_samples[0:int(0.7*number_of_samples)]
val_set = training_samples[int(0.7*number_of_samples)+1:]
bs = 16
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=bs,drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=bs,drop_last=True)


# MODEL
model = QMNIST((1,28,28), 64, 1)
model = model.cuda()
mse = torch.nn.MSELoss()


# TRAINING PARAMS
n_epochs = 30
lambda_reconstruction = 0.0001
lambda_q = 1.0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# TRAINING PROCESS
for i in range(0, n_epochs):
    for sample in enumerate(train_dataloader):
        

        inputs = sample[1].view(bs,1,28,28).float().cuda()
        z, q, rec = model(inputs)
        print(inputs.size())
        print(rec.size())
        # compute loss function
        reconstruction_loss = mse(inputs, rec)
        print(reconstruction_loss.item())
        q_loss = torch.sum(torch.abs(q))
        print(q_loss.item())
        # total_loss = lambda_reconstruction * reconstruction_loss + lambda_q * q_loss # FIXME
        total_loss = reconstruction_loss + q_loss
        print_losses(float(reconstruction_loss.item()), float(q_loss.item()), float(total_loss.item()))

        total_loss.backward()
        optimizer.step()
    
    # Validation
