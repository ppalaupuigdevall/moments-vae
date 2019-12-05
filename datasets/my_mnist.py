import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

def select_idx(mnist, idx):
    # Select the digit we are considering as inlier
    idx_inliers = idx
    idxs = mnist.train_labels == idx_inliers
    mnist.train_labels = mnist.train_labels[idxs]
    mnist.train_data = mnist.train_data[idxs]
    return mnist





class MyMNIST_oneclass(torch.utils.data.Dataset):
    
    def __init__(self, idx_inliers):
        super(MyMNIST_oneclass, self).__init__()
        self.idx_inliers = idx_inliers
        self.mnist_in = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
        self.mnist_in = select_idx_in(self.mnist_in, self.idx_inliers)

    def __getitem__(self, idx):
        return self.mnist_in.train_data[idx], self.mnist_in.train_labels[idx]

    def __len__(self):
        return len(self.mnist_in.train_labels)



def select_idx_1_VS_others(mnist, idx_inliers):
    idxs_in = mnist.train_labels == idx_inliers
    idxs_out = mnist.train_labels != idx_inliers
    data_in = mnist.train_data[idxs_in]
    labels_in = mnist.train_labels[idxs_in]
    data_out = mnist.train_data[idxs_out]
    labels_out = mnist.train_labels[idxs_out]
    return (data_in, labels_in),(data_out, labels_out)


class MyMNIST_oneVSothers(torch.utils.data.Dataset):
    def __init__(self, idx_inliers):
        super(MyMNIST_oneVSothers, self).__init__()
        self.idx_inliers = idx_inliers
        self.mnist = torchvision.datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]))
        # self.mnist_in is a tuple with (data_inliers, labels_inliers)
        self.mnist_in, self.mnist_out = select_idx_1_VS_others(self.mnist, self.idx_inliers)
        self.give_inlier = False
    
    def __getitem__(self, idx):
        if(self.give_inlier):
            self.give_inlier = False
            return self.mnist_in[0][idx], self.mnist_in[1][idx]
        else:
            self.give_inlier = True
            return self.mnist_out[0][idx], self.mnist_out[1][idx]

    def __len__(self):
        return len(self.mnist_in[1]) + len(self.mnist_out[1])
    