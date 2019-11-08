from os.path import join
from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import LSALoss
from utils import novelty_score
from utils import normalize

from tensorboardX import SummaryWriter

import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
matplotlib.use("agg")
from SOS.moments import generateMoments
from SOS.moments import Q
from visualizator import visualize_instance_on_tensorboard
from scipy.special import comb
writer = SummaryWriter('runs/exp_cifar10_10')
class OneClassResultHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, checkpoints_dir, output_file):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file

        # Set up loss function
        self.loss = LSALoss(cpd_channels=100)

    @torch.no_grad()
    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        # Prepare a table to show results
        oc_table = self.empty_table

        # Set up container for metrics from all classes
        all_metrics = []

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            if(cl_idx == 2):
                lala = True
                print('cl_idx ' +str(cl_idx))
                print('cl '+ str(cl))
                # Load the checkpoint
                self.model.load_w(join(self.checkpoints_dir, f'{cl}.pkl'))

                # First we need a run on validation, to compute
                # normalizing coefficient of the Novelty Score (Eq.9)
                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

                # Run the actual test
                self.dataset.test(cl)
                loader = DataLoader(self.dataset)

                sample_llk = np.zeros(shape=(len(loader),))
                sample_rec = np.zeros(shape=(len(loader),))
                sample_y = np.zeros(shape=(len(loader),))
                print("Length of the dataset = " + str(len(self.dataset)))
                
                labels = np.zeros((len(self.dataset)))

                for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                    labels[i] = y.item()
                print(labels)
                print(np.where(labels==0))
                labels_inliers = np.zeros((len(np.where(labels==1)[0])))
                labels_outliers = np.zeros((len(np.where(labels==0)[0])))

                zs_in = np.empty((64,len(labels_inliers)))
                zs_out = np.empty((64, len(labels_outliers)))
                print("number of inliers = " + str(len(labels_inliers)))
                print("number of outliers = " + str(len(labels_outliers)))
                count_in = 0
                count_out = 0

                for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                    x = x.to('cuda')
                    
                    x_r, z, z_dist = self.model(x) # z_dist has shape torch.Size([1, 100, 64])

                    
                    print(i)
                    # print("y.item() = " + str(y.item()))
                    if(y.item() == 1):
                        zs_in[:,count_in] = z.cpu().numpy()
                        count_in += 1
                    else:
                        zs_out[:,count_out] = z.cpu().numpy()
                        count_out += 1
                    #     print("INLIER")     
                    #     # # print(z_dist_sm.size())
                    #     z_d = z.detach()
                    #     z_d = z_d.view(len(z_d), -1).contiguous()
                    #     idxs_of_bins = torch.clamp(torch.unsqueeze(z_d, dim=1) * 100, min=0,
                    #         max=(100 - 1)).long()
                        
                    #     visualize_instance_on_tensorboard(writer,x,idxs_of_bins,z_d,z_dist,i,inlier=True)
                        
                    # elif(y.item()==0):
                    #     print("OUTLIER")      
                    #     z_d = z.detach()
                    #     z_d = z_d.view(len(z_d), -1).contiguous()
                    #     idxs_of_bins = torch.clamp(torch.unsqueeze(z_d, dim=1) * 100, min=0,
                    #         max=(100 - 1)).long()
                        
                    #     visualize_instance_on_tensorboard(writer,x,idxs_of_bins,z_d,z_dist,i,inlier=False)
                    # self.loss(x, x_r, z, z_dist)

                    sample_llk[i] = - self.loss.autoregression_loss
                    sample_rec[i] = - self.loss.reconstruction_loss
                    # if(y.item()==1):
                    #     writer.add_scalar('data/reconstruction_eror_in', sample_rec[i],i)
                    #     writer.add_scalar('data/llk_error_in', sample_llk[i],i)
                    # else:
                    #     writer.add_scalar('data/reconstruction_eror_out', sample_rec[i],i)
                    #     writer.add_scalar('data/llk_error_out', sample_llk[i],i)
                    # writer.add_custom_scalars_multilinechart(['data/reconstruction_eror_in', 'data/reconstruction_eror_out'],title='reconstruction error')
                    # writer.add_custom_scalars_multilinechart(['data/llk_error_in', 'data/llk_error_out'], title='llk error')
                    sample_y[i] = y.item()
                np.save('/data/Ponc/zs_2_in.npy',zs_in)
                np.save('/data/Ponc/zs_2_out.npy',zs_out)

                print("WRITING")
                hist, x_axis, _ = plt.hist(zs[0,:], bins = 100)
                x_axis = x_axis[:-1]
                hist = hist/np.sum(hist)
                ord_g = 4
                M = generateMoments(hist, ord_g,1)
                magic_q = comb(1+ord_g, 1)

                print(magic_q)
                q_eval = Q(M, x_axis)
                
                plt.subplot(211)
                plt.title("Gaussian Distr. mu=0.5, ss=0.1")
                plt.plot(x_axis, hist)
                plt.subplot(212)
                plt.title("Q(x) with M"+str(ord_g))
                plt.plot(x_axis, q_eval)
                plt.plot(x_axis, magic_q*np.ones(len(x_axis)))
                plt.show()


                writer.close()
                # Normalize scores
                sample_llk = normalize(sample_llk, min_llk, max_llk)
                sample_rec = normalize(sample_rec, min_rec, max_rec)

                # Compute the normalized novelty score
                sample_ns = novelty_score(sample_llk, sample_rec)

                # Compute AUROC for this class
                this_class_metrics = [
                    roc_auc_score(sample_y, sample_llk),  # likelihood metric
                    roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                    roc_auc_score(sample_y, sample_ns)    # novelty score
                ]
                oc_table.add_row([cl_idx] + this_class_metrics)

                all_metrics.append(this_class_metrics)

        # Compute average AUROC and print table
        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)
        oc_table.add_row(['avg'] + list(avg_metrics))
        print(oc_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(oc_table))

    def compute_normalizing_coefficients(self, cl):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        self.dataset.val(cl)
        loader = DataLoader(self.dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))
        for i, (x, y) in enumerate(loader):
            x = x.to('cuda')

            x_r, z, z_dist = self.model(x)

            self.loss(x, x_r, z, z_dist)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table
