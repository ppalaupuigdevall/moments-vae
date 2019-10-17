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

import matplotlib.pyplot as plt
from matplotlib import colors
plt.switch_backend('agg')
from SOS.moments import generateMoments
from visualizator import visualize_img_cpd

writer = SummaryWriter('runs/exp1')
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
            if(cl_idx == 1):
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

                for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                    x = x.to('cuda')
                    
                    x_r, z, z_dist = self.model(x) # z_dist has shape torch.Size([1, 100, 64])
                    
                    if(y.item() == 1):
                        print("INLIER")     
                        # print(z_dist_sm.size())
                        z_d = z.detach()
                        z_d = z_d.view(len(z_d), -1).contiguous()
                        idxs_of_bins = torch.clamp(torch.unsqueeze(z_d, dim=1) * 100, min=0,
                            max=(100 - 1)).long().cpu().numpy().reshape(1,64)
                        
                        writer.add_image('in/'+str(i)+'/'+'_img', x.cpu().numpy().reshape(1,28,28))
                        idx_of_bin_repr = np.zeros((100,64), dtype=np.uint8)
                        idxs_of_bins_np = idxs_of_bins
                        z_d_np = z_d.cpu().numpy().reshape(1,64)
                        print("z_n_dp = " + str(z_d_np))
                        for k in range(0,64):
                            idx_of_bin_repr[idxs_of_bins_np[0,k], k] = 255
                            # distributions of z_k
                            fig = plt.figure()
                            h1 = plt.plot(np.linspace(0.0,1.0,100), F.softmax(z_dist[0,:,k], dim=0).cpu().numpy())
                            point_2_draw = np.zeros((1,100))
                            print(z_d_np[0,k])
                            point_2_draw[0,int(100*z_d_np[0,k])] = 0.05
                            plt.stem(np.linspace(0.0,1.0,100),point_2_draw.reshape(100,))
                            ax = plt.gca()
                            writer.add_figure('in/'+str(i)+'/'+'hist/'+str(k),fig)

                        fig = plt.figure()
                        cmap = colors.ListedColormap(['red','blue'])
                        bounds = [0,255]
                        norm = colors.BoundaryNorm(bounds, cmap.N)
                        h1 = plt.imshow(idx_of_bin_repr)
                        # h1 = plt.plot(np.linspace(0.0, 1.0, 100), exp_z0_dist.cpu().numpy())
                        ax = plt.gca()
                        writer.add_figure('in/'+str(i)+'/'+'_z_dist',fig)

                        
                    elif(y.item()==0):
                        print("OUTLIER")      
                        z_d = z.detach()
                        z_d = z_d.view(len(z_d), -1).contiguous()
                        idxs_of_bins = torch.clamp(torch.unsqueeze(z_d, dim=1) * 100, min=0,
                            max=(100 - 1)).long().cpu().numpy().reshape(1,64)
                        
                        writer.add_image('out/'+str(i)+'/' +'_img', x.cpu().numpy().reshape(1,28,28))
                        idx_of_bin_repr = np.zeros((100,64), dtype=np.uint8)
                        idxs_of_bins_np = idxs_of_bins
                        z_d_np = z_d.cpu().numpy().reshape(1,64)
                        print("z_n_dp = " + str(z_d_np))
                        for k in range(0,64):
                            print(int(100*z_d_np[0,k]))
                            idx_of_bin_repr[idxs_of_bins_np[0,k], k] = 255
                            # distributions of z_k
                            fig = plt.figure()
                            h1 = plt.plot(np.linspace(0.0,1.0,100), F.softmax(z_dist[0,:,k], dim=0).cpu().numpy())
                            point_2_draw = np.zeros((1,100))
                            point_2_draw[0,int(100*z_d_np[0,k])] = 0.05
                            plt.stem(np.linspace(0.0,1.0,100),point_2_draw.reshape(100,))
                            ax = plt.gca()
                            writer.add_figure('out/'+str(i)+'/' +'hist/'+str(k),fig)

                        fig = plt.figure()
                        cmap = colors.ListedColormap(['red','blue'])
                        bounds = [0,255]
                        norm = colors.BoundaryNorm(bounds, cmap.N)
                        h1 = plt.imshow(idx_of_bin_repr)
                        # h1 = plt.plot(np.linspace(0.0, 1.0, 100), exp_z0_dist.cpu().numpy())
                        
                        ax = plt.gca()
                        writer.add_figure('out/'+str(i)+'/'+'_z_dist',fig)

                    self.loss(x, x_r, z, z_dist)

                    sample_llk[i] = - self.loss.autoregression_loss
                    sample_rec[i] = - self.loss.reconstruction_loss
                    sample_y[i] = y.item()
                print("WRITING")
                
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
