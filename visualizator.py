import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from matplotlib import colors
plt.switch_backend('agg')
import torch
from SOS.moments import generateMoments
from SOS.moments import Q

def visualize_instance_on_tensorboard(writer,x,idxs_of_bins,z_d,z_dist,i,inlier=True):
    """
    Aquesta funcio esta feta perque el codi de l'altre script no fos demoniac, no jutjar-la

    """
    eps = np.finfo(float).eps
    if(inlier):

        # Compute how likely is that z
        z_dist_c = torch.clamp(z_dist, eps, 1 - eps) # log safe
        log_z_dist = torch.log(z_dist_c)
        selected = torch.gather(log_z_dist, dim=1, index=idxs_of_bins)
        selected = torch.squeeze(selected, dim=1)
        likelihood = torch.sum(selected, dim=-1)
        likelihood = likelihood.cpu().item()
        writer.add_scalar('data/likelihood_inliers', likelihood, i)
        writer.add_image('in/'+str(i)+'/'+'_img___p_z = '+'{:.1f}'.format(likelihood), x.cpu().numpy().reshape(1,28,28))
        

        idxs_of_bins = idxs_of_bins.cpu().numpy().reshape(1,64)
        idx_of_bin_repr = np.zeros((100,64), dtype=np.uint8)
        idxs_of_bins_np = idxs_of_bins
        z_d_np = z_d.cpu().numpy().reshape(1,64)
        
        for k in range(0,64):
            idx_of_bin_repr[idxs_of_bins_np[0,k], k] = 255
            # distributions of z_k
            fig = plt.figure()
            # This distribution is p(zi)
            distribution = F.softmax(z_dist[0,:,k], dim=0).cpu().numpy()
            M = generateMoments(distribution, 4, 1)
            cristofel = Q(M,np.linspace(0.0,1.0,100))

            
            h1 = plt.plot(np.linspace(0.0,1.0,100), distribution)
            
            point_2_draw = np.zeros((1,100))
            
            point_2_draw[0,int(100*z_d_np[0,k])] = 0.1
            plt.stem(np.linspace(0.0,1.0,100),point_2_draw.reshape(100,))
            ax = plt.gca()
            # writer.add_figure('in/'+str(i)+'/'+'hist/'+str(k),fig)
            writer.add_figure('in/'+'hist/'+str(k),fig)
            fig = plt.figure()
            plt.plot(np.linspace(0.0,1.0,100), cristofel)
            writer.add_figure('in/'+'hist/'+str(k)+'Q(z)',fig)


        fig = plt.figure()
        cmap = colors.ListedColormap(['red','blue'])
        bounds = [0,255]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        h1 = plt.imshow(idx_of_bin_repr)
        # h1 = plt.plot(np.linspace(0.0, 1.0, 100), exp_z0_dist.cpu().numpy())
        ax = plt.gca()
        writer.add_figure('in/'+str(i)+'/'+'_z_dist',fig)
    else:
        # Compute how likely is that z
        z_dist_c = torch.clamp(z_dist, eps, 1 - eps) # log safe
        log_z_dist = torch.log(z_dist_c)
        selected = torch.gather(log_z_dist, dim=1, index=idxs_of_bins)
        selected = torch.squeeze(selected, dim=1)
        likelihood = torch.sum(selected, dim=-1)
        likelihood = likelihood.cpu().item()
        idxs_of_bins = idxs_of_bins.cpu().numpy().reshape(1,64)
        writer.add_scalar('data/likelihood_outliers', likelihood, i)
        writer.add_image('out/'+str(i)+'/' +'_img___pz_ = '+'{:.1f}'.format(likelihood), x.cpu().numpy().reshape(1,28,28))
        idx_of_bin_repr = np.zeros((100,64), dtype=np.uint8)
        idxs_of_bins_np = idxs_of_bins
        z_d_np = z_d.cpu().numpy().reshape(1,64)
        
        for k in range(0,64):
            
            idx_of_bin_repr[idxs_of_bins_np[0,k], k] = 255
            
            fig = plt.figure()
            distribution = F.softmax(z_dist[0,:,k], dim=0).cpu().numpy()
            M = generateMoments(distribution, 4, 1)
            cristofel = Q(M,np.linspace(0.0,1.0,100))
            
            h1 = plt.plot(np.linspace(0.0,1.0,100), distribution)
            

            point_2_draw = np.zeros((1,100))
            point_2_draw[0,int(100*z_d_np[0,k])] = 0.1
            plt.stem(np.linspace(0.0,1.0,100),point_2_draw.reshape(100,))
            ax = plt.gca()
            # writer.add_figure('out/'+str(i)+'/' +'hist/'+str(k),fig)
            writer.add_figure('out/'+'hist/'+str(k),fig)
            fig = plt.figure()
            plt.plot(np.linspace(0.0,1.0,100), cristofel)
            writer.add_figure('out/'+'hist/'+str(k)+'Q(z)',fig)

        fig = plt.figure()
        cmap = colors.ListedColormap(['red','blue'])
        bounds = [0,255]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        h1 = plt.imshow(idx_of_bin_repr)
        ax = plt.gca()
        writer.add_figure('out/'+str(i)+'/'+'_z_dist',fig)
    writer.add_custom_scalars_multilinechart(['data/likelihood_inliers', 'data/likelihood_outliers'])