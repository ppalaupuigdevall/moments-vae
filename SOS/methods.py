import numpy as np
import torch
from torch.autograd import Variable, Function
from SOS import veronese
import cv2
from SOS import make_q_kernelized as KIC


def SOS_operation(feature, ts_layer, args, gts=[], US_list=[], tsdim=-1, if_return_feature=False, if_ts=True, flen=-1):
    # Padding
    # feature = padding_one(item, args.GPU)
    if if_ts:
        s = feature.shape
        if tsdim == -1:
            TS_dim = args.SOS.tsDIM
        else:
            TS_dim = tsdim
        if args.SOS.if_batch_ts:
            feature = batch_tslayer_forward(feature, ts_layer, s, args)
        else:
            feature_in = feature.view(s[1], s[2] * s[3])
            if args.SOS.if_mcb:
                feature_in = feature_in.t()
                feature_ts = ts_layer.forward(feature_in).t().view(s[0], TS_dim, s[2], s[3])
            else:
                feature_ts = ts_layer.forward(feature_in).view(s[0], TS_dim, s[2], s[3])
            # feature = normalization(torch.cat([feature[:, 1:, :, :], feature_ts], dim=1))
            feature = torch.cat([feature[:, 1:, :, :], feature_ts], dim=1)
            feature = padding_one(feature, args.GPU)
    # Doing SOS
    shape = feature.shape  # shape = [BatchSize,Num_channels,W,H]
    input_trans = feature.view([shape[0], shape[1], shape[2] * shape[3]])
    # input_vec = [Num_channels,W*H*BatchSize]
    input_vec = torch.cat(torch.chunk(input_trans, 2, 0), 2).squeeze()
    # msk = [1,W*H*BatchSize]
    mat = []
    US_temp_list = []
    if US_list.__len__() == 0:
        ind_class = 2
    else:
        ind_class = US_list.__len__()

    for i in range(0, ind_class):  # TODO: change 2 into num_classes
        if US_list.__len__() == 0:  # computing new moments matrix using gts
            # t1 = timeit.default_timer()
            # mask: index for selecting features
            assert gts.__len__() != 0
            msk = gts.view([1, shape[0] * shape[2] * shape[3]])
            US = SOS_matrix_gen(input_vec, msk, args, class_ind=i, flen=flen)
            US_temp_list.append(US)
        else:  # using pretrained moments matrix
            US = US_list[i]
        if args.SOS.if_ts:
            sos_score = torch.mm(input_trans[0, :, :].t(), US).pow(2).sum(1).unsqueeze(0)
            mat.append(sos_score.unsqueeze(1))
        else:
            temp_store = []
            batch = 30000
            ind = 0
            len_data = input_trans.shape[2]
            while ind <= len_data - 1:
                print('Caculating SOS vector...%{}'.format(ind * 1.0 / len_data * 100))
                t = input_trans[0, :, ind:min(ind + batch, len_data)]
                num = t.shape[1]
                tt, _ = veronese.veronese_nk(t, args.SOS.SOS_ORDER, if_cuda=(args.GPU >= 0))
                sos_score = torch.mm(tt.t(), US).pow(2).sum(1).unsqueeze(0)
                                 # for index in range(num)]
                ind = ind + num
                temp_store.append(sos_score.unsqueeze(1))
            mat.append(torch.cat(temp_store, dim=2))
    # fea_fin = torch.transpose(torch.cat(mat,dim=1),1,2)
    fea_fin = torch.cat(mat, dim=1)  # BatchSize x Num_classes x H*W
    del feature
    if if_return_feature:
        return fea_fin, US_temp_list, input_trans.squeeze(dim=0)
    else:
        return fea_fin, US_temp_list


def SOS_matrix_gen(input_vec, msk, args, class_ind, rho=0, flen=-1):
    """
    generate SOS inversed matrix
    :param input_vec: [ch,W*H*BatchSize]
    :param msk: [1,W*H*BatchSize]
    :param class_ind:
    :param args
    :return:
    """

    if_SVDCond = args.test_online.if_SVDCond
    SVDCond_alpha = args.test_online.SVDCond_alpha
    if_ts = args.SOS.if_ts
    if_batch_ts = args.SOS.if_batch_ts
    ver_order = args.SOS.SOS_ORDER
    # with torch.no_grad():
    #     print(torch.cuda.memory_allocated())
    #     input_vec[:, (msk[0, :] != class_ind)] = 0
    #     print(torch.cuda.memory_allocated())
    #     N = (msk[0, :] == class_ind).sum().float()
    # if if_ts:
    #     input_vec = input_vec / torch.sqrt(N)
    #     input_vec = torch.mm(input_vec, input_vec.transpose(0, 1))
    # else:
    #     temp_store = None
    #     batch = 40000
    #     ind = 0
    #     num_all = 0
    #     while ind <= input_vec.shape[1] - 1:
    #         print('Caculating SOS matrix...%{}'.format(num_all*1.0/input_vec.shape[1]*100))
    #         t = input_vec[:, ind:min(ind+batch, input_vec.shape[1])]
    #         num = t.shape[1]
    #         tt, _ = veronese.veronese_nk(t, ver_order, if_cuda=(args.GPU >= 0))
    #         for k in range(0, num):
    #             temp_result = torch.mm(tt[:, k].unsqueeze(dim=1), tt[:, k].unsqueeze(dim=1).t())
    #             if temp_store is None:
    #                 temp_store = temp_result
    #             else:
    #                 temp_store = temp_store * num_all/(num_all + 1) + temp_result / (num_all + 1)
    #             num_all = num_all + 1
    #         ind = ind + num
    #     input_vec = temp_store
    if if_ts is True and if_batch_ts is False:
        input_vec = input_vec[:, (msk[0, :] == class_ind)]
        N = (msk[0, :] == class_ind).sum().float()
        input_vec = input_vec / torch.sqrt(N)
        print('mean moment vec:{}'.format(input_vec.abs().mean()))
        input_vec = torch.mm(input_vec, input_vec.transpose(0, 1))
        print('mean SOS matirx:{}'.format(input_vec.mean()))
    else:
        batch = 30000
        ind = 0
        temp_store = None
        if if_ts is not True:
            num_all = 0
        else:
            num_all = torch.tensor(0, dtype=torch.float)
            # N = (msk[0, :] == class_ind).sum()
            # N = torch.tensor(N, dtype=torch.float)
            if args.GPU >= 0:
                num_all = num_all.cuda()
                # N = N.cuda()
        while ind <= input_vec.shape[1] - 1:
            print('Caculating SOS matrix...%{}'.format(ind*1.0/input_vec.shape[1]*100))
            m = msk[:, ind:min(ind+batch, msk.shape[1])]
            num = (m[0, :] == class_ind).sum()
            if num == 0:
                ind = ind + batch
                continue
            else:
                print('num:{}'.format(num))
            t = input_vec[:, ind:min(ind+batch, input_vec.shape[1])]
            t = t[:, (m[0, :] == class_ind)]
            assert t.shape[1] == num
            if if_ts is not True:
                t, _ = veronese.veronese_nk(t, ver_order, if_cuda=(args.GPU >= 0))
                for k in range(0, num):
                    temp_result = torch.mm(t[:, k].unsqueeze(dim=1), t[:, k].unsqueeze(dim=1).t())
                    if temp_store is None:
                        temp_store = temp_result
                    else:
                        temp_store = temp_store * num_all/(num_all + 1) + temp_result / (num_all + 1)
                    num_all = num_all + 1
            else:
                temp_result = torch.mm(t, t.transpose(0, 1))
                print('temp_result mean:{} max:{} min:{}'.format(temp_result.mean(), temp_result.max(), temp_result.min()))
                # print(temp_result[0, 0]/num.float())
                if temp_store is None:
                    temp_store = temp_result/(num.float())
                else:
                    temp_store = temp_store * num_all / (num_all + num.float()) + temp_result / (num_all + num.float())
                num_all = num_all + num.float()
            ind = ind + batch
        input_vec = temp_store
    if if_SVDCond:
        [u, s, _] = input_vec.svd()
        s_max = s.max()
        s = s + s_max * SVDCond_alpha
        print('Smallest eigv s:{}'.format(s.min()))
        print('Largest eigv s:{}'.format(s.max()))
        US = torch.mm(u, torch.diag(s.pow(-0.5)))
    else:
        matInv = nnsirhcH()
        # US = matInv(input_vec+(torch.eye(input_vec.shape[0])*rho).cuda())
        US = matInv(input_vec, torch.tensor(flen))
    return US


def adjust_lr(optimizer, total_step, init_lr, step=20, mag=0.1):
    lr = init_lr * (mag ** (total_step // step))
    for param_group in optimizer.param_groups:
        if param_group['lr'] != 0:
            param_group['lr'] = lr * (param_group['lr_scale'] if 'lr_scale' in param_group else 1)


def padding_one(feature, cuda_device):
    '''
    Padding feature vecture with 1
    :param feature: NxCxHxW, feature to be padded
    :return: padded feature Nx(C+1)xHxW
    '''
    size = np.asarray([feature.shape[0], 1])
    size = np.append(size, np.asarray(feature.shape)[2:4])
    ones = np.ones(size)
    if cuda_device >= 0:
        ones = Variable(torch.FloatTensor(ones).cuda(), requires_grad=False)
    else:
        ones = Variable(torch.FloatTensor(ones), requires_grad=False)
    #  N = (feature.shape[0]*feature.shape[1]*feature.shape[2]*feature.shape[3])
    #  feature_padding = torch.cat([feature, ones], dim=1)/np.sqrt(np.asarray(N))
    feature_padding = torch.cat([ones,feature], dim=1)
    return feature_padding


def padding_cord(feature, cuda_device, whitening=False):
    '''
    Padding feature vecture with cord (x and y)
    :param feature: NxCxHxW, feature to be padded
    :return: padded feature Nx(C+2)xHxW
    '''
    size = np.asarray([feature.shape[0], 1])
    size = np.append(size, np.asarray(feature.shape)[2:4])
    w = size[2]
    h = size[3]
    x = np.ones(size)
    for i in range(0, w):
        x[:, :, i, :] = np.ones([size[0], size[1], 1, size[3]])*i
    y = np.ones(size)
    for i in range(0, h):
        y[:, :, :, i] = np.ones([size[0], size[1], size[2], 1]).squeeze(3)*i
    if cuda_device >= 0:
        x = Variable(torch.FloatTensor(x).cuda(), requires_grad=False)
        y = Variable(torch.FloatTensor(y).cuda(), requires_grad=False)
    else:
        x = Variable(torch.FloatTensor(x), requires_grad=False)
        y = Variable(torch.FloatTensor(y), requires_grad=False)
    #  N = (feature.shape[0]*feature.shape[1]*feature.shape[2]*feature.shape[3])
    #  feature_padding = torch.cat([feature, ones], dim=1)/np.sqrt(np.asarray(N))
    cord = torch.cat([x, y], dim=1)
    if whitening:
        cord = input_whitening_per_image(cord)
    feature_padding = torch.cat([cord, feature], dim=1)
    return feature_padding


def trans_for_softmax(vec1, vec2):
    '''
    Transform output of SOS polynomial into a scale so that softmax could works
    Note that for SOS polynomial smaller score means higher prob
    output = exp(-1 * (vec1 ./ vec2).^2)
    :param vec1: vec to get rescaled
    :param vec2: vec to measure the vec1
    :return: rescaled vec1
    '''
    return torch.exp(-1*(torch.mul(vec1, vec2.pow(-1)).pow(0.5)))


def normalization(input):  # TODO: find the non-inplace operation
    """
    do normalize to one sample per channel before sos
    :param input:
    :return:
    """
    mean = input.view(input.shape[0], input.shape[1], input.shape[2]*input.shape[3]).mean(dim=-1)
    std = input.view(input.shape[0], input.shape[1], input.shape[2]*input.shape[3]).std(dim=-1)

    return input.sub(mean.unsqueeze(-1).unsqueeze(-1)).div(std.unsqueeze(-1).unsqueeze(-1) + 1e-9)
    # return input.sub(mean.unsqueeze(-1).unsqueeze(-1))

def scale_ch(input, ch_dim=1):
    '''
    do scaling for each channel of input
    :param input:
    :return:
    '''
    max_, _ = input.max(dim=ch_dim)
    min_, _ = input.min(dim=ch_dim)
    return (input - min_) / (max_ - min_)

def input_whitening_per_image(input):
    """
    do whitening to one sample per image before sos (ZCA?)
    :param input:
    :return:
    """
    input_mat = input.view(input.shape[1], input.shape[2]*input.shape[3])
    mean = input_mat.mean(dim=1).unsqueeze(dim=1).expand_as(input_mat)
    corr = torch.mm(input_mat, input_mat.t())
    [u, s, _] = corr.svd()
    corr = torch.mm(torch.mm(u, torch.diag(s.pow(-0.5))), u.t())
    # input_mat = torch.mm(corr, input_mat - mean)
    input_mat = torch.mm(corr, input_mat)
    # print('max:{}, min:{}'.format(inputF_mat.max(), input_mat.min()))
    return input_mat.view(input.shape)


class nnsirhcH(Function):
    '''
    Inverse Christoffel function
    C = U_k * S_k^(-0.5)
    '''
    #TODO: Extend with classes more than two
    def forward(self,input, flen=-1):
        '''
        SOS forward
        :param input: Matrix of DxD
        :return: U*S^-1/2, U,S,_=M.svd()
            U:DxD S:DxD
        '''
        u, s, _ = input.svd()
        # TODO: input will be very sparse after ts with high dimension, resulted in highly rank defficient
        print('Rank: {}'.format(torch.matrix_rank(input)))
        # rank_input = torch.matrix_rank(input)
        rank_input = len(s)
        if flen > 0:
            rank_input = flen
        u_ = u[:, 0:rank_input]
        s_ = s[0:rank_input]
        p = (s_ / s_.sum()).cumsum(0)
        print('Cumsum s:{}'.format(len(((p <= 1.0) * (p >= 0.9999)).nonzero())*1.0 / len(p)))
        print('Smallest eigv s:{}'.format(s_.min()))
        print('Largest eigv s:{}'.format(s_.max()))
        self.intermediate = [u_, s_]
        return torch.mm(u_, torch.diag(s_.pow(-0.5)))

    def backward(self, grad_output):
        result = self.intermediate
        u = result[0]
        s = result[1]
        dLdU = torch.mm(grad_output, torch.diag(s.pow(-1/2)))
        dLdS = -0.5*torch.diag(s.pow(-3/2))*u.t()*grad_output

        dLdM = u * u.t() * dLdU + torch.diag(torch.diag(dLdS)) * u.t()
        # dLdM[dLdM > 10] = 10
        # dLdM[dLdM < -10] = -10
        print('Smallest grad of SOS:{}'.format(dLdM.min()))
        print('Lagrest grad of SOS:{}'.format(dLdM.max()))
        return dLdM


def save_tmp_img(input, name):
    import scipy.misc as sm
    img = input.view(input.shape[2], input.shape[3]).cpu().detach().numpy()
    _max = img.max()
    _min = img.min()
    sm.imsave(name, (img - _min)/(_max - _min))


def batch_tslayer_forward(feature, net_ts_layer, s, exp_args):  # TODO: add 1st order feature
    """
    tensor sketch forward in batch
    :param feature: [1, ch, w=480, h=854]
    :param net_ts_layer:
    :param exp_args:
    :param s: shape of ori feature
    :return: feature: [1, ts_dim, w=480, h=854]
    """
    f_list = []
    ind_ts = 0
    ts_bs = exp_args.SOS.ts_bs
    while ind_ts <= s[2] - 1:
        print('Go batch ts..%{}'.format(ind_ts * 1.0 / s[2] * 100))
        temp_feature = feature[:, :, ind_ts:min(ind_ts + ts_bs, s[2]), :]

        temp_feature = net_ts_layer.forward(
            temp_feature.view(s[1], temp_feature.shape[2] * s[3])
        ).view(s[0], exp_args.SOS.tsDIM, temp_feature.shape[2], s[3])
        # temp_feature.shape: [1, ts_ch, w=ts_bs, h=854]
        temp_feature = padding_one(temp_feature, exp_args.GPU)
        f_list.append(temp_feature)
        ind_ts = ind_ts + ts_bs
    torch.cuda.empty_cache()
    feature_sos = torch.cat(f_list, dim=2)
    return feature_sos


def feature_preprocessing(inputs, feature_sos, args, save_name):
    if args.test_online.if_rgb:
        if args.test_online.if_rgb_wh:

            feature_sos = torch.cat([input_whitening_per_image(inputs), feature_sos], dim=1)
            save_name = save_name + '_rgbpaddingwh'
        else:
            feature_sos = torch.cat([inputs, feature_sos], dim=1)
            save_name = save_name + '_rgbpadding'
    if args.test_online.if_whitening:
        # feature_sos = methods.input_whitening_per_channel(feature_sos)
        feature_sos = input_whitening_per_image(feature_sos)
        save_name = save_name + '_whiteningmodi'
    # del feature_sos
    # Compute inversed SOS polynomial matrix at the end of training
    if args.test_online.if_cord:
        feature_sos = padding_cord(feature_sos, args.GPU, args.test_online.if_cord_wh)
        if args.test_online.if_cord_wh:
            save_name = save_name + '_cordwh'
        else:
            save_name = save_name + '_cord'
    if args.test_online.if_norm:
        feature_sos = normalization(feature_sos)
        save_name = save_name + '_norm'
    return feature_sos, save_name


def subclustering(feature, num, args_in):
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(num, affinity='precomputed', assign_labels='discretize')
    s = feature.shape
    if args_in.GPU >= 0:
        feature_np = feature.view(s[1], s[2]*s[3]).detach().cpu().numpy()
    else:
        feature_np = feature.view(s[1], s[2] * s[3]).numpy()
    corr = np.corrcoef(feature_np)
    corr = (corr - corr.min())/(corr.max() - corr.min())
    return sc.fit_predict(corr)


def outlier_removal(feature_sos, pred, pred_lastfarme, prob, sos_score, gt_1stframe, ts_layer, args):
    """
    Removal outlier
    :param feature_sos: bs*ch*w*h
    :param pred: bs*1*w*h 0 or 1
    :param pred_lastfarme: bs*1*w*h
    :param prob: bs*2*w*h, 0 for bg, 1 for fg, higher the better, [0, 1]
    :param sos_score: bs*2*w*h, 0 for bg, 1 for fg, lower the better, >0
    :param gt_1stframe: bs*1*w*h
    :param ts_layer: tensorsketch layer for generating features
    :param args: args of exp
    :return:
    """
    if args.GPU >= 0:
        pred_np = pred.view(pred.shape[2], pred.shape[3]).cpu().detach().numpy()
        prv_pred_np = pred_lastfarme.view(pred_lastfarme.shape[2], pred_lastfarme.shape[3]).cpu().detach().numpy()
        prob_np = prob.view(2, prob.shape[2], prob.shape[3]).cpu().detach().numpy()
        feature_np = feature_sos.view(feature_sos.shape[1], feature_sos.shape[2], feature_sos.shape[3]).cpu().detach().numpy()
        sos_score_np = sos_score.view(sos_score.shape[1], sos_score.shape[2], sos_score.shape[3]).cpu().detach().numpy()
    else:
        pred_np = pred.view(pred.shape[2], pred.shape[3]).numpy()
        prv_pred_np = pred_lastfarme.view(pred_lastfarme.shape[2], pred_lastfarme.shape[3]).numpy()
        prob_np = prob.view(2, prob.shape[2], prob.shape[3]).numpy()
        feature_np = feature_sos.view(feature_sos.shape[1], feature_sos.shape[2], feature_sos.shape[3]).numpy()
        sos_score_np = sos_score.view(sos_score.shape[1], sos_score.shape[2], sos_score.shape[3]).numpy()
    #  step one: remove false pred out of area which is a dilation of previous pred
    dis = args.outlier_removal.dis_toofar
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dis, dis))
    msk_dis = cv2.dilate(prv_pred_np.astype(np.float), element)  # (w, h)
    tmp = np.multiply(msk_dis, pred_np)

    # #  step two: SOS-RSC
    if args.outlier_removal.if_2ndremoval:
        prob_dis_removed = np.multiply(msk_dis, prob_np)
        pred_inside = np.multiply(prob_dis_removed, pred_np)
        idx_inside = (pred_inside[1, :, :] > 0).nonzero()
        sample = prob_dis_removed[1, pred_inside[1, :, :] > 0]
        if args.outlier_removal.prob_th > 0:
            msk_new = (prob_dis_removed[1, :, :] > args.outlier_removal.prob_th).astype(np.uint8)  # find reliable inlier
        else:
            th, _ = cv2.threshold((sample * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            msk_new = (prob_dis_removed[1, :, :] > (th/255)).astype(np.uint8)  # find reliable inlier
        pred.data = torch.tensor(msk_new, dtype=torch.float)
    else:
        pred.data = torch.tensor(tmp, dtype=torch.float)
    #
    # import scipy.misc as sm
    # sm.imsave('part.png', msk_new*255)
    #
    # # msk_fe = np.tile(np.expand_dims(msk_new, axis=0), [feature_np.shape[0], 1, 1])
    # # feature_in = np.multiply(feature_np, msk_fe)
    # msk_new_tensor = torch.tensor(msk_new, dtype=torch.int).view(1, feature_sos.shape[2] * feature_sos.shape[3])
    # if args.GPU >= 0:
    #     msk_new_tensor = msk_new_tensor.cuda()
    # US = SOS_matrix_gen(feature_sos.view(feature_sos.shape[1], feature_sos.shape[2] * feature_sos.shape[3]),
    #                     msk_new_tensor, args, class_ind=1)
    # if args.GPU >= 0:
    #     US = US.cpu().detach().numpy()
    # else:
    #     US = US.numpy()
    # worst_idx = np.where(prob_np[0, :, :] == prob_np[0, :, :].max())
    # feature_worst = feature_np[:, worst_idx[0], worst_idx[1]]
    # p_star = np.matmul(US.transpose(), feature_worst)
    # scale = np.linalg.norm(p_star, 2) ** 2
    # p_star = np.matmul(US, p_star)/scale
    # if args.GPU >= 0:
    #     feature_np = feature_sos.view(feature_sos.shape[1], feature_sos.shape[2] * feature_sos.shape[3])\
    #         .cpu().detach().numpy()
    # else:
    #     feature_np = feature_sos.view(feature_sos.shape[1], feature_sos.shape[2] * feature_sos.shape[3]).numpy()
    # h = torch.tensor(np.matmul(feature_np.transpose(), p_star).transpose())\
    #     .view(feature_sos.shape[2], feature_sos.shape[3])

    if args.GPU >= 0:
       pred = pred.cuda()
    return pred.view(pred.shape[0]*pred.shape[1])


def save_plot_surface(data, name):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    ma_ = np.max(data)
    mi_ = np.min(data)
    data = (data - mi_)/(ma_ - mi_)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = data.shape[1]
    y = data.shape[0]
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    sf = ax.plot_surface(X=X, Y=Y, Z=data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.colorbar(sf, shrink=0.5, aspect=5)
    plt.savefig(name)
    plt.close()


def shading_subspace_result(img, index, num_cluster, save_name, save_name2):
    import cv2
    import scipy.misc as sm
    color_max = np.asarray([1, 0, 1])
    img = (img - img.min()) / (img.max() - img.min())
    step = (1.0 / num_cluster) * (color_max - [0, 0, 0])
    mask = np.zeros([img.shape[0], img.shape[1], 3]).astype(float)
    for i in range(1, num_cluster+1):
        for dim in range(0, 3):
            mask[index == i, dim] += (step[dim] * i).astype(float)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    img_shading = cv2.addWeighted(img.astype(float), 0.8, mask, 0.2, 0)
    sm.imsave(save_name, img_shading)
    sm.imsave(save_name2, mask)


def gen_mask_bg_closing(fg, args):
    '''
    generate a mask of background which is close to foreground
    :param fg: w*h, 0 for bg, 1 for fg
    :param args:
    :return:
    '''
    dis = args.clustering.dis_bg_close_fg
    if args.GPU >= 0:
        fg_np = fg.detach().cpu().numpy()
    else:
        fg_np = fg.numpy()
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dis, dis))
    msk_dis = cv2.dilate(fg_np.astype(np.float), element)
    fg_np_re = 1 - fg_np
    bg_close = np.multiply(msk_dis, fg_np_re)
    bg_close_torch = torch.tensor(bg_close, dtype=torch.float)
    if args.GPU >= 0:
        bg_close_torch = bg_close_torch.cuda()
    return bg_close_torch.unsqueeze(dim=0)


def foreground_mask_update(mask, dis):
    '''
    Generate a mask
    :param mask: numpy, w * h
    :param dis: distance
    :return: a mask dilated of param:mask
    '''
    ero_dis = 5
    element_dila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dis, dis))
    element_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_dis, ero_dis))
    msk_dis = cv2.dilate(cv2.erode(mask.astype(np.float), element_ero), element_dila)
    return msk_dis


def subspace_process_sos(feature_vec, clustering_results, num_cluster, tslayer, args):
    '''
    Compute SOS polynomial of dataset for each cluster provided
    :param feature_vec: [batchsize x channel x num_sample] input feature
    :param clustering_results: [num_sample] clustering group label for each data point
    :param num_cluster: int number of clusters
    :param tslayer: tensorsketch layer
    :param args: arguments class read from config file
    :returns:SOS_matrix_cluster_list
    '''
    Q_mean_list_cluster = []
    SOS_matrix_cluster_list = []
    print('Go SOS polynomial caculation')
    for ind_cluster in range(0, num_cluster):
        feature_sos_sub = feature_vec[:, :, clustering_results == ind_cluster]
        feature_sos_sub = padding_one(feature_sos_sub, args.GPU)
        s = feature_sos_sub.shape
        feature_sos_ts_sub = tslayer.forward(feature_sos_sub.squeeze()) \
            .view(s[0], args.SOS.tsDIM, s[2])  # feature_sos.shape: [1, ts_ch, numel]
        feature_sos_sub = normalization(
            torch.cat([feature_sos_sub[:, 1:, :], feature_sos_ts_sub], dim=1).unsqueeze(dim=3))
        feature_sos_sub = padding_one(feature_sos_sub, args.GPU)
        s = feature_sos_sub.shape
        input_trans = feature_sos_sub.view([s[0], s[1], s[2] * s[3]]).squeeze(dim=0)
        msk = torch.zeros([1, s[0] * s[2] * s[3]])
        print('Cluster i: {} points num: {}'.format(ind_cluster, s[2]))
        if args.GPU >= 0:
            msk = msk.cuda()
        US_sub = SOS_matrix_gen(input_trans, msk, args, class_ind=0)
        SOS_matrix_cluster_list.append(US_sub)
        if args.SOS.if_avg:
            Q_t, _ = SOS_operation(input_trans.unsqueeze(dim=3),
                                   ts_layer=tslayer, args=args, US_list=[US_sub], if_ts=False)
            Q_mean_list_cluster.append(Q_t.mean())
    if args.SOS.if_avg:
        print('Cluster Q mean:')
        print(Q_mean_list_cluster)
    return SOS_matrix_cluster_list, Q_mean_list_cluster


def subspace_score_sos(feature_matrix, SOS_matrix_list, num_cluster, tslayer, args):
    '''
    Compute SOS score for given feature with polynomials list
    :param feature_matrix: [batchsize=1 x channel x num_sample] input feature
    :param SOS_matrix_list: a list of SOS matrix
    :param num_cluster: int number of clusters
    :param tslayer: tensorsketch layer
    :param args: arguments class read from config file
    :return: sub_cluster_score, class_score
    '''
    sub_cluster_score = []
    for ind_cluster in range(0, num_cluster):
        feature_sos_score, _ = SOS_operation(feature_matrix, ts_layer=tslayer,
                                             args=args, US_list=[SOS_matrix_list[ind_cluster]],
                                             if_ts=args.SOS.if_ts)
        sub_cluster_score.append(feature_sos_score)
    sub_cluster_score_tensor = torch.cat(sub_cluster_score, dim=0)
    class_score = sub_cluster_score_tensor.min(dim=0)
    return sub_cluster_score, class_score


def subspace_process_KIC(feature_vec, clustering_results, num_cluster, args):
    '''
    Compute SOS polynomial of dataset for each cluster provided using KIC
    :param feature_vec: [batchsize x channel x num_sample] input feature
    :param clustering_results: [num_sample] clustering group label for each data point
    :param num_cluster: int number of clusters
    :param args: arguments class read from config file
    :returns:Q_fun_list
    '''
    Q_fun_list = []
    for ind_cluster in range(0, num_cluster):
        feature_sos_sub = feature_vec[:, :, clustering_results == ind_cluster]
        Q = KIC.make_Q_kernelized_torch(feature_sos_sub.squeeze().t(), args)
        Q_fun_list.append(Q)
    return Q_fun_list


def subspace_score_KIC(feature_matrix, Q_list, num_cluster):
    '''
    Compute SOS score for given feature with Q function list by KIC
    :param feature_matrix: [channel x num_sample] input feature
    :param Q_list: a list of Q function
    :param num_cluster: int number of clusters
    :return: sub_cluster_score, class_score
    '''
    total_numel = feature_matrix.shape[1]
    step_count = 0
    group_num = 10
    step_size = np.floor(total_numel / group_num).astype(np.int)
    list_score = []
    sub_cluster_score = []
    for ind_cluster in range(0, num_cluster):
        while step_count < total_numel:
            feature_sos_score_tmp = Q_list[ind_cluster]\
                (feature_matrix[:, step_count: min(step_count+step_size, total_numel)].t())
            list_score.append(feature_sos_score_tmp)
            step_count = step_count + step_size
        feature_sos_score = torch.cat(list_score, dim=0)
        sub_cluster_score.append(feature_sos_score)
        torch.cuda.empty_cache()
    sub_cluster_score_tensor = torch.cat(sub_cluster_score, dim=1)
    class_score = sub_cluster_score_tensor.min(dim=1)
    return sub_cluster_score, class_score


def vector_detangle(feature):
    '''
    Go detangle of feature vector introduced in paper:
    Decorrelation of Neutral Vector Variables: Theory and Applications
    :param feature: n*dim, feature vector to be detangeled
    :return: n*dim, detangeled feature
    '''
    sum_feature = feature.sum(1, True)
    feature_s1norm = feature/sum_feature
    cum_sum_t = feature_s1norm.cumsum(dim=1)
    cum_sum = torch.ones_like(cum_sum_t)
    cum_sum[:, 1:] = (1 - cum_sum_t)[:, 0:-1]
    return feature_s1norm/(cum_sum + 1e-6)


def exponet_grid(feature, exp_order):
    '''
    Create a grid of feature up to exponet order exp_order
    :param feature: n feature vector to be raised
    :param exp_order: exponet order for grid
    :return: feature n*exp_order
    '''
    feature_grid = torch.ones([feature.shape[0], 1+exp_order], device=feature.device)
    for i in range(0, exp_order):
        if i == 0:
            temp_feature = feature
        else:
            temp_feature = feature**(i + 1) / (i + 1)  # divided by power
        feature_grid[:, i+1] = temp_feature
    return feature_grid


if __name__ == "__main__":
    # import yaml
    # from easydict import EasyDict as edict
    #
    # cfg = './cfg/cfg_09062018.yml'
    #
    # with open(cfg, 'r') as f:
    #     args = edict(yaml.load(f))
    #
    # from SOS import tensorSketch as ts
    # ts_layer = ts.TensorSketchLayer(args.SOS.DIN, args.SOS.tsDIM, args.SOS.SOS_ORDER, cuda_device=args.GPU)
    # x = torch.randn([1, 64, 128, 256], requires_grad=True)
    # gt = torch.zeros([1, 1, 128, 256], requires_grad=True)
    # gt[0, 0, 64:100, 128:200] = 1
    # y, us = SOS_operation(x, ts_layer, args, gts=gt)
    # print(y.shape)
    # sf = torch.nn.Softmax(dim=1)
    # pred = sf(y)
    # loss = pred[0, 1, :].sum()
    # # loss.requires_grad = True
    # loss.backward()
    # x = torch.randn([3, 100])
    # x[x < 0] = 0
    # x = vector_detangle(x)
    # xx = exponet_grid(x[:, 0], 3)
    x = torch.randn([1,3,10,10])
    x.requires_grad = True
    y = scale_ch(x, ch_dim=1)
    z = y.sum()
    z.backward()
