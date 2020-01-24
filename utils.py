import collections
import random
import re
from typing import List

import numpy as np
import torch
from torch.utils.data.dataloader import _use_shared_memory
from torch.utils.data.dataloader import int_classes
from torch.utils.data.dataloader import numpy_type_map
from torch.utils.data.dataloader import string_classes



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




def set_random_seed(seed):
    # type: (int) -> None
    """
    Sets random seeds.
    :param seed: the seed to be set for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(samples, min, max):
    # type: (np.ndarray, float, float) -> np.ndarray
    """
    Normalize scores as in Eq. 10

    :param samples: the scores to be normalized.
    :param min: the minimum of the desired scores.
    :param max: the maximum of the desired scores.
    :return: the normalized scores
    """
    return (samples - min) / (max - min)


def novelty_score(sample_llk_norm, sample_rec_norm):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Computes the normalized novelty score given likelihood scores, reconstruction scores
    and normalization coefficients (Eq. 9-10).
    :param sample_llk_norm: array of (normalized) log-likelihood scores.
    :param sample_rec_norm: array of (normalized) reconstruction scores.
    :return: array of novelty scores.
    """

    # Sum
    ns = sample_llk_norm + sample_rec_norm

    return ns


def concat_collate(batch):
    # type: (List[torch.Tensor]) -> torch.Tensor
    """
    Puts each data field into a tensor stacking along the first dimension.
    This is different to the default pytorch collate that stacks samples rather than
    concatenating them.

    :param batch: the input batch to be collated.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.cat([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: concat_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [concat_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
