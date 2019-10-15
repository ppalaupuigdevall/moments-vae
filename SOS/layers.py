import numpy as np
import torch
import torch.nn.modules as model
from SOS import methods
class SOS_layer(model):
    def __init__(self):
        super(SOS_layer, self).__init__()