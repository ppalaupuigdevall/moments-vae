import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
dg = '/data/Ponc/Q_0_0/Agrads/'
da = '/data/Ponc/Q_0_0/As/'
Agrads = os.listdir(dg)
As = os.listdir(da)

for i in range(len(Agrads)):
    a_np = np.load(os.path.join(da,As[i]))
    print(type(a_np))
    
    
    g_np = np.load(os.path.join(dg,Agrads[i]))
    cv2.imshow('a',np.log(a_np))
    cv2.waitKey(1)

    # plt.matshow(a_np)
    # plt.matshow(g_np)
    # plt.show()