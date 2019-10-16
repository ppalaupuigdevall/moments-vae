import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

def visualize_img_cpd(x, hist, y):
    title_ = 'INLIER'
    if(y==1):
        title_ = 'OUTLIER'
    x_r = x.reshape(28,28)
    plt.title(title_)
    plt.subplot(121)
    plt.imshow(x_r)
    plt.subplot(122)
    plt.plot(np.linspace(0.0, 1.0, 100), hist)
    np.save('/data/Ponc/hist.npy', hist)
    np.save('/data/Ponc/x.npy', x_r)
    plt.show()

writer = SummaryWriter('runs/exp1')
x = np.random.rand(1,28,28)
writer.add_image('my_image', x,0)
writer.close()