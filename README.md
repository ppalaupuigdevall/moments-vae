# moments-vae

This repo tries to do outlier detection with the Sum of Squares polynomial Q(x).

Some details about the code:

There are basically 2 approaches: 

1. Use *empirical Moment Matrix*, which is implemented in:
  * class Q_Real_M
  * class Q_Real_M_Batches (still in maintenance)
2. *Learn* directly ![equation](https://latex.codecogs.com/gif.latex?M%5E%7B-1%7D), which is implemented in:
  * class Q      
  * class Q_MyBilinear
  * class Q_PSD
  
  Q and Q_MyBilinear do the same, is just that one uses a pytorch built-in function, the other not (I did it because I thought there was a bug) 
  
  ## USAGE 
  
  `python train_forwen.py --model Q_real_M --writer experimentname --idx_inliers 1 --device 2`
  
  ## Requirements
  
  Most of them are not strictly necessary, but I paste the pip freeze of the environment I was using :)
  
absl-py==0.8.1
astor==0.8.0
certifi==2019.9.11
chardet==3.0.4
cycler==0.10.0
decorator==4.3.0
gast==0.2.2
google-pasta==0.1.7
grpcio==1.24.1
h5py==2.10.0
idna==2.8
imageio==2.6.1
imageio-ffmpeg==0.3.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.0.1
Markdown==3.1.1
matplotlib==3.0.2
moviepy==1.0.1
networkx==2.2
numpy==1.17.2
opencv-python==3.4.3.18
opt-einsum==3.1.0
Pillow==5.3.0
pkg-resources==0.0.0
prettytable==0.7.2
proglog==0.1.9
protobuf==3.10.0
pyparsing==2.3.0
python-dateutil==2.7.5
PyWavelets==1.0.1
requests==2.22.0
scikit-image==0.16.1
scikit-learn==0.19.1
scipy==1.1.0
six==1.11.0
tensorboard==2.0.0
tensorboardX==1.9
tensorflow==2.0.0
tensorflow-estimator==2.0.0
termcolor==1.1.0
torch==0.4.1
torchvision==0.2.1
tqdm==4.28.1
urllib3==1.25.6
Werkzeug==0.16.0
wrapt==1.11.2
