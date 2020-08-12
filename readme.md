Dataset: [CIFAR100](http://github.com), 100 classes and 600 images per class. 50,000 images for
 training and, 10,000 images reserved for testing.

clone repo:
`git clone https://github.com/RezaSoleymanifar/alex_net_gpu.git`

start training:
`python alex_net_gpu.py`

Best model is saved each time an improvement in validation loss is triggered. tensorboard results are saved at each epoch.
For tensorboard run:

`tensorboard --logdir=tboard`