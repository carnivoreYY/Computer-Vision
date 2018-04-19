In this project:
I choose small batch_size = 50 with high tiers = 2000 since the model will almost converge at the end. On the other side, I will choose drop_out rate = 0.4 to avoid overfitting.
Also, when it comes to CNN layers, I choose padding = "same". For pooling layers, I choose pool3x3 and pool4X4 with strides=2. The structure of my network is 2 CNN layers followed by 1 pooling layer which is inspired by VGG.

How to run:
$ conda install -c conda-forge tensorflow
$ python proj3.py