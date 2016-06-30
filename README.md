MNIST Dataset Renderer
======================


### Info

This script generates N random single digit images.
The digit in each image is randomly sized and placed while some distortion and
noise gets applied.

See: http://yann.lecun.com/exdb/mnist/
for the original dataset.


### Requirements

This python script is mostly using numpy and opencv.
Make sure you have both of them.


### Usage

To get help, just type:
```sh
python mnist_render.py -h
```

Try for example:
```sh
python mnist_render.py -out . -num 1024 -seed 100 -dmax 1.0 -gnmax 0.4 -spnmax 0.1 -concat
```

It will generate 1024 random digit images with some distortion, gaussian and
salt-pepper noise and concatenate all the images into one big image called
"mnist.png".

Note that the default image resolution is 28x28.
1024 images (32x32 images matrix) concatenated into one image will have a
resolution of 896x896.
