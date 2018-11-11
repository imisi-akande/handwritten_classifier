# Convolution Neural Network on handwritten digits

## Description
This repository contains the Convolution Neural Network architecture trained on the MNIST handwritten digits implemented using Python programming language and Numpy library. Here is the description of the model implemented:

1. 2D convolution layer:
    a. 2D convolution with a 10x10 filter, a stride of 1, and 32 output channels.
    b. Add a bias to each output channel. There are 32 bias values, one for
       each output channel.
    c. ReLU activation function.

2. 2D convolution layer:
    a. 2D convolution with a 5x5 filter, a stride of 1, and 16 output channels.
    b. Add a bias to each output channel. There are 16 bias values.
    c. ReLU activation function.
    d. 2x2 max pooling layer.
    e. Flatten the output of layer 2 into a vector.

3. Fully connected layer:
    a. Output neurons of 1024, and with a bias vector.
    b. ReLU activation function.

4. Fully connected layer:
    a. Output neurons of 10, and with a bias vector.
    b. Softmax activation function.

5. Cross entropy loss.

## Restrictions:
1. Initialize all bias vectors with a constant of value 0.1.

2. Train with a batch size of 100 and upto 200 epochs.

## Train the network

#### Run the following commands: 

> $ pip install -r requirements.txt

> $ python network.py parameters.pkl
