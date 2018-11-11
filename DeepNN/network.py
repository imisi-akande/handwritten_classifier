import numpy as np
import backward_layer as backward
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from utility import ModelHelpers
from tqdm import tqdm
from optimizer import *
from forward_layer import ForwardLayers
from backward_layer import BackwardLayers

class Model:
    """
    With the aim of abstraction, this class entails the model entity. The model
    defines a method called 'convolution_layers' that integrates the forward and
    the backward operations of the convolutional neural network. It takes the 
    network’s parameters and hyperparameters as inputs and returns the gradients
    as output.
    """
    
    def convolution_layers(self,image=None, actual_label=None, parameters=None, 
                            conv_stride=None, pool_filter=None, pool_stride=None):
        """
        A fully connected network that uses the Softmax and creates the 
        probabilities. Given the number of classes (10 in total) 
        and the size of each training image example (28x28px.), this network 
        architecture implements the task of digit recognition. 
        The network uses convolutional layers followed by a 
        max pooling operation to extract features from the input image. 
        After the max pooling operation, the representation was flattened and 
        passed through a Multi-Layer Perceptron (MLP) to carry out the task of 
        classification.
        Parameters
        ----------
        image : None
            The input image dimension
        actual_label : None
            The predicted output (0 to 9)
        parameters : None
            The filters, weights and bias utilized
        conv_stride : None
            The number of strides used in convolution layers        
        pool_filt : None
            The dimension of filters(kernels) (height, width)
        pool_stride : None  
            The number of strides used on the pooling layer  
        Returns
        -------
        gradients and loss: The result measures how the cost changes in the 
            vicinity of the current position respect to the inputs parameters
            and hyperparameters.

        """

        [filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four] = parameters 
    
        ################################################
        ############## Forward Operation ###############
        ################################################
        forward = ForwardLayers(image)
        # first convolution operation
        conv1 = forward.forward_convolution_layer(filter_one, bias_one, conv_stride)
        conv1[conv1<=0] = 0 # pass through ReLU non-linearity
        forward = ForwardLayers(conv1)
        # second convolution operation
        conv2 = forward.forward_convolution_layer(filter_two, bias_two, conv_stride) 
        conv2[conv2<=0] = 0 # pass through ReLU non-linearity
        forward = ForwardLayers(conv2)
        # maxpooling operation
        pooled = forward.forward_maxpool_layer(pool_filter, pool_stride) 
    
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
        z = weight_three.dot(fc) + bias_three # first dense layer
        z[z<=0] = 0 # pass through ReLU non-linearity
    
        output = weight_four.dot(z) + bias_four # second dense layer

        # predict class probabilities with the softmax activation function
        desired_label = forward.softmax(output) 
    
        loss = forward.categorical_cross_entropy(desired_label, actual_label) # categorical cross-entropy loss
        
        ################################################
        ############# Backward Operation ###############
        ################################################
        doutput = desired_label - actual_label # derivative of loss w.r.t. final dense layer output
        dweight_four = doutput.dot(z.T) # loss gradient of final dense layer weights
        # loss gradient of final dense layer biases
        dbias_four = np.sum(doutput, axis = 1).reshape(bias_four.shape) 
    
        dz = weight_four.T.dot(doutput) # loss gradient of first dense layer outputs 
        dz[z<=0] = 0 # backpropagate through ReLU 
        dweight_three  = dz.dot(fc.T)
        dbias_three = np.sum(dz, axis = 1).reshape(bias_three.shape)

        # loss gradients of fully-connected layer (pooling layer)
        dfc = weight_three.T.dot(dz) 
        # reshape fully connected into dimensions of pooling layer
        dmax_pool = dfc.reshape(pooled.shape) 
        backward =  BackwardLayers(pool_stride)
        # backprop through the max-pooling layer(only neurons with highest
        # activation in window get updated)
        dconv2 = backward.maxpool_backward(dmax_pool, conv2, pool_filter) 
        dconv2[conv2<=0] = 0 # backpropagate through ReLU
        backward = BackwardLayers(conv_stride)
        # backpropagate previous gradient through second convolutional layer.
        dconv1, dfilter_two, dbias_two = backward.convolution_backward(dconv2, conv1, filter_two) 
        dconv1[conv1<=0] = 0 # backpropagate through ReLU
        backward = BackwardLayers(conv_stride)
        # backpropagate previous gradient through first convolutional layer.
        dimage, dfilter_one, dbias_one = backward.convolution_backward(dconv1, image, filter_one) 
    
        gradients = [dfilter_one, dfilter_two, dweight_three, dweight_four, dbias_one, dbias_two, dbias_three, dbias_four] 
    
        return gradients, loss

    
    #####################################################
    ##################### Training Ops###################
    #####################################################

    def train(num_classes = 10, learning_rate = 0.0001, bheta1 = 0.95, bheta2 = 0.99, 
                img_dim = 28, img_depth = 1, f_layer1 = 10, f_layer2 = 5, num_filt1 = 32,
                num_filt2 = 16, batch_size = 100, num_epochs = 200, save_path = 'parameters.pkl'):
        """
        Training method is an approach to ensure the model is learning on a 
        particular set of data. In this case the model is trained on MNIST 
        dataset so that the Machine can learn and generally predict. To illustrate,
        the machine can predict that an handwritten digit is '3' out of the 
        remaining classes(0,1,2,4,5,6,7,8,9).
        Given 60,000 training dataset, the model is built. This model generally
        will try to predict one variable based on all the others as described 
        above. 
        This training method implements Adam optimization algorithm for optimization.
        Parameters
        ----------
        num_classes: 10(int)
            The output labels (0 - 9)
        learning_rate : 0.0001(float)
            Learning rate is a hyper-parameter that controls how much the 
            weights of the network is being adjusted with respect to the 
            loss gradient. The lower the value, the slower we travel along the 
            downward slope. A low learning rate was utilized(using a low learning rate) 
            in order to ensure that no local minima was missed.
        bheta1: 0.95(float)
            The exponential decay rate for the first moment estimates 
        bheta2: 0.99(float)
            The exponential decay rate for the second-moment estimates 
        img_dim : 28 * 28
            The dimension of the image (height * width)         
        image_depth : 1
            The channel of the image G (greyscale). If RGB then image depth = 3
        f_layer1 : 10
            The filter dimensions of the first convolution layer   
        f_layer2 : 5
            The filter dimensions of the second convolution layer   
        num_filt1 : 32
            Number of output channels of the first convolution layer
        num_filt2 : 16
            Number of output channels of the second convolution layer
        batch_size : 100
            The total number of training examples present in a batch
        num_epochs : 200 
            The number of times the entire dataset in the batch passed 
            forward and backward through the network
        save_path : parameters.pkl
            The network hyperparameters saved file.               
        Returns
        -------
        cost : computes the average of the loss functions of the entire 
            training sets
        """
        # training data
        m =60000
        util = ModelHelpers()
        X = util.extract_dataset('mnist_data/train-images-idx3-ubyte.gz', m, img_dim)
        x_shaped = np.reshape(X, [-1, 28, 28, 1])
        y_dash = util.extract_desired_labels('mnist_data/train-labels-idx1-ubyte.gz', m).reshape(m,1)
        X-= int(np.mean(x_shaped))
        X/= int(np.std(X))
        train_data = np.hstack((X,y_dash))
    
        np.random.shuffle(train_data)

        ## Initializing all the parameters
        filter_one, filter_two, weight_three , weight_four = (num_filt1 ,img_depth,f_layer1,f_layer1), (num_filt2 ,num_filt1,f_layer2,f_layer2), (1024,784), (10, 1024)

        filter_one = util.initialize_filter(filter_one)
        filter_two = util.initialize_filter(filter_two)
        weight_three = util.initialize_weight(weight_three)
        weight_four = util.initialize_weight(weight_four)

        bias_one = np.zeros((filter_one.shape[0],1))
        bias_two = np.zeros((filter_two.shape[0],1))
        bias_three = np.zeros((weight_three.shape[0],1))
        bias_four = np.zeros((weight_four.shape[0],1))

        parameters = [filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four]

        cost = []

        print("Learning-rate:"+str(learning_rate)+", Batch Size:"+str(batch_size))

        for epoch in range(num_epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

            t = tqdm(batches)
            for x,batch in enumerate(t):
                parameters, cost = adamsGradientDescent(batch, num_classes, learning_rate, img_dim, img_depth, bheta1, bheta2, parameters, cost)
                t.set_description("Cost: %.2f" % (cost[-1]))
            
        to_save = [parameters, cost]
    
        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)
        
        return cost

    parser = argparse.ArgumentParser(description='A module for training a convolutional neural network.')
    parser.add_argument('save_path', metavar = 'Save Path', help='File that stores parameters.')

    #####################################################
    ##################### Measure Performance############
    #####################################################

    if __name__ == '__main__':
    
        args = parser.parse_args()
        save_path = args.save_path
        model = Model()
        cost = train(save_path = save_path)

        parameters, cost = pickle.load(open(save_path, 'rb'))
        [filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four] = parameters
    
        # Plot cost over number of iterations
        plt.plot(cost, 'r')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.legend('Loss', loc='upper right')
        plt.show()

        # Get test data
        m =10000
        util = ModelHelpers()
        X = util.extract_dataset('./mnist_data/t10k-images-idx3-ubyte.gz', m, 28)
        y_dash = util.extract_desired_labels('./mnist_data/t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
        
        # Normalize the data
        X-= int(np.mean(X)) # subtract mean
        X/= int(np.std(X)) # divide by standard deviation
        test_data = np.hstack((X,y_dash))
    
        X = test_data[:,0:-1]
        X = X.reshape(len(test_data), 1, 28, 28)
        y = test_data[:,-1]

        corr = 0
        digit_count = [0 for inputs in range(10)]
        digit_correct = [0 for inputs in range(10)]
   
        print()
        print("Next, Computing accuracy operation on the test dataset:")

        t = tqdm(range(len(X)), leave=True)

        for inputs in t:
            x = X[inputs]
            pred, prob = util.predict(x, filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four)
            digit_count[int(y[inputs])]+=1
            if pred==y[inputs]:
                corr+=1
                digit_correct[pred]+=1

            t.set_description("Acc:%0.2f%%" % (float(corr/(inputs+1))*100))
        
        print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
        x = np.arange(10)
        digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
        plt.xlabel('Digits')
        plt.ylabel('Recall')
        plt.title("Recall on Test Set")
        plt.bar(x,digit_recall)
        plt.show()