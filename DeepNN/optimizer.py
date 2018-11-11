import numpy as np
import utility as util
from tqdm import tqdm # to display instantaneous speed of loops
from network import Model

#####################################################
################### Optimization Operation###########
#####################################################
    
def adamsGradientDescent(batch, num_classes, learning_rate, dimension, num_channel, bheta1, bheta2, parameters, cost):
    '''
    This optimization method is called Adam optimization algorithm which is
    responsible to efficiently ensure the network’s parameters learn meaningful 
    representations.
    Adam makes use of the average of the second moments of the gradients 
    (the uncentered variance). Specifically, the algorithm calculates an 
    exponential moving average of the gradient and the squared gradient, 
    and the parameters bheta1 and bheta2 control the decay rates of these moving 
    averages. And finally, update the parameters through Adam gradient descnet.
    Parameters
    ----------
    batch : None
        The total number of training examples but can be sliced through
    num_classes: 10(int)
        The output labels (0 - 9)    
    learning_rate: None
        Learning rate is a hyper-parameter that controls how much the 
        weights of the network is being adjusted with respect to the 
        loss gradient. The lower the value, the slower we travel along the 
        downward slope. A low learning rate was utilized(using a low learning rate) 
        in order to ensure that no local minima was missed. 
    dimension : None
        Dimension of the image(height * width)   
    num_channel : None
        Number of channels
    bheta1: None
        The exponential decay rate for the first moment estimates   
    bheta2: None
        The exponential decay rate for the second-moment estimates
    parameters : None
        List of filters and weights and bias sizes 
    cost : None
        Average of the loss functions of the entire training sets 
    Returns
    -------   
        Returns costs and parameters                  
    '''

    [filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four] = parameters
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), num_channel, dimension, dimension)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS parameter
    dfilter_one = np.zeros(filter_one.shape)
    dfilter_two = np.zeros(filter_two.shape)
    dweight_three = np.zeros(weight_three.shape)
    dweight_four = np.zeros(weight_four.shape)
    dbias_one = np.zeros(bias_one.shape)
    dbias_two = np.zeros(bias_two.shape)
    dbias_three = np.zeros(bias_three.shape)
    dbias_four = np.zeros(bias_four.shape)
    
    v1 = np.zeros(filter_one.shape)
    v2 = np.zeros(filter_two.shape)
    v3 = np.zeros(weight_three.shape)
    v4 = np.zeros(weight_four.shape)
    bv1 = np.zeros(bias_one.shape)
    bv2 = np.zeros(bias_two.shape)
    bv3 = np.zeros(bias_three.shape)
    bv4 = np.zeros(bias_four.shape)
    
    s1 = np.zeros(filter_one.shape)
    s2 = np.zeros(filter_two.shape)
    s3 = np.zeros(weight_three.shape)
    s4 = np.zeros(weight_four.shape)
    bs1 = np.zeros(bias_one.shape)
    bs2 = np.zeros(bias_two.shape)
    bs3 = np.zeros(bias_three.shape)
    bs4 = np.zeros(bias_four.shape)
    
    for inputs in range(batch_size):
        
        x = X[inputs]
        # convert label to one-hot encoded array
        y = np.eye(num_classes)[int(Y[inputs])].reshape(num_classes, 1) 
        
        # Collect Gradients for training example
        model = Model()
        gradients, loss = model.convolution_layers(x, y, parameters, 
                                        conv_stride=1, pool_filter=2, pool_stride=2)
        [dfilter_one_, dfilter_two_, dweight_three_, dweight_four_, dbias_one_, dbias_two_, dbias_three_, dbias_four_] = gradients
        
        dfilter_one+=dfilter_one_
        dbias_one+=dbias_one_
        dfilter_two+=dfilter_two_
        dbias_two+=dbias_two_
        dweight_three+=dweight_three_
        dbias_three+=dbias_three_
        dweight_four+=dweight_four_
        dbias_four+=dbias_four_

        cost_+= loss

        # Parameter Update  
        v1 = bheta1*v1 + (1-bheta1)*dfilter_one/batch_size # momentum update
        s1 = bheta2*s1 + (1-bheta2)*(dfilter_one/batch_size)**2 # RMSProp update
        
        # combine momentum and RMSProp to perform update with Adam
        filter_one-= learning_rate * v1/np.sqrt(s1+1e-7) 
    
        bv1 = bheta1*bv1 + (1-bheta1)*dbias_one/batch_size
        bs1 = bheta2*bs1 + (1-bheta2)*(dbias_one/batch_size)**2
        bias_one -= learning_rate * bv1/np.sqrt(bs1+1e-7)

        v2 = bheta1*v2 + (1-bheta1)*dfilter_two/batch_size
        s2 = bheta2*s2 + (1-bheta2)*(dfilter_two/batch_size)**2
        filter_two -= learning_rate * v2/np.sqrt(s2+1e-7)
                    
        bv2 = bheta1*bv2 + (1-bheta1) * dbias_two/batch_size
        bs2 = bheta2*bs2 + (1-bheta2)*(dbias_two/batch_size)**2
        bias_two -= learning_rate * bv2/np.sqrt(bs2+1e-7)
    
        v3 = bheta1*v3 + (1-bheta1) * dweight_three/batch_size
        s3 = bheta2*s3 + (1-bheta2)*(dweight_three/batch_size)**2
        weight_three -= learning_rate * v3/np.sqrt(s3+1e-7)
    
        bv3 = bheta1*bv3 + (1-bheta1) * dbias_three/batch_size
        bs3 = bheta2*bs3 + (1-bheta2)*(dbias_three/batch_size)**2
        bias_three -= learning_rate * bv3/np.sqrt(bs3+1e-7)
    
        v4 = bheta1*v4 + (1-bheta1) * dweight_four/batch_size
        s4 = bheta2*s4 + (1-bheta2)*(dweight_four/batch_size)**2
        weight_four -= learning_rate * v4 / np.sqrt(s4+1e-7)
    
        bv4 = bheta1*bv4 + (1-bheta1)*dbias_four/batch_size
        bs4 = bheta2*bs4 + (1-bheta2)*(dbias_four/batch_size)**2
        bias_four -= learning_rate * bv4 / np.sqrt(bs4+1e-7)
    

        cost_ = cost_/batch_size
        cost.append(cost_)

        parameters = [filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four]
    
        return parameters, cost
