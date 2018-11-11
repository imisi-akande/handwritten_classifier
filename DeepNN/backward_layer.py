import numpy as np
from utility import ModelHelpers

class BackwardLayers:
    """
    Backward layer operation through convolution:
        This class implements a Backward propagation network.
        The derivations of the backward propagations will differ depending on 
        what layer we are propagating through.
        Main usage should consist of two steps namely convolution_backward
        and maxpoolBackwardl. This class contain predefined methods for the 
        backward prop in the fully connected layer network.
    """
    
    def __init__(self, stride):
        self.stride = stride

    def convolution_backward(self, dconv_previous, conv_input, filt):
        '''
        Backpropagation through a convolutional layer. 
        Taking the partial derivatives for all the components by initializing 
        with zero values. The layers were convolved by “flipping” the filters 
        (horizontally and vertically) on the input shape. The objective was to 
        looking out for how change in a single pixel in the x and y of the 
        input feature map affects the loss function which is the derivatives of 
        the output. The error is just assigned to where it comes from which is
        the “winning unit”. And really, because other units in the previous 
        layer’s pooling blocks did not contribute to it hence all the others are
        assigned values of zero.
        Parameters
        ----------
        dconv_previous: gradient of the cost with respect to output of the previous 
                    conv layer, numpy array of shape (n_H, n_W, n_ch)
        conv_input: current convolution layer
        filter: int- filter shape
        Returns
        -------    
        doutput: gradient of the cost with respect to input of the conv layer (X), 
              numpy array of shape (n_H_prev, n_W_prev) 
        dbias : gradient of cost with respect to bias on convolution layer.
        dfilter: gradient of cost with respect to filter.     
        '''
        (num_filter, num_channel, filter_size, _) = filt.shape
        (_, original_dimension, _) = conv_input.shape

        ## initialize derivatives
        doutput = np.zeros(conv_input.shape) 
        dfilter = np.zeros(filt.shape)
        dbias = np.zeros((num_filter,1))
        for current_filter in range(num_filter):
            # loop through using filters
            current_y = output_y = 0
            while current_y + filter_size <= original_dimension:
                current_x = output_x = 0
                while current_x + filter_size <= original_dimension:
                    # obtain and update loss gradient for the current filter 
                    dfilter[current_filter] += dconv_previous[current_filter, output_y, output_x] * conv_input[:, current_y:current_y+filter_size, current_x:current_x+filter_size]
                    # loss gradient of the input to the convolution operation 
                    doutput[:, current_y:current_y+filter_size, current_x:current_x+filter_size] += dconv_previous[current_filter, output_y, output_x] * filt[current_filter] 
                    current_x += self.stride
                    output_x += 1
                current_y += self.stride
                output_y += 1
                
            # loss gradient of the bias
            dbias[current_filter] = np.sum(dconv_previous[current_filter])
    
        return doutput, dfilter, dbias



    def maxpool_backward(self,dmax_pool, orig_max, filter_size):
        '''
        Backpropagation through a maxpooling layer. 
        During backprop, there is no learning taking place.
        The method computes the error which is acquired by this single value “winning unit”
        obtained during forward propagation. The gradients are 
        passed through the indices of greatest value in the original maxpooling 
        during the forward step.
        Parameters
        ----------
        dmax_pool: Gradient of the cost with respect to the pooling layer
        orig_max: captured maximum value obtained on the image.
        Returns
        -------  
        doutput: Gradient of cost with respect to current convolution.
        '''
        (num_channel, original_dimension, _) = orig_max.shape
    
        doutput = np.zeros(orig_max.shape)
    
        for current_channel in range(num_channel):
            current_y = output_y = 0
            while current_y + filter_size <= original_dimension:
                current_x = output_x = 0
                util = ModelHelpers()
                while current_x + filter_size <= original_dimension:
                    # obtain the index of the largest value in input for current window
                    (a, b) = util.nanargmax(orig_max[current_channel, current_y:current_y+filter_size, current_x:current_x+filter_size])
                    doutput[current_channel, current_y+a, current_x+b] = dmax_pool[current_channel, output_y, output_x]
                    current_x += self.stride
                    output_x += 1
                current_y += self.stride
                output_y += 1
        
        return doutput
