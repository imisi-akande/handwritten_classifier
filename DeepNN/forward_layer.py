import numpy as np

class ForwardLayers:
    """
    Forward layer operation:
        This class implements a Forward propagation network.
        The derivations of the forward propagations will differ depending on 
        what layer we are propagating through.
        The main usage consists of two steps namely forward_convolution_layer
        and forward_maxpool_layer. This class contain predefined methods for the 
        forward prop in the fully connected layer network.
    """
    
    def __init__(self, image):
        self.image = image

    def forward_convolution_layer(self,filt, bias, stride=1):
        """
        A 2D convolutional layer. Confolves `filt` over `image` using stride `stride`
        Parameters
        ----------
        Input image shape: (28 * 28 * 1) 
            (height * width * (channel):grayscale)
        filt : None
            The dimension of filters(kernels) (height, width).
        bias: None
            Allows shifting activation function left or right.    
        stride(stride) : int, default (1)
            The step of the convolution.

        Returns
        -------
        output: single value obtained from the element-wise multiplication of 
        the filter and a section of the input image which is summed to the bias.      
        """
        (num_filter, num_filter_channel, filter_size, _) = filt.shape # filter dimensions
        num_channel, input_dimension, _ = self.image.shape # image dimensions
    
        output_dimension = int((input_dimension - filter_size)/stride)+1 # calculate output dimensions

        # To ensure Dimensions of filter and the input image matches"
        assert num_channel == num_filter_channel 
    
        output = np.zeros((num_filter,output_dimension,output_dimension))
    
        # convolve the filter over every part of the image, 
        # adding the bias at each step. 
        for current_filter in range(num_filter):
            current_y = output_y = 0
            while current_y + filter_size <= input_dimension:
                current_x = output_x = 0
                while current_x + filter_size <= input_dimension:
                    output[current_filter, output_y, output_x] = np.sum(filt[current_filter] * self.image[:,current_y:current_y+filter_size, current_x:current_x+filter_size]) + bias[current_filter]
                    current_x += stride
                    output_x += 1
                current_y += stride
                output_y += 1
        return output

    def forward_maxpool_layer(self,filter_size=2, stride=2):
        '''
        The max-pool operation downsample the representation size
        using kernels and strides. This method computes results in a 2 * 2 pooling 
        block being reduced to a single value 'winning unit' of the input image dimension.  
        Operation utilizes a for loop and a two while loops. The for-loop is 
        used to pass through each layer of the input image, and the while-loops 
        slide the window over every part of the image. At each step, np.max (NumPy’s max) method 
        was used to obtain the maximum value:
        Parameters
        ----------
        image : shape: (28 * 28 * 1)
        filt : int:(2)
            The dimension of filters(kernels) (height, width)
        stride(stride) : int, default (2)
            The step of the downsample.
        Returns
        -------
        downsample : Doing 2x2 pooling with stride of 2 and no padding essentially 
        reduces the image dimension by half.(14 * 14)
        '''
        num_channel, previous_height, previous_weight = self.image.shape
    
        height = int((previous_height - filter_size)/stride)+1
        width = int((previous_weight - filter_size)/stride)+1
    
        downsample = np.zeros((num_channel, height, width))
        for inputs in range(num_channel):
            # slide maxpool window over each part of the image and assign the
            #  max value at each step to the output
            current_y = output_y = 0
            while current_y + filter_size <= previous_height:
                current_x = output_x = 0
                while current_x + filter_size <= previous_weight:
                    downsample[inputs, output_y, output_x] = np.max(self.image[inputs, current_y:current_y+filter_size, current_x:current_x+filter_size])
                    current_x += stride
                    output_x += 1
                current_y += stride
                output_y += 1
        return downsample 

    def softmax(self,X):
        '''
        This activation function is used to represent a categorical distribution
        over class labels, and obtain the probabilities of each input element 
        belonging to a label.
        The softmax activation function takes an output from the final dense layer.
        Parameters
        ----------
        X :  each element in the final layer’s outputs
        Returns
        -------
        The probability of each class (each digit) given the input image
        '''
        output = np.exp(X)
        return output/np.sum(output)

    def categorical_cross_entropy(self,desired_label, actual_label):
        '''
        This computes the loss function. It shows how accurate our network was 
        in predicting the handwritten digit from the input image.
        Parameters
        ----------
        actual_label : The network's prediction
        desired_label: Desired output label
        Returns
        -------
        loss function: defines the model’s accuracy when predicting the 
        output digit.
        '''
        return -np.sum(actual_label * np.log(desired_label))