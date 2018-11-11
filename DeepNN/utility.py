
from abc import ABCMeta, abstractmethod
from forward_layer import ForwardLayers
import gzip
import numpy as np

class Utility(metaclass = ABCMeta):
    """ This file contains different utility functions that are not very
    connected but rather help in processing the outputs into a more 
    understandable way.
    """
    @abstractmethod
    def extract_dataset():
        pass
    @abstractmethod
    def extract_desired_labels():
        pass
    @abstractmethod
    def initialize_filter():
        pass
    @abstractmethod
    def initialize_weight():
        pass
    @abstractmethod
    def nanargmax():
        pass
    @abstractmethod
    def predict():
        pass    

class ModelHelpers(Utility):        

    def extract_dataset(self, input_file, num_images, image_width):
        '''
        The MNIST data set images are stored as tensors hence the need to
        extract images by reading the file bytestream. Reshape the read values 
        into a 3D matrix of dimensions [m, height, width], where m is the number of 
        training examples, height represents the height of the image and width 
        represents the width of the image.
        Parameters
        ----------
        input_file : the file that contains the training set(string)
        num_images : the number of images(int)
        image_width : The width of the image(int)(28)
        Returns
        -------
        input_data : all the training sets
        '''
        print('Extracting', input_file)
        with gzip.open(input_file) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(image_width * image_width * num_images)
            input_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            input_data = input_data.reshape(num_images, image_width*image_width)
            return input_data

    def extract_desired_labels(self, input_file, num_images):
        '''
        Extract desired labels into a single column vector of integer values 
        with dimensions of [m, 1], 
       
        Parameters
        ----------
        input_file: the file that contains the training set(string)
        num_images(m) : the number of images(int)
        Returns
        -------
        labels : all the target sets
        '''
        print('Extracting', input_file)
        with gzip.open(input_file) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def initialize_filter(self, size, scale = 0.1):
        '''
        This method Initializes the filter using a normal distribution with 
        a standard deviation inversely proportional the square root of the 
        number of units
        Parameters
        ----------
        size : The filter size
        scale : The standard deviation which is equal to 0.1
        Returns
        -------
        random samples from a normal distribution and select values close
        to mean for the filter.
        '''
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def initialize_weight(self, size):
        '''
        Parameters
        ----------
        size : Initial weight size 
               Initialize weights with a random normal distribution
        Returns
        -------
        random samples from a normal distribution and select values close
        to mean for the weight       
        '''
        return np.random.standard_normal(size=size) * 0.1
       
    def nanargmax(self, arr):
        '''
        Parameters
        ----------
        arr : list
        Returns
        -------
        indexes of the maximum value in the array ignoring Nan's. 
        '''
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs    

    def predict(self, image, filter_one, filter_two, weight_three, weight_four, bias_one, bias_two, bias_three, bias_four, conv_stride = 1, pool_filter = 2, pool_stride = 2):
        '''
        Parameters
        ----------
        image : 28 * 28 * 1
        filter_one, filter_two : size of filters
        weight_three, weight_four: weight values
        bias_one, bias_two, bias_three, bias_four : bias values
        conv_stride :  step of the convolution operation equals 2
        pool_filter : pool filter equals 2
        pool_stride :  step of the pooling operation equals 2
        Returns
        -------
        np.argmax(desired_label) : The index of the maximum values
        np.max(desired_label) : The maximum probability values
        '''
        forward = ForwardLayers(image)
        conv1 = forward.forward_convolution_layer(filter_one, bias_one, conv_stride) # first convolution operation
        conv1[conv1<=0] = 0 #relu activation
        forward = ForwardLayers(conv1)
        
        conv2 = forward.forward_convolution_layer(filter_two, bias_two, conv_stride) # second convolution operation
        conv2[conv2<=0] = 0 # pass through ReLU non-linearity
        forward = ForwardLayers(conv2)
        pooled = forward.forward_maxpool_layer(pool_stride, pool_stride) # maxpooling operation
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
        
        z = weight_three.dot(fc) + bias_three # first dense layer
        z[z<=0] = 0 # pass through ReLU non-linearity
        
        output = weight_four.dot(z) + bias_four # second dense layer
        # predict class probabilities with the softmax activation function
        forward = ForwardLayers(image)
        desired_label = forward.softmax(output) 
        
        return np.argmax(desired_label), np.max(desired_label)