from builtins import object
import numpy as np

from implemention.utils.layers import *
from implemention.utils.fast_layers import *
from implemention.utils.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    [conv - relu - 2x2 max pool] - [affine - relu] - [affine] - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        maxed_height, maxed_width = (H-2)//2+1, (W-2)//2+1
        self.params['W2'] = weight_scale * np.random.randn(num_filters*maxed_height*maxed_width, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2)+np.sum(W2**2)+np.sum(W3))
        
        dout3, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
        grads['W3'] += self.reg * W3
        dout2, grads['W2'], grads['b2'] = affine_relu_backward(dout3, cache2)
        grads['W2'] += self.reg * W2
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout2, cache1)
        grads['W1'] += self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class CNN(object):
    """
    Structure: 
        input -> batchnorm -> (conv -> relu -> conv -> relu -> pool) * 3
              -> (affine -> relu) * 2 -> affine -> softmax
    """
    def __init__(self, input_dim=(1, 48, 48), num_filters=32, filter_size=3,
                 hidden_dim=500, num_classes=7, weight_scale=1e-3, reg=0.0001,
                 dtype=np.float64):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        
        # initialize the weights and bias
        C, H, W = input_dim
        # spatial batchnorm
        self.params['gamma'] = np.ones(C)
        self.params['beta'] = np.zeros(C)
        # N, 1, 48 ,48
        
        # conv -> relu #1 
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        # conv -> relu -> pool #1
        self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b2'] = np.zeros(num_filters)
        # N, 32, 24, 24
        C, H, W = num_filters, (H-2)//2+1, (W-2)//2+1
        
        # conv -> relu #2 
        self.params['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b3'] = np.zeros(num_filters)
        # conv -> relu -> pool #2
        self.params['W4'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b4'] = np.zeros(num_filters)
        # N, 32, 12, 12
        C, H, W = num_filters, (H-2)//2+1, (W-2)//2+1
        """
        # conv -> relu #3
        self.params['W5'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b5'] = np.zeros(num_filters)
        # conv -> relu -> pool #3
        self.params['W6'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b6'] = np.zeros(num_filters)
        # N, 32, 6, 6
        C, H, W = num_filters, (H-2)//2+1, (W-2)//2+1
        """
        # affine -> relu 1
        self.params['W7'] = weight_scale * np.random.randn(num_filters*H*W, hidden_dim)
        self.params['b7'] = np.zeros(hidden_dim)
        """
        # affine -> relu 2
        self.params['W8'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b8'] = np.zeros(hidden_dim)
        """
        # affine
        self.params['W9'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b9'] = np.zeros(num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    
    def loss(self, X, y=None):
                
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        bn_param = {'mode':'train'}
        scores = None
        caches = []
        # batchnorm
        out, cache = spatial_batchnorm_forward(X, self.params['gamma'], self.params['beta'], bn_param)
        caches.append(cache)
        
        # conv -> relu #1
        out, cache = conv_relu_forward(out, self.params['W1'], self.params['b1'], conv_param)
        caches.append(cache)
        # conv -> relu -> 2x2 maxpool #1
        out, cache = conv_relu_pool_forward(out, self.params['W2'], self.params['b2'], conv_param, pool_param)
        caches.append(cache)
        
        # conv -> relu #2
        out, cache = conv_relu_forward(out, self.params['W3'], self.params['b3'], conv_param)
        caches.append(cache)
        # conv -> relu -> 2x2 maxpool #2
        out, cache = conv_relu_pool_forward(out, self.params['W4'], self.params['b4'], conv_param, pool_param)
        caches.append(cache)
        """
        # conv -> relu #3
        out6, cache = conv_relu_forward(out5, self.params['W5'], self.params['b5'], conv_param)
        caches.append(cache)
        # conv -> relu -> 2x2 maxpool #3
        out7, cache = conv_relu_pool_forward(out6, self.params['W6'], self.params['b6'], conv_param, pool_param)
        caches.append(cache)
        """
        # affine -> relu #1
        out, cache = affine_relu_forward(out, self.params['W7'], self.params['b7'])
        caches.append(cache)
        """
        # affine -> relu #2
        out9, cache = affine_relu_forward(out8, self.params['W8'], self.params['b8'])
        caches.append(cache)
        """
        scores, cache = affine_forward(out, self.params['W9'], self.params['b9'])
        caches.append(cache)
        
        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dscores = softmax_loss(scores, y)
        for i in range(1,10):
            if i==5 or i==6 or i==8:
                continue
            W_ind = 'W'+str(i)
            loss += 0.5 * self.reg * np.sum(self.params[W_ind]**2)
        
        dout, grads['W9'], grads['b9'] = affine_backward(dscores, caches.pop())     # affine
        grads['W9'] += self.reg * self.params['W9']
        """
        dout, grads['W8'], grads['b8'] = affine_relu_backward(dout, caches.pop())   # affine relu
        grads['W8'] += self.reg * self.params['W8']
        """
        dout, grads['W7'], grads['b7'] = affine_relu_backward(dout, caches.pop())   # affine relu
        grads['W7'] += self.reg * self.params['W7']
        """
        dout, grads['W6'], grads['b6'] = conv_relu_pool_backward(dout, caches.pop())# conv relu pool
        grads['W6'] += self.reg * self.params['W6']
        dout, grads['W5'], grads['b5'] = conv_relu_backward(dout, caches.pop())     # conv relu
        grads['W5'] += self.reg * self.params['W5']
        """
        dout, grads['W4'], grads['b4'] = conv_relu_pool_backward(dout, caches.pop())# conv relu pool
        grads['W4'] += self.reg * self.params['W4']
        dout, grads['W3'], grads['b3'] = conv_relu_backward(dout, caches.pop())     # conv relu
        grads['W3'] += self.reg * self.params['W3']
        
        dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, caches.pop())# conv relu pool
        grads['W2'] += self.reg * self.params['W2']
        dout, grads['W1'], grads['b1'] = conv_relu_backward(dout, caches.pop())     # conv relu
        grads['W1'] += self.reg * self.params['W1']
        
        _, grads['gamma'], grads['beta'] = spatial_batchnorm_backward(dout, caches.pop()) # batchnorm
        
        assert len(caches) == 0, "cache is not devoid"
        
        return loss, grads