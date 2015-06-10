'''
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: Theano implementation of a MLP Hidden Layer
'''

import nnfuns as nnf
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class HiddenLayer(object):
    """
    Theano implementation of MLP Hidden Layer
    """
    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 activation_mode="tanh"):
        """
        A typical hidden laye of MLP where units are fully connected.
        Defaults to the tanh activation function.
        
        @param input: symbolic tensor of shape (n_examples, n_in) (theano.tensor.dmatrix)
        @param n_in: dimension of input data (int)
        @param n_out: dimension of hidden unit (int)
        @param activation_mode: activation mode,
                                "tanh" for tanh function
                                "relu" for ReLU function
                                "sigmoid" for Sigmoid function
                                "softplus" for Softplus function
                                "linear" for linear function (string)   
        """
        
        self.input=input
        self.n_in=n_in
        self.n_out=n_out
        self.activation_mode=activation_mode
        # random number generator for initialize weights (numpy.random.RandomState)
        self.rng=np.random.RandomState(23455)
        
        if (self.activation_mode=="tanh"):
          self.activation=nnf.tanh
        elif (self.activation_mode=="relu"):
          self.activation=nnf.relu
        elif (self.activation_mode=="sigmoid"):
          self.activation=nnf.sigmoid
        elif (self.activation_mode=="softplus"):
          self.activation=nnf.softplus
        elif (self.activation_mode=="softmax"):
          self.activation=nnf.softmax
        else:
          raise ValueError("Value %s is not a valid choice of activation function"
                           % self.activation_mode)

    def initialize(self, W=None, b=None):
        """
        Set values for weights and biases
        
        @param W: the weight matrix of shape (n_in,n_out)
        @param b: bias vector of shape (n_out,)

        @note 
        `W` is initialized with `W_values` which is uniformly sampled from a distribution in range
        sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)) for tanh activation function
        
        According to the results presented in [Xavier10], the initial weights for sigmoid should 
        be set 4 times that of tanh for better results.
        """
        if W is None:
            W_bound=np.sqrt(6. / (self.n_in + self.n_out))
            W_values=np.asarray(self.rng.uniform(low=-W_bound,
                                                 high=W_bound,
                                                 size=(self.n_in, self.n_out)),
                                dtype='float32')

            if (self.activation_mode=="sigmoid"):
                W_values *= 4

            self.W=theano.shared(W_values,
                                 name='W',
                                 borrow=True).astype('float32')
        
        if b is None:
            b_values=np.zeros((self.n_out,), dtype='float32')
            self.b=theano.shared(value=b_values, borrow=True)
        
        self.weighted_sum=None
        self.output=None
    
    def get_weighted_sum(self,
                         input,
                         W=None,
                         b=None):
        """
        Get weighted sum of the input
        @return (W*input + b)
        """
        return T.dot(input, W)+b
    
    def get_pre_activation(self, input):
        """
        Get weighted sum using self weight and bias
        @return (W*input + b) with instance values for W, b
        """
        self.weighted_sum=self.get_weighted_sum(input, W=self.W, b=self.b)
        return self.weighted_sum
    
    def get_activation(self, s):
        """
        Get layer activation based on activation function
        @return f(s) where s is a pre-activation (W*input + b)
        """
        return self.activation(s)
    
    @property
    def params(self):
        return (self.W, self.b)
    
    def get_output(self, x=None):
        """
        Get final output of hidden layer activation
        @param x: if no explicit output given, use self.input
        
        @return f(W*input + b) where f is the activation function
        """
        if x is None:
            x=self.input
        self.output=self.get_activation(self.get_pre_activation(x))
        return self.output


class SoftmaxLayer(HiddenLayer):
    """
    Softmax Layer implementation
    
    @param input: symbolic tensor of shape (n_examples, n_in) (theano.tensor.dmatrix)
    @param n_in: dimension of input data (int)
    @param n_out: dimension of hidden unit (int)
    @param activation_mode: activation mode,
                            "tanh" for tanh function
                            "relu" for ReLU function
                            "sigmoid" for Sigmoid function
                            "softplus" for Softplus function
                            "linear" for linear function (string) 
    """
    def __init__(self,
                 input,
                 n_in,
                 n_out):
      
        super(SoftmaxLayer, self).__init__(input=input,
                                           n_in=n_in,
                                           n_out=n_out,
                                           activation_mode="softmax")
        self.output=None
        self.prediction=None
    
    def get_output(self, x=None):
        super(SoftmaxLayer, self).get_output(x)                            
        self.prediction=T.argmax(self.output, axis=1)
        return self.output
                                         
    def cost(self, y):
        """
        Cost of softmax regression
        """
        if self.output is None:
            self.get_output()
        return T.nnet.categorical_crossentropy(self.output, y).mean()
    
    def error(self, y):
        """
        Difference between true label and prediction
        """
        if self.prediction is None:
            self.get_output()
        return T.mean(T.neq(self.prediction, y))










