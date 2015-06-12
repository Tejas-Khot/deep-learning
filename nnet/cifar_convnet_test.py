"""ConvNet CIFAR-10 test"""

import time

from conv import ConvLayer
from cost import L2_regularization
from mlp import HiddenLayer
from mlp import SoftmaxLayer
import numpy as np
from optimize import gd_updates
import theano
import theano.tensor as T
import utils as utils


t_0=time.time()

n_epochs=100
batch_size=100
num_filters=[50,20]

X_train, Y_train, X_test, Y_test=utils.load_CIFAR10("/home/tejas/Desktop/cifar-10-batches-py")

print X_train.shape
print X_test.shape

train_set_x=theano.shared(np.asarray(X_train,
                                     dtype='float32'),
                                     borrow=True)

train_set_y=theano.shared(np.asarray(Y_train,
                                     dtype='float32'),
                                     borrow=True)

train_set_y=T.cast(train_set_y, dtype="int32")                          

test_set_x=theano.shared(np.asarray(X_test,
                                    dtype='float32'),
                                    borrow=True)
test_set_y=theano.shared(np.asarray(Y_test,
                                    dtype='float32'),
                                    borrow=True)

test_set_y=T.cast(test_set_y, dtype="int32")

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"

X=T.matrix("data")
y=T.ivector("label")
idx=T.lscalar()

images=X.reshape((batch_size, 1, 32, 32))

# input size (32, 32), (7, 7)
layer_0=ConvLayer(image_shape=(batch_size, 1, 32, 32),
                 filter_shape=(num_filters[0], 1, 7, 7),
                 pool=True,
                 activation_mode="relu")
layer_0.initialize()
layer_0.apply_conv(images)
filters=layer_0.W

# input size (13, 13), (4, 4)
layer_1=ConvLayer(image_shape=(batch_size, num_filters[0], 13, 13),
                 filter_shape=(num_filters[1], num_filters[0], 4, 4),
                 pool=True,
                 activation_mode="relu")
layer_1.initialize()
layer_1.apply_conv(layer_0.get_output())  

# output size (5, 5)
layer_2=HiddenLayer(input=layer_1.get_output().flatten(2),
                    n_in=num_filters[1]*25,
                    n_out=500)
layer_2.initialize()

                    
layer_3=SoftmaxLayer(input=layer_2.get_output(),
                     n_in=500,
                     n_out=10)
layer_3.initialize()

params=layer_0.params+layer_1.params+layer_2.params+layer_3.params
                     
cost=layer_3.cost(y)+L2_regularization(params, 0.01)
updates=gd_updates(cost=cost, params=params)

train=theano.function(inputs=[idx],
                      outputs=cost,
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                              y: train_set_y[idx * batch_size: (idx + 1) * batch_size]})

test=theano.function(inputs=[idx],
                     outputs=layer_3.error(y),
                     givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                             y: test_set_y[idx * batch_size: (idx + 1) * batch_size]})
                              
print "[MESSAGE] The model is built"

test_record=np.zeros((n_epochs, 1))
epoch = 0
while (epoch < n_epochs):
    epoch+=1
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train(minibatch_index)
        
        iteration = (epoch - 1) * n_train_batches + minibatch_index
        
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL'
            test_losses = [test(i) for i in xrange(n_test_batches)]
            test_record[epoch-1] = np.mean(test_losses)
            
            print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.))

t_1=time.time()   

print "[MESSAGE] Total time taken: ", t_1-t_0         