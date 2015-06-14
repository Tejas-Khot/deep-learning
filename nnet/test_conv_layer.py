import time
import numpy as np 
import theano
import theano.tensor as T 
from convnet import ConvLayer
import utils as utils
from theano import function

t_0=time.time()

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
idx=T.lscalar()

images=X.reshape((batch_size, 1, 32, 32))
layer_0=ConvLayer(image_shape=(batch_size, 1, 32, 32),
                 filter_shape=(num_filters[0], 1, 7, 7),
                 pool=True,
                 activation_mode="relu")
layer_0.initialize()

train_output_feature=theano.function(inputs=[idx],
                                    outputs=layer_0.get_output(X.reshape((batch_size, 1, 32, 32))),
                                    givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

out=[]
for batch_index in xrange(0, n_train_batches):
	temp=train_output_feature(batch_index)
	out.append(temp)

	
print type(out), out.shape
print out
filters=layer_0.W

print "done"
# f=function([out], out*1)

# print f(out)
# print type(f(out))
