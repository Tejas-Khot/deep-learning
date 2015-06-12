"""
@author: Tejas Khot
@contact: tjskhot@gmail.com

@note: CIFAR-10 testing using CNN 
"""
import time
import cPickle as pickle
from conv import ConvLayer
from mlp import HiddenLayer
from mlp import SoftmaxLayer
import numpy as np
import theano
import theano.tensor as T
import utils as utils

t_0=time.time()

n_epochs=50
training_portion=1
batch_size=200
num_filters=[50, 20]


X_train, Y_train, X_test, Y_test=utils.load_CIFAR10("/home/tejas/Desktop/cifar-10-batches-py")

X_train=np.mean(X_train, 3)
X_test=np.mean(X_test, 3)
X_train=X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])/255.0
X_test=X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])/255.0

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

n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion)
n_test_batches=test_set_x.get_value(borrow=True).shape[0]

print n_train_batches
print n_test_batches
    
n_train_batches /= batch_size # number of train data batches
n_test_batches /= batch_size  # number of test data batches

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"

index=T.lscalar() # batch index

X=T.matrix('X')  # input data source
y=T.ivector('y') # input data label

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

cost=layer_3.cost(y)+0.001*((layer_0.W**2).sum()+(layer_1.W**2).sum()+(layer_2.W**2).sum()+(layer_3.W**2).sum())

gparams=T.grad(cost, params)

updates=[(param_i, param_i-0.1*grad_i) for param_i, grad_i in zip(params, gparams)]
print updates         

test_model=theano.function(inputs=[index],
                           outputs=layer_3.error(y),
                           givens={X: test_set_x[index * batch_size:(index + 1) * batch_size],
                                   y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    
train_model=theano.function(inputs=[index],
                            outputs=cost,
                            updates=updates,
                            givens={X: train_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: train_set_y[index * batch_size: (index + 1) * batch_size]})

print "[MESSAGE] The model is built"
print "[MESSAGE] Start training"       

validation_frequency=n_train_batches
validation_record=np.zeros((n_epochs, 1))
test_record=np.zeros((n_epochs, 1))

epoch=0
while (epoch < n_epochs):
    epoch=epoch + 1
    for minibatch_index in xrange(n_train_batches):
      mlp_minibatch_avg_cost=train_model(minibatch_index)
      iter=(epoch - 1) * n_train_batches + minibatch_index
          
      if (iter + 1) % validation_frequency == 0:
        test_losses=[test_model(i) for i in xrange(n_test_batches)]
        test_record[epoch-1]=np.mean(test_losses)
                  
        print(('     epoch %i, minibatch %i/%i, test error %f %%') %
              (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.))
            

filters=layer_0.W

## Prepare data
train_output_feature=theano.function(inputs=[index],
                                    outputs=layer_2.get_output(layer_1.get_output(layer_0.get_output(X.reshape((batch_size, 1, 32, 32)))).flatten(2)),
                                    givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]})

train_feature=np.asarray([])
for batch_index in xrange(n_train_batches):
    temp=train_output_feature(batch_index)
  
    if not train_feature.size:
        train_feature=temp
    else:
        train_feature=np.vstack((train_feature, temp))
    
train_feature=np.hstack((train_set_y.eval()[None].T, train_feature))

print train_feature.shape

train_feature_random=train_feature
train_feature.view("float32, float32, float32").sort(order=["f1"], axis=0)

print train_feature.shape
print "[MESSAGE] Writing training set to file"

pickle.dump(train_feature, open("cifar10_train_convnet_feature_500_ordered.pkl", "w"))
pickle.dump(train_feature_random, open("cifar10_train_convnet_feature_500_random.pkl", "w"))

print "[MESSAGE] Training set is prepared"

test_output_feature=theano.function(inputs=[index],
                                    outputs=layer_2.get_output(layer_1.get_output(layer_0.get_output(X.reshape((batch_size, 1, 32, 32)))).flatten(2)),
                                    givens={X: test_set_x[index * batch_size: (index + 1) * batch_size]})

test_feature=np.asarray([])        
for batch_index in xrange(n_test_batches):
    temp=test_output_feature(batch_index)
  
    if not test_feature.size:
        test_feature=temp
    else:
        test_feature=np.vstack((test_feature, temp))
    
test_feature=np.hstack((test_set_y.eval()[None].T, test_feature))

test_feature_random=test_feature
test_feature.view("float32, float32, float32").sort(order=["f1"], axis=0)

print test_feature.shape
print "[MESSAGE] Writing testing set to file"

pickle.dump(test_feature, open("cifar10_test_convnet_feature_500_ordered.pkl", "w"))
pickle.dump(test_feature_random, open("cifar10_test_convnet_feature_500_random.pkl", "w"))

print "[MESSAGE] Testing set is prepared"
t_1=time.time()
print "Total time taken is ", t_0 - t_1
