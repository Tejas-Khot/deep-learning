"""Feature extraction """

import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T
import time
import nnet.datasets as ds
from nnet.convnet import ReLUConvLayer

t_0=time.time()
n_epochs=100
batch_size=100

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("/home/ubuntu/destin/cifar-10-batches-py")

Xtr=np.mean(Xtr, 3)
Xte=np.mean(Xte, 3)

Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0


print "Training data shape: ", Xtr.shape, "\t Labels shape: ", Ytr.shape

train_set_x, train_set_y=ds.shared_dataset((Xtrain, Ytr))
test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte))

del train_set_y, test_set_y, Ytr, Yte

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

print "n_train_batches : ", n_train_batches
print "n_test_batches : ", n_test_batches

print "[MESSAGE] The data is loaded"

X=T.matrix("data")
y=T.ivector("label")
idx=T.lscalar()

images=X.reshape((batch_size, 1, 32, 32))

layer_0=ReLUConvLayer(filter_size=(7,7),
                      num_filters=50,
                      num_channels=1,
                      fm_size=(32,32),
                      batch_size=batch_size)
                      
extract_train=theano.function(inputs=[idx],
			                  outputs=layer_0.apply(images),
			                  givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

extract_test=theano.function(inputs=[idx],
	                         outputs=layer_0.apply(images),
	                         givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size]})

print "\nStarting training data feature extraction\n"
features=np.asarray([])  
start=str(0)    

for batch in xrange(0,n_train_batches):
	batch_img=extract_train(batch)	# (100, 50, 26, 26)
	for i in xrange(0, batch_size):
		temp=batch_img[i].flatten()
		if not features.size:
			features=temp
		else:
			features=np.vstack((features, temp))
	print "Batch ", batch+1, "\t\t\t\t features size: ", features.shape
	if (batch+1)%25 == 0:
		pickle.dump(features, open("/home/ubuntu/destin/pickled_cifar/train-"+start+"-"+str(batch+1)+".p", "wb"))
		print "----------- Pickled ", str(batch+1), " -----------"
		start=str(batch+1)
		features=np.asarray([])

print "\n\nTotal time taken for training data : ", (time.time()-t_0)/60, " minutes"
print "\nStarting testing data feature extraction\n"
features=np.asarray([])  
start=str(0)    

for batch in xrange(0,n_test_batches):
	batch_img=extract_test(batch)	# (100, 50, 26, 26)
	for i in xrange(0, batch_size):
		temp=batch_img[i].flatten()
		if not features.size:
			features=temp
		else:
			features=np.vstack((features, temp))
	print "Batch ", batch+1, "\t\t\t\t features size: ", features.shape
	if (batch+1)%25 == 0:
		pickle.dump(features, open("/home/ubuntu/destin/pickled_cifar/test-"+start+"-"+str(batch+1)+".p", "wb"))
		print "----------- Pickled ", str(batch+1), " -----------"
		start=str(batch+1)
		features=np.asarray([])  

del features			

print "\n\nTotal time taken for testing data : ", (time.time()-t_0)/60, " minutes"


#####################################################
## combining the individual pickle files
#####################################################

train_names=np.arange(0,476,25)
test_names=np.arange(0,76,25)

cifar_dir="/home/ubuntu/destin/pickled_cifar/"

data_train=np.asarray([])
for num in train_names:
	temp=pickle.load(open(cifar_dir+"train-"+str(num)+"-"+str(num+25)+".p", "rb"))
	if not data_train.size:
		data_train=temp
	else:
		data_train=np.vstack((data_train, temp))
	print "Stacked till : ", str(num+25)

pickle.dump(data_train, open(cifar_dir+"data_train.p","wb"))
print "Training data completed. Shape is: ", data_train.shape	# (50000, 33800)
del data_train

data_test=np.asarray([])
for num in test_names:
	temp=pickle.load(open(cifar_dir+"test-"+str(num)+"-"+str(num+25)+".p", "rb"))
	if not data_test.size:
		data_test=temp
	else:
		data_test=np.vstack((data_test, temp))
	print "Stacked till : ", str(num+25)

pickle.dump(data_test, open(cifar_dir+"data_test.p","wb"))
print "Testing data completed. Shape is: ", data_test.shape	# (50000, 33800)
del data_test