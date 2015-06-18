"""Feature extraction """

import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T
import time
import nnet.datasets as ds
from nnet.convnet import ReLUConvLayer
from nnet.whitening import ZCA

t_0=time.time()
batch_size=100
num_features=1600

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("/home/ubuntu/destin/cifar-10-batches-py")
del Ytr, Yte
# pre-processing the data

#flatten across channels
Xtr=np.mean(Xtr, 3)	#shape (50000, 32, 32)
Xte=np.mean(Xte, 3) #shape (10000, 32, 32)

flattenedX=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])	#shape(50000, 1024)
flattenedX_=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])	#shape(10000, 1024)
std=np.std(flattenedX, axis=1)
mean=np.mean(flattenedX, axis=1) 	
zca=ZCA()
whiteningComponent=zca.fit(flattenedX) 
whitenedXtr=zca.transform(flattenedX)
whitenedXte=zca.transform(flattenedX_)
del Xtr, Xte, flattenedX, whiteningComponent

whitenedXtr=whitenedXtr.reshape(50000,32,32)
whitenedXte=whitenedXte.reshape(10000,32,32)
#dividing the image into 4 quarters each of shape (8,8)
quarters_train= [ whitenedXtr[:, 0:whitenedXtr.shape[1]/2, 0:whitenedXtr.shape[2]/2],
			     whitenedXtr[:, 0:whitenedXtr.shape[1]/2, whitenedXtr.shape[2]/2:whitenedXtr.shape[2]],
			     whitenedXtr[:, whitenedXtr.shape[1]/2:whitenedXtr.shape[1], 0:whitenedXtr.shape[2]/2],
			     whitenedXtr[:, whitenedXtr.shape[1]/2:whitenedXtr.shape[1], whitenedXtr.shape[2]/2:whitenedXtr.shape[2]] ]

quarters_test=[ whitenedXte[:, 0:whitenedXte.shape[1]/2, 0:whitenedXte.shape[2]/2],
			    whitenedXte[:, 0:whitenedXte.shape[1]/2, whitenedXte.shape[2]/2:whitenedXte.shape[2]],
			    whitenedXte[:, whitenedXte.shape[1]/2:whitenedXte.shape[1], 0:whitenedXte.shape[2]/2],
			    whitenedXte[:, whitenedXte.shape[1]/2:whitenedXte.shape[1], whitenedXte.shape[2]/2:whitenedXte.shape[2]] ]

del whitenedXtr, whitenedXte
#zero-centering across individual features and 
#normalizing the data dimensions so that they are of approximately the same scale
quarters_train=[ X_.reshape(X_.shape[0], X_.shape[1]*X_.shape[2]) for X_ in quarters_train]
quarters_test=[ X_.reshape(X_.shape[0], X_.shape[1]*X_.shape[2]) for X_ in quarters_test]

quarters_train=[ (X-mean[i*X.shape[1]:(i+1)*X.shape[1]])/std[i*X.shape[1]:(i+1)*X.shape[1]] for i in xrange(0,4)]		   
quarters_test=[ (X-mean[i*X.shape[1]:(i+1)*X.shape[1]])/std[i*X.shape[1]:(i+1)*X.shape[1]] for i in xrange(0,4)]

print "[MESSAGE] The data is loaded and pre-processing is over"

X=T.matrix("data")
y=T.ivector("label")
idx=T.lscalar()
idy=T.lscalar()

images=X.reshape((batch_size, 1, 16, 16))

layer_0=ReLUConvLayer(filter_size=(6,6),
                      num_filters=num_features,
                      num_channels=1,
                      fm_size=(16,16),
                      batch_size=batch_size)
                      
extract_train=theano.function(inputs=[idx, idy],
			                  outputs=layer_0.apply(images),
			                  givens={X: quarters_train[idx][idy * batch_size: (idy + 1) * batch_size]})

extract_test=theano.function(inputs=[idx, idy],
	                         outputs=layer_0.apply(images),
	                         givens={X: quarters_test[idx][idy * batch_size: (idy + 1) * batch_size]})

print "Time taken : ", time.time()-t_0
print "\nStarting training data feature extraction\n"


for q in xrange(0,4):
	start_time=time.time()
	print "[QUARTER", q, "]"
	train_set_x=ds.shared_dataset(quarters_train[q])
	test_set_x=ds.shared_dataset(quarters_test[q])

	n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
	n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

	print "n_train_batches : ", n_train_batches
	print "n_test_batches : ", n_test_batches

	features=np.asarray([])  
	start=str(0)    

	for batch in xrange(0,n_train_batches):
		batch_img=extract_train(batch)	# (100, 1600, 21, 21)
		for i in xrange(0, batch_size):
			temp=batch_img[i].flatten()
			if not features.size:
				features=temp
			else:
				features=np.vstack((features, temp))
		print "Batch ", batch+1, "\t\t\t\t features size: ", features.shape
		if (batch+1)%25 == 0:
			pickle.dump(features, open("/home/ubuntu/destin/pickled_cifar/train-"+"Q"+(q+1)+"-"+start+"-"+str(batch+1)+".p", "wb"))
			print "----------- Pickled ", str(batch+1), " -----------"
			start=str(batch+1)
			features=np.asarray([])
	print "\n\nTotal time taken for training data : ", (time.time()-start_time)/60, " minutes"
	
	print "\nStarting testing data feature extraction\n"
	features=np.asarray([])  
	start=str(0)    

	start_time=time.time()
	for batch in xrange(0,n_test_batches):
		batch_img=extract_test(batch)	# (100, 1600, 21, 21)
		for i in xrange(0, batch_size):
			temp=batch_img[i].flatten()
			if not features.size:
				features=temp
			else:
				features=np.vstack((features, temp))
		print "Batch ", batch+1, "\t\t\t\t features size: ", features.shape
		if (batch+1)%25 == 0:
			pickle.dump(features, open("/home/ubuntu/destin/pickled_cifar/test-"+"Q"+(q+1)+"-"+start+"-"+str(batch+1)+".p", "wb"))
			print "----------- Pickled ", str(batch+1), " -----------"
			start=str(batch+1)
			features=np.asarray([])  
	print "\n\nTotal time taken for testing data : ", (time.time()-start_time)/60, " minutes"

del features			

print "[RESULT]	Total time taken for feature extraction is : ", time.time()-t_0

