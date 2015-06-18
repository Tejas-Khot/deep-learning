import cPickle as pickle 
import numpy as np 

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