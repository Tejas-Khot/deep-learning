import cPickle as pickle 
import numpy as np 

cifar_dir="/home/tejas/Documents/pickled_cifar"

d_150=pickle.load(open(cifar_dir+"/200-225", "rb"))
d_150_200=pickle.load(open(cifar_dir+"/225-250", "rb"))
data=np.vstack((d_150, d_150_200))
print data.shape
print type(data)