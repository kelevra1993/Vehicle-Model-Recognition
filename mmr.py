import numpy as np
from sklearn.mixture import gaussian_mixture  as gauss
import cv2
import os
import functions as fun


'''This script is the main script of the make and model recognition using unsupervised learning'''
print("\n")

#First we define the PCA REDUCTOR VECTOR
paths=["buildings","sports"]
id="reducer"
num_images=fun.file_counter(paths,".png")
print("we have detected %d images"%(num_images))
#If no reducer file, create one 
if(not(os.path.isfile(id+".npy"))):
	print("No reducer file was found")
	print("A new reducer file is being generated...")
	fun.compute_save_reduce_vector(paths,id,pc_comp=100)
	print("The Reducer file has been generated")
	print("\n")
#Load it
print("Loading reducer file...")
reducer=np.load(id+".npy")
print("Reducer file loaded")
print("\n")


#Second we create and store the reduced rootsift vectors
#first we check to see if reduced files are already in the the reduced data file
#we do that by counting the number of files in the system
num_npy=fun.file_counter(paths,".npy","reduced_data/")
print("We have detected %d root_sift files"%(num_npy))
if(num_npy!=num_images):
	print("no root sift files were found")
	print("generating root sift files...")
	fun.compute_save_reduced_root_sift(reducer,paths)
	print("root sift files generated")

#Third Step we create a Gaussian Mixture Model and we make it fit the model
GMM=gauss.GaussianMixture(n_components=2,covariance_type="full",max_iter=100,n_init=1,init_params="kmeans")

print(dir(GMM))

#Now we get all of the sift descriptors of all of our images and we make it fit our model.
#we transpose it to make it readable to for the gmm fitting function
descriptors=fun.compute_save_reduce_vector(paths,id,pc_comp=100,reduced=True).T
print(descriptors.shape)
GMM.fit(descriptors)
#So now that our GMM model has fitted our data time to create our fisher vector

labels=GMM.predict(descriptors[0:20])
labels_prob=GMM.predict_proba(descriptors[0:20])
print(labels.shape)
print(labels)
# print(np.mean(labels))
print(labels_prob)
print("The shape of GMM weights is : ",(np.atleast_2d(GMM.weights_).shape))
print("The shape of GMM means is : ",GMM.means_.shape)
print("The shape of GMM covariance 0 is : ",GMM.covariances_[0].shape)
print(GMM.covariances_[1].shape)
#Calculation of the first element of deviation vector
# first=(1/(20*np.sqrt(GMM.weights_[0])))*np.dot((GMM.predict_proba(descriptors[0:20]).T)[0],((descriptors.T)[0]-GMM.means_[0][0]*(np.ones(20))))
# first=np.dot((GMM.predict_proba(descriptors[0:20]).T)[0],((descriptors.T)[0]-GMM.means_[0][0]*(np.ones(20))))
# print((GMM.predict_proba(descriptors[0:20]).T)[0].shape)
print(((descriptors.T)[0]-GMM.means_[0][0]*(np.ones(20))).shape)
print(((descriptors.T)[0]-GMM.means_[0][0]*(np.ones(20))).shape)
# image=cv2.imread("niss.jpg",0)
# rs=fun.compute_save_root_sift(image)
# print(rs.shape)
# print(a.shape)
































