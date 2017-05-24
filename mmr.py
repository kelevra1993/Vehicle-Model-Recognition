import numpy as np
from sklearn import mixture
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
gmm=GaussianMixture(n_components=1,covariance_type="full",max_iter=20,n_init=1,init_params="kmeans")
print(dir(gmm))
# image=cv2.imread("niss.jpg",0)
# rs=fun.compute_save_root_sift(image)
# print(rs.shape)
# print(a.shape)