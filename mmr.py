import numpy as np
import sklearn
import cv2
import os
import functions as fun

'''This script is the main script of the make and model recognition using unsupervised learning'''
print("\n")

#First we define the PCA REDUCTOR VECTOR
paths=["buildings","sports"]
id="reducer"
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

#Second we create and store the reduced rootsift vectors
fun.compute_save_reduced_root_sift(reducer,paths)

# image=cv2.imread("niss.jpg",0)
# rs=fun.compute_save_root_sift(image)
# print(rs.shape)
# print(a.shape)