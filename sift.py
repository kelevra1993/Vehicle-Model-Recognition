import numpy as np
import rootsift as root
import cv2
import os

'''This script initially extracts all sift features from a certain number of images, stacks them and feeds them to a PCA reduction function
   that gives out a vector for data reducing'''


######################################
#Definition of PCA reduction function#
######################################

def _compute_and_reduce_components_(data,num_comp_keep=0):
	#We assume that data was fed with the rows as observations
	#Columns as features that we will want to reduce

	#First we subtract the mean of coulums data columns to all columns
	print(data.shape)
	centered_data=(data-np.mean(data.T,axis=1)).T
	#We get eigenvector matrix and eigenvalues of covariance matrix
	[values,vectors]=np.linalg.eig(np.cov(centered_data))
	#Total number of components
	num_comp=np.size(vectors,axis=1)
	# print(num_comp)
	#Sorting eigenvalues in ascending order and eigenvectors accordingly
	sorted=np.argsort(values)
	sorted=sorted[::-1]
	vectors = vectors[:,sorted]
	values = values[sorted]
	# print(values.astype(int))
	# Removal of some components, those that we deem "unprincipal"

	if num_comp_keep < num_comp and num_comp_keep > 0:
		vectors = vectors[:,range(num_comp_keep)]
	# Data Projection in new reduced space
	# reduced_data = np.dot(vectors.T,centered_data).T
	# print(reduced_data.shape)
	# reduced_data = np.dot(vectors.T,centered_data)
	# reduced_data=np.dot(vectors,reduced_data).T+np.mean(data,axis=0)
	# print(reduced_data.shape)
	# print("\n")
	# print(vectors.shape)
	# print(centered_data.shape)

	# return(reduced_data)
	print(vectors.shape)
	return(vectors)
#First we create the rootsift class
rs=root.RootSIFT()

directory="./buildings"
files=os.listdir(directory)
sift=[]
for file in files : 
	if file.endswith(".png"):
		#extract RootSIFT descriptors
		gray=cv2.imread(directory+"/"+file)
		detector=cv2.xfeatures2d.SIFT_create()
		(kps, desc)=detector.detectAndCompute(gray,None)
		(kps,descs)=rs.compute(gray,kps)
		# print ("ROOTSIFT : kps=%d, descriptors=%s" %(len(kps),descs.shape))
		rows=descs.shape[0]
		for i in range(rows):
			sift.append(descs[i])
	
directory="./sports"
files=os.listdir(directory)
for file in files : 
	if file.endswith(".png"):
		# extract RootSIFT descriptors
		gray=cv2.imread(directory+"/"+file)
		detector=cv2.xfeatures2d.SIFT_create()
		(kps, desc)=detector.detectAndCompute(gray,None)
		(kps,descs)=rs.compute(gray,kps)
		# print ("ROOTSIFT : kps=%d, descriptors=%s" %(len(kps),descs.shape))
		rows=descs.shape[0]
		for i in range(rows):
			sift.append(descs[i])
	
sift=np.asarray(sift)
print("the shape of sifts are : ",sift.shape)
pca_reductor=_compute_and_reduce_components_(sift,80)
np.save("reductor",pca_reductor)
















