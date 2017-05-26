import numpy as np
import cv2
import os 
import sklearn

'''Functions for Make and Model Recognition'''
#################
#ROOT SIFT CLASS#
#################
class RootSIFT:
    def __init__(self):
        #Initialize the SIFT feature extractor
        self.extractor=cv2.xfeatures2d.SIFT_create()
        
    def compute(self, image, descs, eps=1e-7):
        #Applying Hellinger kernel by first L1 Normalizing and taking the square root
        descs /= (descs.sum(axis=1, keepdims=True) +eps)
        descs=np.sqrt(descs)
        return(descs)

######################################
#Definition of PCA reduction function#
######################################

def _compute_and_reduce_components_(data,num_comp_keep=0,reduced=False):
	#We assume that data was fed with the rows as observations
	#Columns as features that we will want to reduce
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
	# Removal of some components, those that we deem "unprincipal"
	if num_comp_keep < num_comp and num_comp_keep > 0:
		vectors = vectors[:,range(num_comp_keep)]
	# Data Projection in new reduced space
	if(reduced):
		reduced_data = np.dot(vectors.T,centered_data)
		return reduced_data
	#Data projection in initial space
	# reduced_data=np.dot(vectors,reduced_data).T+np.mean(data,axis=0)
	# print("\n")
	# print(centered_data.shape)
	return(vectors)		

def compute_save_reduce_vector(paths,id,pc_comp,reduced=False):
	rs=RootSIFT()
	sift=[]
	for directory in paths:
		files=os.listdir("./"+directory)	
		for file in files : 
			if file.endswith(".png"):
				#extract RootSIFT descriptors
				gray=cv2.imread(directory+"/"+file)
				detector=cv2.xfeatures2d.SIFT_create()
				(kps, desc)=detector.detectAndCompute(gray,None)
				root_desc=rs.compute(gray,desc)
				rows=root_desc.shape[0]
				for i in range(rows):
					sift.append(root_desc[i])
	sift=np.asarray(sift)
	if(reduced):
		return _compute_and_reduce_components_(sift,pc_comp,reduced=True)
	pca_reductor=_compute_and_reduce_components_(sift,pc_comp)
	np.save(id,pca_reductor)

####################################
#COMPUTE AND SAVE REDUCED ROOTSIFTS#
####################################

def compute_save_reduced_root_sift(reducer,paths):
	for directory in paths:
		files=os.listdir("./"+directory)
		for file in files : 
			if file.endswith(".png"):
				rs=RootSIFT()
				image_path=directory+"/"+file
				image=cv2.imread(image_path)
				detector=cv2.xfeatures2d.SIFT_create()
				(kps, desc)=detector.detectAndCompute(image,None)
				root_desc=rs.compute(image,desc)
				root_sift=np.asarray(root_desc)
				reduced_root_sift = np.dot(reducer.T,root_sift.T).T
				root_sift_path="./reduced_data/"+image_path.split(".")[0]+"_root_sift"
				np.save(root_sift_path,reduced_root_sift)	
				# print("reduced_root_sift saved")
				# print("---------------------------")

def file_counter(paths,extension,folder="",remove=False,loader=False):
	counter=0
	for directory in paths:
		files=os.listdir("./"+folder+directory)
		for file in files :
			if file.endswith(extension):
				counter=counter+1
				if(loader):
					matrice=np.load("./"+folder+directory)
					print(matrice.shape)
	return counter
	
def create_fisher_vector():
	return 1

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		
		
		
		
		
		
		
		