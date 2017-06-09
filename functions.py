import numpy as np
import cv2
import os 
import sklearn
import math
from scipy.stats import multivariate_normal
from sklearn.mixture import gaussian_mixture  as gauss
from sklearn.mixture.gmm import log_multivariate_normal_density
from sklearn.utils.extmath import logsumexp
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

def file_counter(paths,extension,folder="",remove=False,loader=False):
	counter=0
	load=[]
	for directory in paths:
		files=os.listdir("./"+folder+"/"+directory)
		for file in files :
			if file.endswith(extension):
				counter=counter+1
				if(loader):
					matrice=np.load("./"+folder+"/"+directory+"/"+file)
					load.append(matrice)
				if(remove):
					os.remove("./"+folder+"/"+directory+"/"+file)
					print("removing file")
	if(loader):
		return load
	return counter

	
##############################
#FISHER VECTOR IMPLEMENTATION#
##############################
#Author: Jacob Gildenblat, 2014 modified by Guichard Laurent
#License: you may use this for whatever you like 
def likelihood_moment(x, ytk, moment):    

    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.ones(x.shape[0]).reshape(x.shape[0], 1)
    return x_moment * ytk.reshape(ytk.shape[0], 1)
	
def likelihood_statistics(samples, means, covs, weights):
	ss0 = []
	ss1 = []
	ss2 = []
	"""log_multivariate_normal_density is a deprecated function that is only in sklearn 0.18 and will be removed afterwards"""
	lpr = (log_multivariate_normal_density(samples, means, covs,"full") + np.log(weights))
	logprob = logsumexp(lpr, axis=1)
	probabilities = (np.exp(lpr - logprob[:, np.newaxis])).T
	
	for k in range(0, len(weights)):

		lm = likelihood_moment(samples, probabilities[k], 0)
		ss0.append(np.sum(lm, axis=0))
		lm = likelihood_moment(samples, probabilities[k], 1)
		ss1.append(np.sum(lm, axis=0))
		lm = likelihood_moment(samples, probabilities[k], 2)
		ss2.append(np.sum(lm, axis=0))
		
	ss0 = np.asarray(ss0)
	ss1 = np.asarray(ss1)
	ss2 = np.asarray(ss2)

	return np.reshape(ss0, (ss0.shape[0], 1)), ss1, ss2	

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):       
    return (s0 - T * w) / np.sqrt(w)       

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):    
    return (s1 - means * s0) / (np.sqrt(np.multiply(sigma, w)))

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return (s2 - 2 * means * s1 + (means * means - sigma) * s0) / (np.sqrt(2*w)*sigma)  
	
def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return (v / np.sqrt(np.dot(v, v)))	
	
def fisher_vector(samples, means, covs, w):    
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)    
	T = samples.shape[0]
	covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
	s0 = np.reshape(s0, (s0.shape[0], 1))    
	w = np.reshape(w, (w.shape[0], 1))
	
	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)        
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)  

	fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
	fv = normalize(fv)    
	return fv    
	
	
def generate_fisher_vectors(paths,means,covs,w,comp):
	global gg
	for directory in paths:
		files=os.listdir("./reduced_data/"+directory)
		for file in files:
			file_name=file.split("_")[0]+comp
			sample=np.load("./reduced_data/"+directory+"/"+file)
			gg=None
			fv=fisher_vector(sample,means,covs,w)
			np.save("./fisher_vectors/"+directory+"/fisher_vector_"+file_name,fv)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		
		
		
		
		
		
		
		