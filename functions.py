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
        # self.extractor=cv2.xfeatures2d.SIFT_create()
		self.extractor = cv2.DescriptorExtractor_create("SIFT")
        
    def compute(self, image, descs, eps=1e-7):
        #Applying Hellinger kernel by first L1 Normalizing and taking the square root
        descs /= (descs.sum(axis=1, keepdims=True) +eps)
        descs=np.sqrt(descs)
        return(descs)

######################################
#Definition of PCA reduction function#
######################################
#Function that gives the reducer matrice
def _compute_and_reduce_components_(data,num_comp_keep=0,reduced=False):
	#We assume that data was fed with the rows as observations
	#Columns as features that we will want to reduce
	centered_data=(data-np.mean(data.T,axis=1)).T
	#We get eigenvector matrix and eigenvalues of covariance matrix
	[values,vectors]=np.linalg.eig(np.cov(centered_data))
	#Total number of components
	num_comp=np.size(vectors,axis=1)
	#Sorting eigenvalues in ascending order and eigenvectors accordingly
	sorted=np.argsort(values)
	sorted=sorted[::-1]
	vectors = vectors[:,sorted]
	values = values[sorted]
	# Removal of some components, those that we deem "unprincipal"
	if num_comp_keep < num_comp and num_comp_keep > 0:
		vectors = vectors[:,range(num_comp_keep)]
	return(vectors)	
	
#function that gets all sift descriptors from all of the images 
#then creates a reducer matrice used later on for PCA reduction
def compute_save_reduce_vector(paths,id,pc_comp):
	rs=RootSIFT()
	sift=[]
	for directory in paths:
		print("opening %s to create the PCA reduction vector..."%(directory))
		files=os.listdir("./"+directory)	
		for file in files : 
			if file.endswith(".png"):
				#extract RootSIFT descriptors
				#here to gain time we could save the original root sift files in a folder, but not implemented
				gray=cv2.imread(directory+"/"+file,0)
				
				'''python3 implementation'''
				# detector=cv2.xfeatures2d.SIFT_create()
				# (kps, desc)=detector.detectAndCompute(image,None)
				'''end of python3 implementation'''
				
				'''opencv2.4.13 implementation'''
				detector = cv2.FeatureDetector_create("SIFT")
				kps=detector.detect(gray)
				extractor=cv2.DescriptorExtractor_create("SIFT")
				(kps, desc)=extractor.compute(gray,kps)
				'''end of iimplementation'''
				
				root_desc=rs.compute(gray,desc)
				rows=root_desc.shape[0]
				for i in range(rows):
					sift.append(root_desc[i])
	sift=np.asarray(sift)
	pca_reductor=_compute_and_reduce_components_(sift,pc_comp)
	np.save(id,pca_reductor)

####################################
#COMPUTE AND SAVE REDUCED ROOTSIFTS#
####################################  
#this function uses the reducer file that was saved to do PCA Reduction, then save these ROOT SIFT VECTORS for one particular file
def compute_save_reduced_root_sift(reducer,paths):
	for directory in paths:
		files=os.listdir("./"+directory)
		for file in files : 
			if file.endswith(".png"):
				#we could have saved original rootsift files and then loaded them from here, but not implemented
				rs=RootSIFT()
				image_path=directory+"/"+file
				image=cv2.imread(image_path,0)
				
				'''python3 implementation'''
				# detector=cv2.xfeatures2d.SIFT_create()
				# (kps, desc)=detector.detectAndCompute(image,None)
				'''end of python3 implementation'''
				
				'''opencv2.4.13 implementation'''
				detector = cv2.FeatureDetector_create("SIFT")
				kps=detector.detect(image)
				extractor=cv2.DescriptorExtractor_create("SIFT")
				(kps, desc)=extractor.compute(image,kps)
				'''end of iimplementation'''
				
				root_desc=rs.compute(image,desc)
				root_sift=np.asarray(root_desc)
				reduced_root_sift = np.dot(reducer.T,root_sift.T).T
				root_sift_path="./reduced_data/"+image_path.split(".")[0]+"_root_sift"
				np.save(root_sift_path,reduced_root_sift)	
#simple function for file management, uses to load files and remove them
def file_counter(paths,extension,folder="",remove=False,loader=False,Fisher=False):
	counter=0
	load=[]
	for directory in paths:
		files=os.listdir("./"+folder+"/"+directory)
		for file in files :
			if file.endswith(extension):
				counter=counter+1
				if(loader):
					matrice=np.load("./"+folder+"/"+directory+"/"+file)
					if(Fisher):
						#taking care of particular case when we deal with fisher vector
						load.append(matrice)
					else:	
						row=(matrice.shape)[0]
						for r in range(row) :
							load.append(matrice[r])
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
	"""to get rid of error message go to your sklearn package in python or python27 /Lib/sites-packages/sklearn/mixture/gmm.py
	and comment that deprecated line, but keep in mind that it will be removed"""
	
	lpr = (log_multivariate_normal_density(samples, means, covs,"diag") + np.log(weights))
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
	#CASE WHERE WE HAVE A FULL COVARIANCE FOR GMM, JUST UNCOMMENT THE FOLLOWING LINE
	#and change log_multivariate_normal_density cov from "diag" to "full" in the likelihood_statistics function
	# covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
	# CASE WHERE WE HAVE A DIAGONAL COVARIANCE FOR FISHER VECTOR
	#we do nothing
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		
		
		
		
		
		
		
		