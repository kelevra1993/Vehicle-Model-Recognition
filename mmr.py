import numpy as np
from sklearn.mixture import gaussian_mixture  as gauss
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import functions as fun
import time


'''This script is the main script of the make and model recognition using unsupervised learning
All of the functions used are in functions.py file
'''
#Number of SIFT components that we will be keeping after PCA reduction, original number of components is 128
pc_comp=120

#Booleans that track of the fisher vector pipeline
compute_all_steps=True

#First we define the PCA REDUCTOR VECTOR
paths=["aston","bmw","clio","dodge","peugot"]

#Name of the file that stores the reducer matrix that can be used for the PCA reduction process
id="reducer"
'''
"""this is going to be removed it used to track what was going on"""
num_images=fun.file_counter(paths,".png")

#Check to see if there is a reducer file, if not create one 
if(not(os.path.isfile(id+".npy")) or compute_all_steps):
	print("No reducer file was found")
	print("A new reducer file is being generated...")
	fun.compute_save_reduce_vector(paths,id,pc_comp=pc_comp)
	print("The Reducer file has been generated")
	print("\n")

#Once the reducer file has been created it is time to load it and use it for PCA Reduction 
print("Loading reducer file...")
reducer=np.load(id+".npy")
print("Reducer file loaded")
print("\n")


#Creation and storage of Reduced ROOT SIFT VECTORS

"""will be removed """
#first we check to see if reduced files are already in the the reduced data file
#we do that by counting the number of files in the system
num_npy=fun.file_counter(paths,".npy","reduced_data")

if(num_npy!=num_images or compute_all_steps):
	print("No root sift files were found")
	print("Generating root sift files...")
	fun.compute_save_reduced_root_sift(reducer,paths)
	print("Reduced root sift files generated and saved")
	print("\n")
	
#Load all of the saved ROOT SIFT DESCRIPTORS and then use them to fit a GMM model

"""this is a previous implementation case that we do not consider anymore """
# start=time.time()
# descriptors=fun.compute_save_reduce_vector(paths,id,pc_comp=pc_comp,reduced=True).T
# end=time.time()
# print(end-start ," seconds")
# print("the shape of the descriptors is ",descriptors.shape)
# second_descriptors=np.atleast_2d(fun.file_counter(paths,".npy","reduced_data",remove=False,loader=True))

"""Implementation that has to kbe kept """
descriptors=np.atleast_2d(np.asarray(fun.file_counter(paths,".npy","reduced_data",remove=False,loader=True)))
print("the shape of the descriptors using the second function is ", descriptors.shape)

#Check to see if there are any trained GMM models
#If so load them and use them to create a fisher vector 
#We will be using a range 
for gmm_comp in range(1000,4000,200):
	gmm_means_file="./GMM/means"+str(gmm_comp)+".gmm.npy"
	gmm_covariance_file="./GMM/covs"+str(gmm_comp)+".gmm.npy"
	gmm_weight_file="./GMM/weights"+str(gmm_comp)+".gmm.npy"
	if(os.path.isfile(gmm_means_file) and os.path.isfile(gmm_covariance_file) and os.path.isfile(gmm_weight_file)):
		print("all the GMM files are in place and now we are going to load them")
		print("loading files...")
		gmm_means_file="./GMM/means"+str(gmm_comp)+".gmm.npy"
		gmm_covariance_file="./GMM/covs"+str(gmm_comp)+".gmm.npy"
		gmm_weight_file="./GMM/weights"+str(gmm_comp)+".gmm.npy"
		means=np.load(gmm_means_file)
		covs=np.load(gmm_covariance_file)
		weights=np.load(gmm_weight_file)
		print("GMM "+str(gmm_comp)+" loaded")
		print("\n")
	else: 
		# print("we did not find all of our files")
		# print("we train a GMM Model")
		# print("gathering ROOT SIFT descriptors...")
		descriptors=fun.compute_save_reduce_vector(paths,id,pc_comp=pc_comp,reduced=True).T
		# print("descriptors gathered")
		print("training GMM %d..."%(gmm_comp))
		GMM=gauss.GaussianMixture(n_components=gmm_comp,covariance_type="full",max_iter=10000,n_init=1,init_params="kmeans")
		GMM.fit(descriptors)
		# print(np.sum(GMM.predict_proba(descriptors[0:20]),axis=1))
		print("trained GMM %d..."%(gmm_comp))
		# print("saving the GMM model")
		means=GMM.means_
		covs=GMM.covariances_
		weights=GMM.weights_
		gmm_means_file="./GMM/means"+str(gmm_comp)+".gmm.npy"
		gmm_covariance_file="./GMM/covs"+str(gmm_comp)+".gmm.npy"
		gmm_weight_file="./GMM/weights"+str(gmm_comp)+".gmm.npy"
		
		np.save(gmm_means_file,means)
		np.save(gmm_covariance_file,covs)
		np.save(gmm_weight_file,weights)
		# print("GMM model has been saved")
		print("\n")

	# now we check to see if there is any fisher vector
	num_fis=fun.file_counter(paths,".npy","fisher_vectors",remove=False)
	if(compute_all_steps):
		# print("No fisher vector files were found")
		# print("generating them...")
		print("Generate and Save fisher files for GMM %d..."%(gmm_comp))
		fun.generate_fisher_vectors(paths,means,covs,weights,"_"+str(gmm_comp))
		print("Fisher files saved")
		print("\n")
		# print("loading our fisher files...")
		# fisher_vectors=np.atleast_2d(fun.file_counter(paths,".npy","fisher_vectors",remove=False,loader=True))
		# print("fisher files have been generated")
		# print(fisher_vectors.shape)
	else:
		print("we found our fisher files")
		print("loading our fisher files...")
		# fisher_vectors=np.atleast_2d(fun.file_counter(paths,".npy","fisher_vectors",remove=False,loader=True))
		# print(fisher_vectors.shape)'''
		
	
"""
	norm=np.linalg.norm(fisher_vectors[1])
	cosine_metric=cosine_similarity(fisher_vectors)
	print(cosine_metric.shape)
	# for i in range(10):
		# print(cosine_metric[i,i])
		# print(cosine_metric[i])
	print("\n")
	"""

"""
	for ind in range(45,55):
		indices=np.flip(np.argsort(cosine_metric[ind]),axis=0)
		print(indices)
		print("\n")
		for sim in range(5):
			if(indices[sim]<50):
				# print("./buildings/%03d.png"%(indices[sim]+1))
				image=cv2.imread("./buildings/%03d.png"%(indices[sim]+1))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
			if(indices[sim]>49):
				image=cv2.imread("./sports/%03d.png"%(indices[sim]+1-50))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
		cv2.waitKey(0)"""
			

"""
	del means
	del covs
	del weights
	# del fisher_vectors
	means=None
	covs=None
	weights=None
	fisher_vectors=None
	"""


	
	
#BIC COMPUTATION 
"""
bicmin = 100000
for n_components in range(2, 10):
	print("Computing GMM for %d" %n_components)
	#Third Step we create a Gaussian Mixture Model and we make it fit the model
	GMM=gauss.GaussianMixture(n_components=n_components,covariance_type="full",max_iter=1000,n_init=1,init_params="kmeans")
	GMM.fit(descriptors)
	bic = GMM.bic(descriptors)
	if bicmin > bic:
		print ("BIC of %d is" %(n_components),bic)
		bicmin = bic
		"""
"""
GMM=gauss.GaussianMixture(n_components=3,covariance_type="full",max_iter=1000,n_init=1,init_params="kmeans")
GMM.fit(descriptors)		
means=GMM.means_
covs=GMM.covariances_
weights=GMM.weights_

np.save("./GMM/means.gmm",means)
np.save("./GMM/covs.gmm",covs)
np.save("./GMM/weights.gmm",weights)

sample=np.load("./reduced_data/sports/001_root_sift.npy")
print("Shape of the sample of this image is : ",sample.shape)
print("Shape of the means of the GMM model is : ",means.shape)
print("Shape of the covariance matrix of the GMM is : ",covs.shape)
print("Shape of the weights matrix of the GMM is : ",weights.shape)

#testing the fisher vector function 
fisher_vector=fun.fisher_vectorNew(sample,means,covs,weights)
print("the shape of the NORMALIZED fisher vector is : ",fisher_vector.shape)"""


######################################
# FINAL STAGE OF PROOF OF CONCEPT    #
######################################
for gmm_comp in range(1000,4000,200):

	print("loading our fisher files...")
	print("-------------------------------------------------")
	fisher_vectors=np.atleast_2d(fun.file_counter(paths,str(gmm_comp)+".npy","fisher_vectors",remove=False,loader=True,Fisher=True))
	# print(fisher_vectors.shape)
	
	cosine_metric=cosine_similarity(fisher_vectors)
	# print(cosine_metric.shape)
	# for i in range(10):
		# print(cosine_metric[i,i])
		# print(cosine_metric[i])
	all_bmw=0		
	all_clio=0			
	all_dodge=0			
	all_peugot=0				
	all_aston=0		
		
	for ind in range(5,10):
		indices=np.flip(np.argsort(cosine_metric[ind]),axis=0)
		# print(indices)
		aston=0
		bmw=0
		dodge=0
		clio=0
		peugot=0
		for  sim in range(1,20):
		
			if(indices[sim]<20):
				aston=aston+1
			if(indices[sim]>19 and indices[sim]<40):
				bmw=bmw+1
			if(indices[sim]>39 and indices[sim]<60):
				clio=clio+1
			if(indices[sim]>59 and indices[sim]<80):
				dodge=dodge+1
			if(indices[sim]>79):
				peugot=peugot+1
		print("there are %d ASTON vehicles in the first 20 images"%aston)		
		print("there are %d BMW vehicles in the first 20 images"%bmw)		
		print("there are %d CLIO vehicles in the first 20 images"%clio)		
		print("there are %d DODGE vehicles in the first 20 images"%dodge)		
		print("there are %d PEUGOT vehicles in the first 20 images"%peugot)		
		print("\n")
		"""for sim in range(5):
		
			if(indices[sim]<20):
				# print("./buildings/%03d.png"%(indices[sim]+1))
				if (indices[sim]==0):
					image=cv2.imread("./aston/%03d.png"%(indices[sim]+1))
				else:
					image=cv2.imread("./aston/%03d.png"%(indices[sim]))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
					
					
			if(indices[sim]>19 and indices[sim]<40):
				# print("./buildings/%03d.png"%(indices[sim]+1))
				image=cv2.imread("./bmw/%03d.png"%(indices[sim]+1-20))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
					
					
			if(indices[sim]>39 and indices[sim]<60):
				# print("./buildings/%03d.png"%(indices[sim]+1))
				image=cv2.imread("./clio/%03d.png"%(indices[sim]+1-40))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
					
					
					
			if(indices[sim]>59 and indices[sim]<80):
				# print("./buildings/%03d.png"%(indices[sim]+1))
				image=cv2.imread("./dodge/%03d.png"%(indices[sim]+1-60))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)
					
					
			if(indices[sim]>79):
				image=cv2.imread("./peugot/%03d.png"%(indices[sim]+1-80))
				height, width = image.shape[:2]
				image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				if(sim==0):
					cv2.imshow("original",image)
				else:
					cv2.imshow("similar %d"%(sim),image)"""
					
		all_bmw+=bmw			
		all_clio+=clio			
		all_dodge+=dodge			
		all_peugot+=peugot			
		all_aston+=aston	
		# cv2.waitKey(0)
	print("Overall GMM with %d components gives us : "%gmm_comp)
	print("BMW: %d"%all_bmw)
	print("CLIO: %d"%all_clio)
	print("DODGE: %d"%all_dodge)
	print("PEUGOT: %d"%all_peugot)
	print("ASTON: %d"%all_aston)
	cv2.waitKey(0)
	
	
	
	print("-------------------------------------------------")
	# break






























