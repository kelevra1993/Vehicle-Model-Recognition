import numpy as np
from sklearn.mixture import gaussian_mixture  as gauss
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import functions as fun


'''This script is the main script of the make and model recognition using unsupervised learning'''
pc_comp=100
problem=True

#First we define the PCA REDUCTOR VECTOR
# paths=["buildings","sports"]
# model_paths=["aston","bmw","clio","dodge","peugot"]
paths=["aston","bmw","clio","dodge","peugot"]
id="reducer"
num_images=fun.file_counter(paths,".png")
#If no reducer file, create one 
"""
if(not(os.path.isfile(id+".npy")) or problem):
	print("No reducer file was found")
	print("A new reducer file is being generated...")
	fun.compute_save_reduce_vector(paths,id,pc_comp=pc_comp)
	print("The Reducer file has been generated")
	print("\n")
	problem=True
#Load it
print("Loading reducer file...")
reducer=np.load(id+".npy")
print("Reducer file loaded")
print("\n")


#Second we create and store the reduced rootsift vectors
#first we check to see if reduced files are already in the the reduced data file
#we do that by counting the number of files in the system
num_npy=fun.file_counter(paths,".npy","reduced_data")
if(num_npy!=num_images or problem):
	print("no root sift files were found")
	print("generating root sift files...")
	fun.compute_save_reduced_root_sift(reducer,paths)
	print("root sift files generated")
print("\n")
#Now we get all of the sift descriptors of all of our images and we make it fit our model.
#we transpose it to make it readable to for the gmm fitting function
# descriptors=fun.compute_save_reduce_vector(paths,id,pc_comp=100,reduced=True).T
# print("the shape of the descriptors is : ",descriptors.shape)
# print("\n")

#Once we have got all of the descriptors the first thing we want to do is to check that there is a trained GMM
#if there is a trained GMM we load it and if not we train a GMM using our descriptors
"""

gmm_means_file="./GMM/means.gmm.npy"
gmm_covariance_file="./GMM/covs.gmm.npy"
gmm_weight_file="./GMM/weights.gmm.npy"
# gmm_comp=1000
for gmm_comp in range(1000,4000,100):
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
	if(num_fis!=num_images or problem):
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
		# print(fisher_vectors.shape)
	
	"""
	norm=np.linalg.norm(fisher_vectors[1])
	cosine_metric=cosine_similarity(fisher_vectors)
	print(cosine_metric.shape)
	# for i in range(10):
		# print(cosine_metric[i,i])
		# print(cosine_metric[i])
	print("\n")"""

	"""for ind in range(45,55):
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
			

	del means
	del covs
	del weights
	# del fisher_vectors
	means=None
	covs=None
	weights=None
	fisher_vectors=None


	
	
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
		bicmin = bic"""
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

































