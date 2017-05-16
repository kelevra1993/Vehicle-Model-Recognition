import numpy as np
import rootsift as root
import cv2

#Get the image 
image = cv2.imread('image.jpg')
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Detect DoG
detector=cv2.xfeatures2d.SIFT_create()

#extract normal SIFT descriptors
(kps, desc)=detector.detectAndCompute(gray,None)

#extract RootSIFT descriptors
#First we create the rootsift class
rs=root.RootSIFT()
(kps,descs)=rs.compute(gray,kps)
# print ("ROOTSIFT : kps=%d, descriptors=%s" %(len(kps),descs.shape))
img=cv2.drawKeypoints(image,kps,gray)
cv2.imwrite('image_root_sift.jpg',img)

######################################
#Definition of PCA reduction function#
######################################

def _compute_and_reduce_components_(data,num_comp_keep=0):
    #We assume that data was fed with the rows as observations
    #Columns as features that we will want to reduce
    
    #First we subtract the mean of coulums data columns to all columns
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
    reduced_data = np.dot(vectors.T,centered_data)
    reduced_data=np.dot(vectors,reduced_data).T+np.mean(data,axis=0)
    # print(vectors.shape)
    # print(centered_data.shape)

    return(reduced_data)
    
    
    
    
# image = cv2.imread('aston.jpg',0)   
# image = cv2.imread('aston.jpg') 
# print(image3d.shape)  
# print(image3d[:,:,1].shape)
# componenents = image.shape[1]
# for keep_components in range(1,componenents,50):
    # width=image.shape[0]
    # height=image.shape[1]
    # buffer=np.zeros((width,height,3))
    # for i in range(0,3):
        # reduced_data=_compute_and_reduce_components_(image[:,:,i],keep_components)
        # buffer[:,:,i]=reduced_data.astype(int)/255
    # cv2.imshow(str(keep_components),buffer)

# cv2.waitKey(0)  .

print(descs)
print("\n"*5)
reduced_sift=_compute_and_reduce_components_(descs,num_comp_keep=127)
print(reduced_sift)
print("\n"*5)
reduced_sift=_compute_and_reduce_components_(descs,num_comp_keep=100)
print(reduced_sift)
print("\n"*5)
reduced_sift=_compute_and_reduce_components_(descs,num_comp_keep=50)
print(reduced_sift)
print("\n"*5)
reduced_sift=_compute_and_reduce_components_(descs,num_comp_keep=1)
print(reduced_sift)
print("\n"*5)
















