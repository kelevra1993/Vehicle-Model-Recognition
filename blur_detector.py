import os 
import imutils
import cv2

'''script used to check how many images we are working with'''

#threshold under which we consider an image to be blurry
threshold=600

def variance_of_laplacian(image):
	#this function compute the laplacian of an image and returns it's focus
	#the measure is simply the variancce of the laplacian
	return cv2.Laplacian(image,cv2.CV_64F).var()


directories=os.listdir(".")
file_counter=0
for directory in directories :
	if(not(directory.endswith(".py"))):
		image_paths=os.listdir("./"+directory)
		for path in image_paths:
			file_counter+=1
			image=cv2.imread("./"+directory+"/"+path)
			image=cv2.resize(image,(512,384))
			gray=cv2.imread("./"+directory+"/"+path,0)
			fm=variance_of_laplacian(gray)
			text="Not Blurry"
			color=(0,255,0)
			
			print(path)
			if(fm<threshold):
				text="Blurry"
				color=(0,0,255)
			cv2.putText(image,"%s : %lf"%(text,fm),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
			cv2.imshow("Image",image)
			cv2.waitKey(0)
			if(file_counter==10):
				break
	