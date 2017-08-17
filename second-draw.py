import os 
import cv2
import csv
import numpy as np

root="./original_images"
makes=os.listdir(root)
#First we are going to open the csv file with all the information necessary for segmentation
'''OPEN A NEW CSV FILE FOR READING'''
upper_left=[0,0]
plate_upper_left=[0,0]
middle_upper_left=[0,0]
full_upper_left=[0,0]
lower_right=[0,0]
plate_lower_right=[0,0]
middle_lower_right=[0,0]
full_lower_right=[0,0]
buffer_right=[0,0]
buffer_left=[0,0]


counter=0

with open('final_trainer.csv','r') as csvfile:
	linereader=csv.reader(csvfile,delimiter=";")

	for row in linereader :
		make=row[15]

		#to ignore the next line which is the second image, when there is a second image....
		image_names=row[10].split("_")	
		license_plate=row[13]

		positions=license_plate.split("_")

		firsts=positions[0].split("x")
		seconds=positions[2].split("x")

		delta_x=abs(int(seconds[0])-int(firsts[0]))
		delta_y=abs(int(seconds[1])-int(firsts[1]))

		if((os.path.exists(root+"/"+make+"/"+image_names[0]+"_2.tiff"))):
			image_2=cv2.imread(root+"/"+make+"/"+image_names[0]+"_2.tiff")
			
			upper_left[0]=int(firsts[0])-2*delta_x
			middle_upper_left[0]=int(firsts[0])-delta_x
			full_upper_left[0]=int(firsts[0])-int(2.5*delta_x)
			upper_left[1]=int(firsts[1])-2*delta_y
			middle_upper_left[1]=int(firsts[1])-2*delta_y
			full_upper_left[1]=int(firsts[1])-int(3*delta_y)
			lower_right[0]=int(seconds[0])+2*delta_x
			middle_lower_right[0]=int(seconds[0])+delta_x
			full_lower_right[0]=int(seconds[0])+int(2.5*delta_x)
			lower_right[1]=int(seconds[1])+int(1.5*delta_y)
			middle_lower_right[1]=int(seconds[1])+int(1.5*delta_y)
			full_lower_right[1]=int(seconds[1])+int(2*delta_y)
			buffer_right[0]=int(seconds[0])+delta_x
			buffer_right[1]=int(firsts[1])-2*delta_y
			buffer_left[0]=int(firsts[0])-delta_x
			buffer_left[1]=int(seconds[1])+int(1.5*delta_y)
			
			if(upper_left[0]<0):
				# print("upper left is giving us a problem")
				upper_left[0]=0
			if(buffer_left[0]<0):
				# print("buffer left is giving us a problem")
				buffer_left[0]=0
			if(lower_right[0]>image_2.shape[1]):
				# print("lower right is giving us a problem")
				# print("lower right is : ",lower_right[0])
				# print("the width of the image is : " ,image_2.shape[1])
				lower_right[0]=image_2.shape[1]
				# print("corrected")
			if(buffer_right[0]>image_2.shape[1]):
				# print("buffer right is giving us a problem")
				buffer_right[0]=image_2.shape[1]
				# print("corrected")
				# print("lower right is : ",lower_right[0])

				
			# color=(200,200,0)
			
			# print(root+"/"+make+"/"+image_names[0]+"_2.tiff")
			# if(len(image_names)==3):
				# image_1=cv2.imread(root+"/"+make+"/"+image_names[0]+"_1.tiff")
				# cv2.rectangle(image_1,(int(seconds[0]),int(seconds[1])),(int(firsts[0]),int(firsts[1])),(0,255,0),-1)
				# cv2.rectangle(image_1,(upper_left[0],upper_left[1]),(lower_right[0],lower_right[1]),(0,0,255),8)
			
			
			
			# cropped_region=image_2[]
			# cropped_left_buffer=image_2[upper_left[1]:buffer_left[1],upper_left[0]:buffer_left[0]]
			
			# if(buffer_left[0]!=upper_left[0]):
				# var_left=variance_of_laplacian(cropped_left_buffer)
				# cropped_left_buffer=cv2.resize(cropped_left_buffer,(200,200))
				# cv2.imshow("cropped left",cropped_left_buffer)
			# else:
				# var_left=0
			# if(buffer_right[0]!=lower_right[0]):
				# cropped_right_buffer=image_2[buffer_right[1]:lower_right[1],buffer_right[0]:lower_right[0]]
				# var_right=variance_of_laplacian(cropped_right_buffer)
				# cropped_right_buffer=cv2.resize(cropped_right_buffer,(200,200))
				# cv2.imshow("cropped right",cropped_right_buffer)
			# else:
				# var_right=0
				
				
			# sift=cv2.FeatureDetector_create("SIFT")
			sift=cv2.xfeatures2d.SIFT_create()
			kp=sift.detect(image_2,None)
			

			plate_upper_left[0]=int(firsts[0])
			plate_upper_left[1]=int(firsts[1])
			plate_lower_right[0]=int(seconds[0])
			plate_lower_right[1]=int(seconds[1])
			
			undesired_keypoints=[]
			length=len(kp)
			image_shape=image_2.shape
			if(image_shape[0]==900):
				banner_height=720
			if(image_shape[0]==1488):
				banner_height=1250
			if(image_shape[0]==2136):
				banner_height=1900
			if(image_shape[0]==3000):
				banner_height=2660
				
			for i in range(len(kp)):
				if((kp[i].pt[0]<upper_left[0] or kp[i].pt[0]>lower_right[0] or kp[i].pt[1]<full_upper_left[1] or kp[i].pt[1]>full_lower_right[1]) 
				or (kp[i].pt[0]>plate_upper_left[0] and kp[i].pt[0]<plate_lower_right[0] and kp[i].pt[1]>plate_upper_left[1] and kp[i].pt[1]<plate_lower_right[1]) or kp[i].pt[1]>banner_height):
					undesired_keypoints.append(i)
					
			for i in reversed(undesired_keypoints):
				kp.pop(i)
			
			image_2=cv2.drawKeypoints(image_2,kp,image_2)
			counter+=1
			sh=image_2.shape
			cv2.rectangle(image_2,(upper_left[0],full_upper_left[1]),(lower_right[0],full_lower_right[1]),(0,255,0),8)
			if(image_shape[0]==3000 or image_shape[0]==2136):
				image_2=cv2.resize(image_2,(int(sh[1]/3),int(sh[0]/3)))
			else:
				image_2=cv2.resize(image_2,(int(sh[1]/2),int(sh[0]/2)))
			print(image_shape)
			cv2.imshow("image",image_2)
			cv2.waitKey(0)
			# break
			continue
			
			
			# cv2.putText(image_2,"%s : %lf"%("variance right",var_right),(1000,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
			# cv2.putText(image_2,"%s : %lf"%("variance left",var_left),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
			
			# cv2.rectangle(image_2,(int(seconds[0]),int(seconds[1])),(int(firsts[0]),int(firsts[1])),(0,255,0),-1)
			# cv2.rectangle(image_2,(upper_left[0],upper_left[1]),(lower_right[0],lower_right[1]),(0,0,255),8)
			# cv2.rectangle(image_2,(middle_upper_left[0],middle_upper_left[1]),(middle_lower_right[0],middle_lower_right[1]),(0,255,255),8)
			# cv2.rectangle(image_2,(full_upper_left[0],full_upper_left[1]),(full_lower_right[0],full_lower_right[1]),(255,255,0),8)
			# cv2.rectangle(image_2,(upper_left[0],full_upper_left[1]),(lower_right[0],full_lower_right[1]),(0,255,0),8)
			# cv2.rectangle(image_2,(upper_left[0],upper_left[1]),(buffer_left[0],buffer_left[1]),(255,255,255),-1)
			# cv2.rectangle(image_2,(buffer_right[0],buffer_right[1]),(lower_right[0],lower_right[1]),(255,255,255),-1)
			# image_1=cv2.resize(image_1,(600,300))
			# image_2=cv2.resize(image_2,(600,300))
			# cv2.imshow("first image ",image_1)
			
			cropped=image_2[full_upper_left[1]:full_lower_right[1],upper_left[0]:lower_right[0]]
			# cv2.imshow("cropped image ",cropped)
			# cv2.imshow("second image ",image_2)
			
			# cv2.imshow("./train/processed/"+make+"/"+image_names[0]+"_2.tiff",cropped)
			# cv2.waitKey(0)
			# cv2.imwrite("./train/processed/"+make+"/"+image_names[0]+"_2.tiff",cropped)
			
			
			# cv2.waitKey(0)
			
			
			
			
			
		else:
			# print("this image does not interest us")
			continue
		# print(root+"/"+make+"/"+image_names[0]+"_2.tiff")
				