import os 
import cv2
import csv
import numpy as np

root="./train/makes"
makes=os.listdir(root)
#First we are going to open the csv file with all the information necessary for segmentation
'''OPEN A NEW CSV FILE FOR READING'''
upper_left=[0,0]
middle_upper_left=[0,0]
full_upper_left=[0,0]
lower_right=[0,0]
middle_lower_right=[0,0]
full_lower_right=[0,0]
buffer_right=[0,0]
buffer_left=[0,0]

def variance_of_laplacian(image):
	#this function compute the laplacian of an image and returns it's focus
	#the measure is simply the variancce of the laplacian
	return cv2.Laplacian(image,cv2.CV_64F).var()
	
with open('processed_trainer.csv','rb') as csvfile:
	linereader=csv.reader(csvfile,delimiter=";")
	for row in linereader :
		make=row[15]
		
		#to ingore the next line which is the seocnd image, when there is a second image....
		image_names=row[10].split("_")
		if(len(image_names)==3):
			try:
				linereader.next()
			except StopIteration:
				print("we caught stop iteration ")	
		license_plate=row[13]
		print(license_plate)
		positions=license_plate.split("_")
		print(positions)
		firsts=positions[0].split("x")
		seconds=positions[2].split("x")
		print("first are : ",firsts)
		print("seconds are : ",seconds)
		print("\n")
		delta_x=abs(int(seconds[0])-int(firsts[0]))
		delta_y=abs(int(seconds[1])-int(firsts[1]))
		print("delta x is : ",delta_x)
		print("delta y is : ",delta_y)
		print("\n")
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
			print("upper left is giving us a problem")
			upper_left[0]=0
		if(buffer_left[0]<0):
			print("buffer left is giving us a problem")
			buffer_left[0]=0
		if(lower_right[0]>image_2.shape[1]):
			print("lower right is giving us a problem")
			print("lower right is : ",lower_right[0])
			print("the width of the image is : " ,image_2.shape[1])
			lower_right[0]=image_2.shape[1]
			print("corrected")
		if(buffer_right[0]>image_2.shape[1]):
			print("buffer right is giving us a problem")
			buffer_right[0]=image_2.shape[1]
			print("corrected")
			# print("lower right is : ",lower_right[0])

		color=(200,200,0)
		
		# print(root+"/"+make+"/"+image_names[0]+"_2.tiff")
		if(len(image_names)==3):
			image_1=cv2.imread(root+"/"+make+"/"+image_names[0]+"_1.tiff")
			cv2.rectangle(image_1,(int(seconds[0]),int(seconds[1])),(int(firsts[0]),int(firsts[1])),(0,255,0),-1)
			cv2.rectangle(image_1,(upper_left[0],upper_left[1]),(lower_right[0],lower_right[1]),(0,0,255),8)
		
		
		
		# cropped_region=image_2[]
		cropped_left_buffer=image_2[upper_left[1]:buffer_left[1],upper_left[0]:buffer_left[0]]
		var_left=variance_of_laplacian(cropped_left_buffer)
		cropped_left_buffer=cv2.resize(cropped_left_buffer,(200,200))
		if(buffer_right[0]!=lower_right[0]):
			cropped_right_buffer=image_2[buffer_right[1]:lower_right[1],buffer_right[0]:lower_right[0]]
			var_right=variance_of_laplacian(cropped_right_buffer)
			cropped_right_buffer=cv2.resize(cropped_right_buffer,(200,200))
			cv2.imshow("cropped right",cropped_right_buffer)
		cv2.imshow("cropped left",cropped_left_buffer)
		
		cv2.putText(image_2,"%s : %lf"%("variance right",var_right),(1000,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
		cv2.putText(image_2,"%s : %lf"%("variance left",var_left),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
		
		cv2.rectangle(image_2,(int(seconds[0]),int(seconds[1])),(int(firsts[0]),int(firsts[1])),(0,255,0),-1)
		cv2.rectangle(image_2,(upper_left[0],upper_left[1]),(lower_right[0],lower_right[1]),(0,0,255),8)
		cv2.rectangle(image_2,(middle_upper_left[0],middle_upper_left[1]),(middle_lower_right[0],middle_lower_right[1]),(0,255,255),8)
		cv2.rectangle(image_2,(full_upper_left[0],full_upper_left[1]),(full_lower_right[0],full_lower_right[1]),(255,255,0),8)
		cv2.rectangle(image_2,(upper_left[0],full_upper_left[1]),(lower_right[0],full_lower_right[1]),(0,255,0),8)
		cv2.rectangle(image_2,(upper_left[0],upper_left[1]),(buffer_left[0],buffer_left[1]),(255,255,255),-1)
		cv2.rectangle(image_2,(buffer_right[0],buffer_right[1]),(lower_right[0],lower_right[1]),(255,255,255),-1)
		image_1=cv2.resize(image_1,(600,300))
		image_2=cv2.resize(image_2,(600,300))
		# cv2.imshow("first image ",image_1)
		cv2.imshow("second image ",image_2)
		cv2.waitKey(0)