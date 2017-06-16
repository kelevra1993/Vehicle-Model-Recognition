import os

directories=["fisher_vectors","reduced_data"]

for directory in directories:
	files=os.listdir("./"+directory+"/")
	for file in files:
		file_path="./"+directory+"/"+file
		print(file_path)
		if(not(file.endswith(".py"))):
			final_files=os.listdir(file_path)
			for fp in final_files:
				final_path=file_path+"/"+fp
				print(final_path)
				os.remove(final_path)

gmm_dir="./GMM"			
gmm_files=os.listdir(gmm_dir)
for gmm in gmm_files:
	final_gmm=gmm_dir+"/"+gmm
	print(final_gmm)
	os.remove(final_gmm)