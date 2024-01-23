
# # Patch Extraction from Mammograms
# 
# The following code provides the tools for extracting patches from the [BCDR dataset](https://bcdr.eu/information/about). Given the path where this dataset is saved, the code outputs the following:
# 
# - A folder named 'healthy_patch_dataset' containing healthy patches extracted from film and digital mammograms.
# - A folder named 'lesion_patch_dataset' containing patches of lesions extracted from film and digital mammograms using manually annotated masks. These patches are extracted at different levels of zoom, with varying percentages of healthy adjacent tissue included. There are three different levels of zoom available. Zoom group 1 corresponds to the minimum percentage of healthy adjacent tissue, i.e. to a bounding box of the annotated lesion.
# - 'digital_healthy_patch_features.csv': a CSV file containing the features of the healthy patches extracted from digital mammograms.
# - 'film_healthy_patch_features.csv': a CSV file containing the features of the healthy patches extracted from film mammograms.
# - 'lesion_patch_features.csv': a CSV file containing the features of the lesion patches.
# 
# In the BCDR dataset, only normal (healthy) mammograms are available in the digital dataset. To extract healthy patches from film mammograms as well, I extracted patches from healthy tissue in suspicious film images. I ensured that there was a margin between the extracted healthy patches and the lesion patch.
# 
# Note:
# - All generated images are 224x224 pixels, black and white, and in PNG format.
# - Each generated patch has a unique ID that corresponds to the metadata saved in the respective CSV file.
# 

import cv2 as cv
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import pandas as pd
import re

root=input("Path of BCDR folder: ")

# In order to extract the lesion patches, I have used the corresponding mask annotations. Here is a sample image along with its corresponding mask:

example_image_path= os.path.join(root, "BCDR-F01_dataset\patient_9\study_10\img_9_10_1_RCC.tif")
example_mask_path= os.path.join(root, "BCDR-F01_dataset\patient_9\study_10\img_9_10_1_RCC_mask_id_10.tif")

img1 = skio.imread(example_image_path, plugin="tifffile")
img1_mask = skio.imread(example_mask_path, plugin="tifffile")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img1, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Mammogram')

axes[1].imshow(img1_mask, cmap='gray')
axes[1].axis('off')
axes[1].set_title('Mask')

plt.tight_layout()
plt.show()

# Applying a threshold to the mammogram and joining it with the mask:

ret, thresh = cv.threshold(img1, 1, 255, cv.THRESH_OTSU)
#find the contours in the image
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
plt.imshow(thresh, cmap='gray')
plt.show()

img1_mask_inverted=cv.bitwise_not(img1_mask)
dst = cv.addWeighted(thresh, 0.5, img1_mask_inverted, 0.5, 0.0)
plt.imshow(dst, cmap='gray')
plt.show()

# Some utils to create the folders and to save the patches with an unique id as file name.

def create_patch_dataset(rootpath, healthy=False):
    '''It creates a folder for healthy patches and lesion patches.'''
    if healthy:
        path=os.path.join(rootpath, r'data\\healthy_patch_dataset')
    else:
        path=os.path.join(rootpath, r'data\\lesion_patch_dataset')
    if not os.path.exists(path):
        os.makedirs(path)
        
    return path

def save_patch(folder, patch, file_name, extension, patch_id):
    '''It saves the patch in the corresponding folder with the corresponding patch ID. '''
    #save the image in the folder 
    os.chdir(folder)
    cv.imwrite(str(patch_id)+'.png', patch)
    print('\n patch saved as '+ str(patch_id)+'.png')

# The following functions are utility functions for extracting the patches:
# 
# - 'check_intersection': Given two rectangles, it checks if they intersect. This function is used to extract the healthy patches from the film mammograms.
# 
# - 'white_pixels': Given a thresholded patch, it returns 'True' if the number of white pixels exceeds the specified 'ratio_white'. This function is used to avoid patches with a higher ratio of background pixels than desired.
# 
# - 'sliding_window': it yields a window of the mammogram covering the full image.
# 
# - 'gif_healthy_patch_extractor'. This function takes as input: 
#     - an image (matrix)
#     - the width and height of the window to be extracted (they will be resized to 221x221)
#     - the percentage of the width to be the step size of the sliding windows process
#     - the maximum ratio of white pixels for the 'white_pixels' function
# 	- the mask (matrix) of the lesion, if it is a suspicious image, otherwise, it is None
#     
#     The output is a sequence of images with the patch contours to visualise how the algorithm works.
# 
# - 'healthy_patch_extractor': It does the same procedure as 'gif_healthy_patch_extractor' function, but it saves the patches in the corresponding folder.

def check_intersection(rect1, rect2):
    # rect1 and rect2 are tuples of (x, y, width, height)
	#It returns True if they overlap and False if they don't
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    # Calculate the coordinates of the corners of the two rectangles
    x1_right, y1_bottom = x1 + w1, y1 + h1
    x2_right, y2_bottom = x2 + w2, y2 + h2

    # Check if there is any overlap between the two rectangles
    if x1_right >= x2 and x2_right >= x1 and y1_bottom >= y2 and y2_bottom >= y1:
        return True
    else:
        return False
    
def white_pixels(ratio_white, patch):
    #Returns true if more than 'ratio_white' pixels are white
    #ratio_white is for example 0.5 for checking a 50%
    width, height = patch.shape
    white_pixels = 0
    for x in range(width):
        for y in range(height):
            pixel = patch[x, y]
            if pixel == 255:
                white_pixels += 1

    # Check if more than X of the pixels are white
    total_pixels = width * height
    if white_pixels / total_pixels >= ratio_white:
        return True
    else:
        return False

import time 

def sliding_window(image, stepSize, windowSize):
	#stepSize=1 i.e. sliding windows pixel by pixel
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def gif_healthy_patch_extractor(image, w, h, step_size_per, white_pix_per, mask=None, suspicious=False):
	'''
		image: matrix
		w, h: width and height of the window, they will be resized to 221x221
		step_size_per: percentage of the width to be the step size of the sliding windows process
		white_pix_per: minimum percentage of the patch to correspond to mamma and not to background 
		mask: matrix, only for the suspicious one, for non suspicious it is None
	'''
	#Apply thresholding to the image
	ret, thresh = cv.threshold(image, 1, 255, cv.THRESH_OTSU)
	
	if suspicious:
		img1_mask_inverted=cv.bitwise_not(mask)
		dst = cv.addWeighted(thresh, 0.5, img1_mask_inverted, 0.5, 0.0)   
	else:
		dst=thresh

	#Step size based on the percentage passed to the function
	step_size= int(step_size_per*w)
	for (x, y, window) in sliding_window(dst, step_size, (w,h)):
		if window.shape[0] != w or window.shape[1] != h:
				continue
		
		if suspicious:
			#We get the lesion coordinates and its width and height
			x_lesion, y_lesion, width_lesion, height_lesion=cv.boundingRect(mask) 

			#We check that any of the coordinates included in 'window' are not inside the lesion patch + A MARGIN (from (x_lesion,y_lesion) to (x_lesion+width_lesion,y_lesion+height_lesion))
			#Window coordinates: from (x, y) to (x+221, y+221)
			if check_intersection((x_lesion, y_lesion, width_lesion, height_lesion),(x, y, w, h))== True:
				continue
		
		if white_pixels(white_pix_per, window)==True:
			#save image
			# Create a Rectangle patch
			img_box = cv.rectangle(image, (x,y),(x+w,y+h), (255, 0, 0) , 3)
			plt.imshow(img_box, cmap='gray')
			plt.show()
			time.sleep(0.25)
		else: 
			#too many pixels from the background
			continue


def healthy_patch_extractor(dataset, patch_path,  patch_id, image, zoom_group, w, h, step_size_per, white_pix_per, file_name, mask=None, suspicious=False):
	'''
		patch_id: next patch identifier
		image: matrix
		w, h: width and height of the window, they will be resized to 221x221
		step_size_per: percentage of the width to be the step size of the sliding windows process
		white_pix_per: minimum percentage of the patch to correspond to mamma and not to background 
		file_name: name of the mammogram in order to save the patch with it
		mask: matrix, only for the suspicious one, for non suspicious it is None
	'''

	ret, thresh = cv.threshold(image, 1, 255, cv.THRESH_OTSU)

	if suspicious:
		img1_mask_inverted=cv.bitwise_not(mask)
		dst = cv.addWeighted(thresh, 0.5, img1_mask_inverted, 0.5, 0.0)
	else:
		dst=thresh
  
	step_size= int(step_size_per*w)

	for (x, y, window) in sliding_window(dst, step_size, (w,h)):
		if window.shape[0] != w or window.shape[1] != h:
				continue
		if zoom_group==1:
			w=w
			h=h
		if suspicious:
			#We get the lesion coordinates and its width and height
			x_lesion, y_lesion, width_lesion, height_lesion=cv.boundingRect(mask) 

			#WE ADD A MARGIN TO THIS RECTANGLE
			x_lesion=x_lesion-0.1*w
			y_lesion=y_lesion-0.1*w
			width_lesion=width_lesion+2*0.1*w
			height_lesion=height_lesion+2*0.1*w

			#We check that any of the coordinates included in 'window' are not inside the lesion patch + A MARGIN (from (x_lesion,y_lesion) to (x_lesion+width_lesion,y_lesion+height_lesion))
			#Window coordinates: from (x, y) to (x+221, y+221)
			if check_intersection((x_lesion, y_lesion, width_lesion, height_lesion),(x, y, w, h))== True:
				continue
			
		if white_pixels(white_pix_per, window)==True:
			patch=image[y:y+h, x:x+w]
			patch=cv.resize(patch, (224,224))
			#Save it:
			save_patch(patch_path, patch, file_name, dataset, patch_id)
			patch_id=patch_id+1
		else: 
			continue
	return patch_id

# Running the 'gif' function for suspicious mammograms to visualise the patches extracted from a suspicious film image.

#Demo of how the healthy patch extractor works for a suspicious mammo
path=os.path.join(root, 'BCDR-F01_dataset\patient_9\study_10\img_9_10_1_RCC.tif')
img1 = skio.imread(path, plugin="tifffile")
mask_path=os.path.join(root, 'BCDR-F01_dataset\patient_9\study_10\img_9_10_1_RCC_mask_id_10.tif')
img1_mask = skio.imread(mask_path, plugin="tifffile")

step_per=0.7
white_per=0.5
gif_healthy_patch_extractor(img1, 221, 221, step_per, white_per, img1_mask, suspicious=True)

# Running the 'gif' function for a non suspicious mammogram to visualise the patches extracted from a normal digital image.

path=os.path.join(root, 'BCDR-DN01_dataset\patient_23\study_33\img_23_33_1_LCC.tif')
img2 = skio.imread(path, plugin="tifffile")
w=h=int(img2.shape[1]*0.2)
step_size_per=1
white_per=0.5


gif_healthy_patch_extractor(img2, w, h, step_size_per, white_per)

# # Extracting patches sistematically

# Some utilities:
# - 'image_view_from_file': from the image view label, create an integer mapping
# - 'img_id': given the original mammography file name, it returns the patient id, the study id, the series and the image view
# - 'non_suspicious_patch_extraction': Patch extraction from NON suspicious digital mammograms.

#Type of image view (1-RCC, 2-LCC, 3-RO, 4-LO)
def image_view_from_file(s):
    if (s.find("RCC")>0): return 1
    elif (s.find("LCC")>0): return 2
    elif (s.find("RO")>0): return 3
    elif (s.find("LO")>0): return 4
    else: return 0

#Retrieve information from the mammography file name 
def img_id(s):
    patient_id, study_id, series=re.findall(r'\d+', s)
    image_view=image_view_from_file(s)
    return patient_id, study_id, series, image_view

def non_suspicious_patch_extraction(patch_id):
    '''
        Patch extraction from NON suspicious digital mammograms
    '''
    healthy_patch_path=create_patch_dataset(root,healthy=True)

    ext='BCDR-DN01_dataset'
    rootdir=os.path.join(root, ext)

    print('Working on ', ext, ' subdataset.')

    # Read the corresponding .csv file
    df= pd.DataFrame()
    
    df_path=os.path.join(root,'BCDR-DN01_dataset\bcdr_dn01_img.csv')
    df = pd.read_csv(df_path)
        
    #Name of the subdataset
    dataset= ext.partition("_dataset")[0]
    dataset=dataset.partition("-")[2]  

    #Create an empty patch_dataset file to save the info related to every patch
    digital_healthy_patch=pd.DataFrame(columns = df.columns)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".tif"):
                file_mammo=file.partition(".tif")[0] 
                
                patient_id, study_id, series, image_view=img_id(file_mammo) #retrieve info of the image from the file name

                file_path=os.path.join(rootdir, 'patient_'+ patient_id, 'study_'+ study_id, file) #directory of the mammo
                mammo = skio.imread(file_path, plugin="tifffile") #read the mammo

                w=h=int(mammo.shape[1]*0.2)
                step_size_per=1.0
                white_per=0.6
                zoom=1

                #The function returns the patch_id of the following one (the first one extracted from the following image)
                patch_id_new=healthy_patch_extractor(dataset, healthy_patch_path, patch_id, mammo,zoom, w, h, step_size_per, white_per, file_mammo)

                n_patches=patch_id_new-patch_id

                '''Save healthy patches metadata'''

                patch_data=pd.DataFrame()

                patch_data=df.loc[(df['patient_id'] == int(patient_id)) & (df['study_id'] == int(study_id)) & (df['series'] == int(series)) & (df['image_type_id']== int(image_view-1))]
                if patch_data.shape[0]>1:
                    patch_data=patch_data.head(1)

                #print(patch_data)
                row_to_copy=patch_data.iloc[0].tolist()       
                #del row_to_copy[0] 
                copies = [row_to_copy] * n_patches
                copies=pd.DataFrame(copies, columns = df.columns)

                copies['zoom_group']=zoom
                copies['patch_id']=range(patch_id, patch_id_new)
                copies['patch_file_name']=copies['patch_id'].astype(str) + '.png'
                
                patch_id=patch_id_new

                #Setting an index to the dataframe
                copies.index = list(range(len(copies)))
                pd.set_option('mode.chained_assignment', None)
                if not copies.empty:
                    copies.loc[:,'Format'] = 'Digital'
                    copies['patient_id']=copies['patient_id'].astype(str) + '_' + dataset
                    
                    digital_healthy_patch = pd.concat([digital_healthy_patch, copies], ignore_index=True)
                else: 
                    print('Empty metadata for: '+ str(patch_id)+'_'+dataset+'_'+ file_mammo+ '.png')
                    
    ###os.chdir(os.path.join(root, 'data'))
    os.chdir(root)
    #Move the patch_id column to the first position
    first_column =digital_healthy_patch.pop('patch_id')
    digital_healthy_patch.insert(0, 'patch_id', first_column)
    digital_healthy_patch['patch_id'] = digital_healthy_patch['patch_id'].astype(int)

    #Save the Dataframe as a .csv file in the root directory folder
    digital_healthy_patch.to_csv('digital_healthy_patch_features.csv',index=False)  

# - 'suspicious_patch_dataset' function is the complete pipeline of the patch extraction from mammograms of healthy and lesion patches. For the film dataset, the healthy patches are extracted from suspicious mammograms.

def check_coordinates(image, x, y):
    height=image.shape[0]
    width=image.shape[1]
    if x<0: x=0
    if x>width: x=width
    if y<0: y=0
    if y>height: y=height
    return x, y

def suspicious_patch_dataset():
    '''
        Patch extraction from mammograms, healthy and non-healthy patches. 
        For the film dataset, we extract the healthy patches from suspicious mammograms.
    '''
    healthy_patch_path=create_patch_dataset(root,healthy=True)
    lesion_patch_path=create_patch_dataset(root,healthy=False)

    ### FROM FILM SUSPICIOUS
    ext=['BCDR-F01_dataset','BCDR-F02_dataset', 'BCDR-F03_dataset', 'BCDR-D01_dataset', 'BCDR-D02_dataset']
    df_ext=['bcdr_f01_features.csv','bcdr_f02_features.csv','BCDR-F03\\bcdr_f03_features.csv', 'bcdr_d01_features.csv','bcdr_d02_features.csv']
    zoom_groups=[1, 2, 3]

    # Initialize the new id for the patches created
    patch_id=0

    for extension, df_extension in zip(ext, df_ext):

        print('Working on ', extension, ' subdataset.')
        rootdir=os.path.join(root, extension)

        #Name of the subdataset
        dataset= extension.partition("_dataset")[0]
        dataset=dataset.partition("-")[2]

        # Read the corresponding _outlines.csv file
        df= pd.DataFrame()
        df_path=os.path.join(rootdir,df_extension)
        df = pd.read_csv(df_path)

        #Create an empty patch_dataset file to save the info related to every patch
        if extension=='BCDR-F01_dataset':
            # For healthy patches 
            healthy_patch_outlines=pd.DataFrame()#columns = df.columns)
            #For non healthy patches
            lesion_patch_outlines=pd.DataFrame(columns = df.columns)

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith(".tif"):
                    if file.find("mask")>0: #look for the mammo with a mask created, as not all the mammograms have a mask
                        file_mammo=file.partition("_mask_id_")[0] #retrieve the name of the original mammo of which the mask was created
                        patient_id, study_id, series, image_view=img_id(file_mammo) #retrieve info of the image from the file name
                        mask_id=os.path.splitext(file.partition("_mask_id_")[2])[0]
                        mask_id=int(mask_id)

                        file_path=os.path.join(rootdir, 'patient_'+ patient_id, 'study_'+ study_id, file) #directory of the mask
                        mask = skio.imread(file_path, plugin="tifffile") #read the mask
                        
                        mammo_path=os.path.join(rootdir, 'patient_'+ patient_id, 'study_'+ study_id, file_mammo + '.tif') #directory of the mammo
                        mammo = skio.imread(mammo_path, plugin="tifffile") #read the mammo
                        
                        '''Lesion patch extraction'''

                        #Bounding box of the mask
                        x, y, width, height=cv.boundingRect(mask) 
                        
                        #Repeat for different zooms:
                        for zoom_factor in zoom_groups:
                            #patch = mammo[y:y+height, x:x+width]
                            new_height = int(height * zoom_factor)
                            new_width = int(width * zoom_factor)
                            new_y = int(y + height/2 - new_height/2)
                            new_x = int(x + width/2 - new_width/2)
                            new_x, new_y=check_coordinates(mammo, new_x, new_y)
                            x_2, y_2 =check_coordinates(mammo, new_x+new_width, new_y+new_height)
                           
                            patch = mammo[new_y:y_2, new_x:x_2]
                            
                            
                            # RESIZE THE PATCH IMAGE ONCE EXTRACTED
                            patch = cv.resize(patch, (224, 224))

                            save_patch(lesion_patch_path, patch, file_mammo, dataset, patch_id)

                            '''Save lesion patch metadata'''
                            #Row with the patch information
                            patch_data=pd.DataFrame()
                            patch_data=df.loc[(df['patient_id'] == int(patient_id)) & (df['study_id'] == int(study_id)) & (df['series'] == int(series)) & (df['image_view']== int(image_view)) & (df['lesion_id']==int(mask_id))]

                            # Group of zoom label
                            patch_data['zoom_group']= zoom_factor

                            #Add a column specifying the Film or Digital format
                            patch_data.insert(loc=len(patch_data.columns), column='Format', value='Film')
                            #Add a column with the patch_id and another one with the patch-file_name 
                            patch_data.insert(loc=0, column='patch_id', value=patch_id)
                            patch_data['patch_file_name'] = patch_data['patch_id'].astype(str) + '.png'
                        
                            #For some few cases, 2 rows instead of 1 where found for the same patient_id, study_id, series and image_view, they only differed on the segmentation_id variable.
                            if patch_data.shape[0]>1:
                                patch_data=patch_data.head(1)

                            if extension == 'BCDR-D02_dataset' or extension == 'BCDR-D01_dataset': 
                                pd.set_option('mode.chained_assignment', None)
                                patch_data.loc[:,'Format'] = 'Digital'

                            #in order to identify each patient_id from each subdataset, I made them unique in the whole BCDR dataset
                            substitute_value = lambda x: f"{x['patient_id']}_{dataset}"
                            patch_data['patient_id'] = patch_data.apply(substitute_value, axis=1)                        
                            
                            #Append the rows to the lesion dataframe
                            if patch_data.empty:
                                print('lesion patch data empty\n')
                                continue
                            
                            lesion_patch_outlines=pd.concat([lesion_patch_outlines,patch_data], ignore_index=True)
                            patch_id=patch_id+1
                        
                        '''Extract healthy patches only for the film ones'''
                        if extension == 'BCDR-F01_dataset' or extension == 'BCDR-F02_dataset' or extension == 'BCDR-F03_dataset':
                            w=h=224
                            step_size_per=0.8
                            white_per=0.5
                            zoom=1

                            #The function returns the patch_id of the following one (the first one extracted from the following image)
                            patch_id_new=healthy_patch_extractor(dataset, healthy_patch_path, patch_id, mammo, zoom, w, h, step_size_per, white_per, file_mammo, mask, suspicious=True)
                            #NOTE: patch_id is already incremented in the function
                            n_patches=patch_id_new-patch_id

                            '''Save healthy patches metadata'''
                            del patch_data['zoom_group']
                            del patch_data['patch_file_name']
                            
                            row_to_copy=patch_data.iloc[0].tolist()     
                            
                            copies = [row_to_copy] * n_patches
                            cols=patch_data.columns
                            
                            cols=cols[1:].to_list()
                            
                            copies=pd.DataFrame(copies)
                            if not copies.empty:
                                copies['zoom_group']=zoom
                                copies=copies.drop(copies.columns[0], axis=1)
                                #copies=copies.drop(copies.columns[44], axis=1)                    
                                copies['patch_id']=range(patch_id, patch_id_new)
                                
                            if copies.empty:
                                print('healthy film patch data empty\n')
                                continue
                            copies['patch_file_name'] = copies['patch_id'].astype(str) + '.png'
                                #copies['patch_file_name']=str(i)+'_'+dataset+'_'+ file_mammo+ '.png'
                            #print('Patch data: ', patch_data, '\nRow to copy', row_to_copy, '\nCopies', copies)
                            patch_id=patch_id_new
        
                            copies.rename(columns=dict(zip(copies.columns, cols)),  inplace = True)
                        
                            #Append the rows to the healthy dataframe
                            healthy_patch_outlines=pd.concat([healthy_patch_outlines,copies], ignore_index=True)

    os.chdir(root)
    #Move the patch_id column to the first position
    first_column =lesion_patch_outlines.pop('patch_id')
    lesion_patch_outlines.insert(0, 'patch_id', first_column)
    lesion_patch_outlines['patch_id'] = lesion_patch_outlines['patch_id'].astype(int)

    #Save the Dataframe as a .csv file in the root directory folder
    lesion_patch_outlines.to_csv('lesion_patch_features.csv',index=False)

    #Move the patch_id column to the first position
    first_column =healthy_patch_outlines.pop('patch_id')
    healthy_patch_outlines.insert(0, 'patch_id', first_column)
    healthy_patch_outlines['patch_id'] = healthy_patch_outlines['patch_id'].astype(int)

    #Save the Dataframe as a .csv file in the root directory folder  
    healthy_patch_outlines.to_csv('film_healthy_patch_features.csv',index=False)

    '''Digital healthy patch extraction from the digital normal subdataset'''
    non_suspicious_patch_extraction(patch_id)

    print('\nHealthy patch dataset saved in: ', healthy_patch_path, '\nPatch file names have the following structure:\n (patch_id)_(dataset folder)_(patient_id)_(study_id)_(series)_(image_view).png')
    print('\nInformation gathered about each lesion patch saved in :', root, 'as lesion_patch_features.csv')
    print('\nInformation gathered about each healthy patch saved in :', root, 'as film_healthy_patch_features.csv')
    print('\nInformation gathered about each digital healthy patch saved in :', root, 'as digital_healthy_patch_features.csv')

suspicious_patch_dataset()
