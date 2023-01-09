import cv2
import numpy as np
import matplotlib.image as mpimg
import pathlib
import os
import sys
from csv import writer
PARENT_PATH = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.append(PARENT_PATH + "cfg")
from config import info, debug, log_config
import shutil
from time import perf_counter

"""
This file aims to us to preprocess all images in our data. 
The purpose is to crop images and resized them to (256 x 256)
"""

PATH = os.path.dirname(os.path.abspath(__file__))
Nbr_images = 202599
SIZE_IMG = 256

def preprocessing_images ():
    log_config("preprocessing_images")
    #verifying if all images are in data
    if len(os.listdir(PATH + "\\img_align_celeba")) <  Nbr_images:
        debug("You do not have all images ! Please Check")
        print("You do not have all images ! Please Check")
        return
    else: 
        info("OK : All images are downloded")
    
    #verifying if the folder of the images resized exists.
    if  "img_align_celeba_resized" not in os.listdir(PATH):
        os.mkdir(PATH + "\\img_align_celeba_resized")
        info("img_align_celeba_resized folder is created")
        print("img_align_celeba_resized folder is created")
    
    if len(os.listdir(PATH + "\\img_align_celeba_resized")) == 0:
        info("OK : ../../img_align_celba_resized is empty, we will start resizing images ")
        print("OK : ../../img_align_celba_resized is empty, we will start resizing images ")

    elif len(os.listdir(PATH + "\\img_align_celeba_resized")) == Nbr_images  : 
        info("OK : All resized images are downloded ")
        print("OK : All resized images are downloded ")
        return

    else : 
        debug("You do not have all images resized ! Please Check")
        shutil.rmtree(PATH + "\\img_align_celeba_resized")
        debug("Suppression of ../../img_align_celeba_resized and create a new empty folder")
        os.makedirs(PATH + "\\img_align_celeba_resized")
        info('Creation of new folder OK, please restart the program')
        print('Creation of new folder OK, please restart the program')
        return
    
    
    print("############ Reading images ##############")
    print(f'resize operation will take about {5.80*Nbr_images/(1000*60)} minutes ...')
    tps1 = perf_counter()
    for i in range (1, Nbr_images + 1) :
        if i % 10000 == 0:
            print('iteration :',i)
        
        I = cv2.imread(PATH + "\\img_align_celeba\\%06i.jpg" % i)[20:-20]                # %06i% means that we have a number of 6 digits | we do [20 : -20] to crop images into 178 x 178
        if I.shape != (178,178,3):
            debug("Error cropped image")
            raise Exception (" Image %06i%  .does not been cropped correctly % i")
      
        I_resized = cv2.resize(I, (SIZE_IMG, SIZE_IMG), interpolation=cv2.INTER_LANCZOS4)
        assert I_resized.shape == (SIZE_IMG, SIZE_IMG, 3)
        cv2.imwrite(PATH + "\\img_align_celeba_resized\\%06i.jpg" % i, I_resized)
    info("Save resized images OK")
    print("Save resized images OK")
    tps2 = perf_counter()
    print(tps2 - tps1, (tps2 - tps1)/60)

    
def preprocessing_labels():
    log_config("preprocessing_labels")
    dataset_table = PATH + '\\Anno\\list_attr_celeba.txt'
    attr_lines = [line.rstrip() for line in open(dataset_table, 'r')]
    attr_keys = 'file_name' + ' '+ attr_lines[1]
    matdata = []
    for i in range(1,Nbr_images+2):
        # Add the header
        if i == 1:
            list_data = np.array(attr_keys.replace('  ',' ').replace(',','').split()).reshape(1,-1)[0]
            info("Header added correctly")
        # Add the attributs values
        else:
            list_data = np.array(attr_lines[i].replace('-1','0').replace('  ',' ').replace(',','').split()).reshape(1,-1)[0]
        matdata.append(list_data)
        info("Attributs values added correctly")

        # to save as csv
        # with open('./data/list_attr_celebatest.csv', 'a', newline='') as f_object:  
        #     # Pass the CSV  file object to the writer() function
        #     writer_object = writer(f_object)
        #     # Result - a writer object
        #     # Pass the data in the list as an argument into the writerow() function
        #     writer_object.writerow(list_data)  
        #     # Close the file object
        #     f_object.close()
 

    # to save as npy
    matdata = np.array(matdata)
    if matdata.shape != (Nbr_images+1,41):
        debug(f"Found {matdata.shape}, must have {(Nbr_images+1,41)} ")
        raise Exception (f"Found {matdata.shape}, must have {(Nbr_images+1,41)} ")
        
    np.save(PATH + '\\ATTRIBUTS.npy',np.array(matdata))
    info("ATTRIBUTS.npy saved correctly")



if __name__ == '__main__':
    # preprocessing_images()
    preprocessing_labels()
