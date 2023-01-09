# Fader_Network

### <u>Clone The project :
Create a folder on your machine then run : 
` git clone https://github.com/anesnabti/Fader_Network.git `

### requirements

- Tensorflow 
- opencv
- You can go to Fader_Network/utils/install/ and run the file `Install.bat`
The file will install all requirement librairies.

### Dataset

After downloding dataset from [CelebA] (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Extract from **CelebA-20221101T214610Z-001.zip** the folders **Eval**, **img_align_celeba** and **Anno**. Copy them in ...\Fader_Network\data . 

### Preprocessing and Split Data 

On Fader_Network, run `preprocess_split_data.bat`
This file will resize to 256x256 all images of our dataset and copy them into Fader_Network/data/img_align_celeba_resize
It will also generated a preproced file of attributs. You can find it on Fader_Network/data/Attributs.npy

Also, this file will split resized images into train_images, test_images ans validation_images. 
The same thing will be done for attributs
Results are saved on ` ../data/train     ../data/test   ../data/validation `

You can change the rate of split data by changing the adequats variables in `preprocess_split_data.bat`. They are (0.7, 0.15, 0.15) by default.

### Train model

On Fader_Network, run **train_shell.bat** 
The file will execute `train_main.py` which is in Fader_Network/src
Epochs, nbr_itr, batch_size, and attributs are set as default. You can modify them by changing the argument of **train_shell.bat** 

Some results of the training phase will be saved :
- `utils/loss/ ` : loss of training
- `utils/result_train/ : resulting images of training phase. One image per 2 epochs. 
