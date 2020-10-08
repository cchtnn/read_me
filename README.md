# Automated Optical Inspection

## 1. Introduction
Automated Optical Inspection is an automatic defect tracking method for detecting defect in sample scanned images. 
We present an improved deep learning-based end to end approach for solving problems of detection defective images 
using a pre trained resnet50 and GRAD-CAM model. 

<img src="imgs/main_res.png"/>

## 2. Setup
<b>Models are developed in Pytorch based <a href="https://github.com/open-mmlab/mmdetection">MMdetection</a> framework (Version 1.2)</b>
<br>

<pre>
conda install -y jupyter
conda create --name torch python==3.7
conda activate torch
conda install nb_conda
git clone https://github.com/jacobgil/pytorch-grad-cam.git
pip install -r {"requirements.txt"}
pip install pillow==7.2.0
pip install scikit-image==0.17.2
</pre>

<b>Code is developed under following library dependencies</b> <br>

PyTorch = 1.6.0<br>
Torchvision = 0.7.0<br>
Cudatoolkit = 10.2.89<br>

<pre>
** Install Pytorch for CPU only
conda install pytorch -c pytorch

** Install Pytorch for GPU and CPU
conda install pytorch cudatoolkit -c pytorch 

** Install Additional library for ML
conda env update --file tools.yml

** Register your Environment
python  -m ipykernel install --user --name pytorch --display-name "Python 3.7 (pytorch)"
</pre>

**If you are using Google Colaboratory (Colab), Then you need add**
```
from google.colab.patches import cv2_imshow
```
and replace all the `cv2.imshow` with `cv2_imshow`

## 3. Model Architecture
<img src="imgs/model arch.png" width="550"/>
<a href="imgs/theonnx.onnx.svg">Model Computation Graph</a>

## 4. Image Augmentation
<img src="imgs/3imgs.png" width="750"/><br>
Codes: <a href="https://github.com/DevashishPrasad/CascadeTabNet/blob/master/Data%20Preparation/Dilation.py">Code for dilation transform</a> <a href="https://github.com/DevashishPrasad/CascadeTabNet/blob/master/Data%20Preparation/Smudge.py">Code for smudge transform</a>

## 5. Usage
### 5.1. Prepare the image data sets
In order to start the transfer learning process, a folder named ``training_dataset`` needs to be created in the root of the project folder. This folder will contain the image data sets for all the subjects, for whom the classification is to be performed.

Create the ``training_dataset`` ,``testing_dataset``, ``validatation_dataset``, folder and add the images for all the data sets in the following manner

```CV_Project
/
|
|
---- /EEPReg_Cropped
|    |
|    |
|    ---- /train
|    |    
|    | 
     --------------Defect   
|    |             1:a.png
|    |             2:b.png
|    |
|    --------------Non Defect
|    |             1.c.png
|    |             2:d.png
|    |
|    ---- /test
     |    
|    | 
     --------------Defect 
|    |             1:d.png
|    |             2:e.png
|    |
|    --------------Non Defect   
|    |             1:f.png
|    |             2:g.png
|    |             
|    ---- /valid
|    |
|    |
|    ---------------Defect
|    |             1:f.png
|    |             ....
|    |
|    ---------------Non Defect
|    |              1:f.png
|    |              ....
|    |
```
This enables classification of images between the ``Defective`` and ``Non Defective`` data sets.

> Make sure to include multiple variants of the subject (side profiles, zoomed in images etc.), the more the images, the better is the result.

## 6. 

### Crop large image of size (2150 * 3124) into smaller pieces (224 * 224 * 3).
```python
python cropped_cv.py
```


### Model Training (Initiate transfer learning)
Go to the project directory and run -
```python
python transfer_learning.py
```

This script creates the augmented images and initiates the re-training process for the specified image data sets. The final layer of the resnet50 model was 
changed for fine tuning. We are using Adam as the optimizaer.

Once the process is complete, it will return a training accuracy somewhere between ``85% - 95%``.

After successfull completion of model training and validation, model weights will be saved in a root folder named ``CV_Project``.

### Testing Model's Accuracy on sample Images
```python
python3 test_acc.py
```

This script picks the images from "datasets" folder , preprocess it and feed to the model which predicts whether the image is Defective or Non Defective
with percentage of each category.

### Create Heatmap on sample Image
```python
python3 gradcam_orig.py
```
<img src="D:/Download Chriome/cat-dog.png" width="750"/><br>

This script picks the images from datasets folder , preprocess it and feed to the model which creates heatmap on defective part of the image in case if the image is defective else
returns the image without any changes.

Reference : https://github.com/jacobgil/pytorch-grad-cam

Note : Added loop in the code so that in case if multiple image will be passed as input then it will generate multiple images with heatmaps on it. Using model created from
transfer_learning.py script.

## 7. Datasets

1) Full Dataset(Image Files) is present on Microsoft Azure Storage Container <br>
Dataset Path :  /Curated/MedicalDevices/tmp/CS_Project/EEPReg_Cropped<br>

2) Cropped Images of Large sample image is present on Microsoft Azure Storage Container <br>
Dataset Path : /Curated/MedicalDevices/tmp/CS_Project/datasets<br>

## 8. Training
You may refer this <a href="https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/">tutorial</a> for training Mmdetection models on your custom datasets in colab.<br>

You may refer this <a href="/Data Preparation/generateVOC2JSON.py">script</a> to convert your Pascal VOC XML annotation files to a single COCO Json file.

## Contact
Marco Perez Cuello : marco [dot] perezcuello [at] philips [dot] com <br>

## Acknowledgements

We thank the following contributions because of which
the paper was made possible