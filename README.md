# Automated Optical Inspection

## 1. Introduction
Automated Optical Inspection is an automatic defect tracking method for interpretation of tabular data in document images. We present an improved deep learning-based end to end approach for solving both problems of table detection and structure recognition using a single Convolution Neural Network (CNN) model. CascadeTabNet is a Cascade mask Region-based CNN High-Resolution Network (Cascade mask R-CNN HRNet) based model that detects the regions of tables and recognizes the structural body cells from the detected tables at the same time. We evaluate our results on ICDAR 2013, ICDAR 2019 and TableBank public datasets. We achieved 3rd rank in ICDAR 2019 post-competition results for table detection while attaining the best accuracy results for the ICDAR 2013 and TableBank dataset. We also attain the highest accuracy results on the ICDAR 2019 table structure recognition dataset. 

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
|    |             1.png
|    |             2.png
|    |
|    --------------Non Defect
|    |             1.png
|    |             2.png
|    |
|    ---- /test
     |    
|    | 
     --------------Defect 
|    |             1.png
|    |             2.png
|    |
|    --------------Non Defect   
|    |
|    |
|    ---- /valid
|    |
|    |
|    ---------------Defect
|    |
|    |
|    |
|    ---------------Non Defect
|     
```
This enables classification of images between the ``Defective`` and ``Non Defective`` data sets.

> Make sure to include multiple variants of the subject (side profiles, zoomed in images etc.), the more the images, the better is the result.

## 6. 
### Model Training (Initiate transfer learning)
Go to the project directory and run -
```python
$ bash train.sh
```

This script installs the ``Inception`` model and initiates the re-training process for the specified image data sets.

Once the process is complete, it will return a training accuracy somewhere between ``85% - 100%``.

The ``training summaries``, ``retrained graphs`` and ``retrained labels`` will be saved in a folder named ``tf_files``.

### Testing Model's Accuracy on sample Images
```javascript
python3 test_acc.py
```

This script picks the images from datasets folder , preprocess it and feed to the model which predicts whether the image is Defective or Non Defective with percentage.

### Create Heatmap on sample Image
```javascript
python3 gradcam_orig.py
```
This script picks the images from datasets folder , preprocess it and feed to the model which creates heatmap on defective part of the image in case if the image is defective else
returns the image without any changes.

## 7. Datasets

1) Full Dataset(Image Files) is present on Microsoft Azure Storage Container <br>
Dataset Path :  /Curated/MedicalDevices/tmp/CS_Project/EEPReg_Cropped<br>

2) Cropped Images of  Dataset (ICDAR 19 + Marmot + Github)<br>
Dataset Path : /Curated/MedicalDevices/tmp/CS_Project/datasets<br>

## 8. Training
You may refer this <a href="https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/">tutorial</a> for training Mmdetection models on your custom datasets in colab.<br>

You may refer this <a href="/Data Preparation/generateVOC2JSON.py">script</a> to convert your Pascal VOC XML annotation files to a single COCO Json file.

## Contact
Marco Perez Cuello : marco [dot] perezcuello [at] philips [dot] com <br>

## Acknowledgements

We thank the following contributions because of which
the paper was made possible