# CNN-Model-Animal-Detection

## Dataset About - Animals with Attributes 2


Deep learning tabanlı bir **Convolutional Neural Network (CNN)** modeli ile  
**Animals with Attributes 2** veri setindeki hayvan sınıflarını sınıflandırıyorum.
---


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNNS-blue?style=for-the-badge)


---

JPEG Images of 50 Animal Classes
**Disclaimer**: The images in this archive are not from the original "Animals with Attributes (AwA)" dataset.
Original dataset: http://cvml.ist.ac.at/AwA, from publications [3,4].These images are not currently publicly available.
This dataset is created as a plug-in replacement for the original AwA dataset under the name "Animals with Attributes 2 (AwA2)" and shares similar characteristics.It contains the same classes and a similar number of images per class.

**Collection of Images**:The images in this archive were collected in 2016 from publicly available web sources (Flickr, Wikimedia, etc.).During the collection process, care was taken to ensure that all images were provided with licenses that allow free use and redistribution.The license information for the images can be found in the "licenses" folder.
If you hold the copyright for one or more of these images and they were not actually released under the indicated license,
please contact us. We will update the information or remove the image from the collection.

**Dataset Overlap**:
Great care has been taken to ensure that the newly created AwA2 dataset does not overlap with images from the original AwA dataset.

## Aygaz Image Processing Bootcamp
As part of the bootcamp, the Animals with Attributes 2 dataset will be used. The classification process will focus on 10 classes: **collie, dolphin, elephant, fox, moose, rabbit, sheep, squirrel, giant panda, and polar bear**.You can either delete the remaining files or copy the selected ones to a new folder for the project.To ensure balanced data, only the first 650 images from each class will be used. (You can write a script to keep the first 650 images for each of the 10 classes and delete the rest.)All images must be resized to the same dimensions and normalized. Resize the images according to the requirements of your model's input layer.Next, divide the data into train (training) and test sets. Ensure that the data is randomly split into training and testing sets **(e.g., 70% training, 30% testing)**.Finally, to enhance the model's ability to handle different scenarios and reduce overfitting, apply **data augmentation** to the training set. Use various manipulation techniques such as blurring, edge detection, resizing, rotation, and noise addition.

## CNN Model Building
Design a CNN model to distinguish between 10 animal classes. Consider incorporating various layers such as convolution, activation, pooling, fully connected, and dropout layers.Select an appropriate **loss function**.


