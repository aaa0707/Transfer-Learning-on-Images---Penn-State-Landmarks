# Transfer Learning for classifying Penn State Landmarks
**Aditya Agarwal,
Sahil Mishra**

This project explores transfer learning for image classification. It tries to classify various landmarks at The Pennsylvania State University into different classes. Since the size of our dataset is not large enough to train a CNN from scratch, we try to take advantage of pre-trained CNNs that have been trained on *ImageNET* (1.2 million images with 1000 classes). We use deep learning frameworks like *PyTorch*, *Keras with TensorFlow* and Machine Learning libraries like *Scikit-Learn* (for Python) to extract feature vectors from Alexnet and fine tune the VGG16 network. 

## Platform and Frameworks

We used Python to develop our system. 

1. **PyTorch** is a python package that provides high-level features like Tensor computation (like numpy) and deep neural networks built on tape-based autograd system. 
2. **Keras** (With **TensorFlow** backend) is a high-level neural networks API, written in Python and capable of running on top of either TensorFLow or Theano. It was developed to enable fast experimentation with various deep nueral networks. 
3. **Scikit-Learn** is an open source Machine Learning library in Python that provides simple and efficient tools for data mining and data analyis. It is built on NumPy, SciPy, and matplotlib.

## Dataset
The landmarks include the *Nittany Lion*, the *Old Main*, and the *Beaver Stadium*. Some examples are shown below.

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Data.png)

This data was taken from Google Images. We used Python to scrape 100 images from Google Images at time. After cleaning and removing unwatned images, we ended up with a training set that includes about 65 images per class and test data that includes 25-30 images per class.

## Approach

We used two main ways to classify the images:
1. SVM Classifier on Features extracted from **AlexNet** using PyTorch, a deep learning framework.
2. Replacing last layer of the **VGG16** network to re-train on new classes

### 1. Transfer Learning using SVM Classifier using AlexNet

*Required: PyTorch, scikit-learn installed*

AlexNet is a large, deep convolutional neural network used to classify 1.3 million hight resolution images in the LSRVC-2010 Imagenet trainig set into 1000 different classes. The last fully connected of *Alexnet* is removed and this model is used to compute 4096 dimensional vectors for each image, which is preprocessed before fed into the model, in our training set. These vectors, or Tensors, are converted into numpy arrays and are used as features for our *SVM classifier* which is implemented using the *sklearn* library for Python. 

#### Testing
The data was tested on 25-30 separate images per class. **The accuracy recorded was 96.3%**. We used a simple accuracy calculation model where we calculated the total percentage of images that were classified correctly by our system. A total of 79 images out of 82 images were classified correctly. 

The following images were classified incorrectly:

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Misclassifications.png)

### 2. Transfer Learning on VGG16 by replacing last layer

*Required: Keras for Python and Tensorflow installed*

VGG16 [1] is a 16 layer neural network trained on ImageNet dataset. We imported a Keras implementation of the VGG16 which is available with the Keras API package. Additionally, TensorFlow was used as the backend for Keras.
VGG16 was specifically chosen because it is a very deep network and is trained on various types of images. This gives the advantage of a very diverse set of features that the network can automatically extract. One disadvantage is that the feedforward step can get slow (0.5 sec in out case) on CPUs. Since our goal is to classify images accurately and to not focus on speed, we decided to go with the deep network. The model can be downloaded [here](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/model.yaml)

We removed the last layer of the network and added a new Fully (Densely) Connected Layer with 3 the three classes mentioned above. We decided to replace only 1 layer because of the scarcity of data. The training data involves only 60 images per class. Hence, training only 1 layer is quicker and more efficient with the less amount of data. Additionally, the diversity of features extracted in VGG16 until the second to last layer ensures that the features required for discrimination (and classification) of our particular dataset are not missing.

#### Testing

The data was tested on 25-30 separate images per class. **The accuracy recorded was 97%**

The classfier correctly classified the following images correctly:

Nittany Lion:

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Testing_Example1_nittanylion.jpg)

Old Main:

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Testing_Example2_oldmain.jpg)

Beaver Stadium:

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Testing_Example3_beaverstadium.jpg)


The following images were classified incorrectly:

![alt text](https://github.com/aaa0707/Transfer-Learning-on-Images---Penn-State-Landmarks/blob/master/Misclassifications_VGG.png)

#### Refereces
[1] Simonyan, K., & Zisserman, A. (2015, April 10). Retrieved from https://arxiv.org/pdf/1409.1556.pdf

[2] Krizhevsky, A., Sutskever, I., & Hinton G (NIPS 2012). Retrieved from http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
