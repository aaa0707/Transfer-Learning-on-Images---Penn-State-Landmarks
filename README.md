# Transfer-Learning-on-Images---Penn-State-Landmarks

Aditya Agarwal,
Sahil Mishra

This project explores transfer learning for image classification. It tries to classify various landmarks at The Pennsylvania State University into different classes.

The landmarks include the *Nittany Lion*, the *Old Main*, and the *Beaver Stadium*. Some examples are shown below.
//Images

We used two main ways to classify the images:
1. SVM Classifier on Features extracted from **AlexNet**
2. Replacing last layer of the **VGG16** network to re-train on new classes

### Transfer Learning using SVM Classifier using AlexNet

### Transfer Learning on VGG16 by replacing last layer

*Required: Keras for Python and Tensorflow installed*

VGG16 is a 16 layer neural network. We imported a Keras implementation of the VGG16 which is available with the Keras API package.
Tensorflow was used as the backend for Keras.

