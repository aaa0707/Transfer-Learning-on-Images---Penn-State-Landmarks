from keras.applications import vgg16
from keras.models import Model
import os
import cv2
import numpy as np
import scipy.ndimage
import random
from keras.models import model_from_yaml


model = vgg16.VGG16()
new_model = Model(inputs=model.input,outputs=model.layers[-2].output)
#
from keras.layers import Dense
#
for l in new_model.layers:
    l.trainable=False

#functional interface
# define a new layer, and call it on the output of the last model
x = Dense(3,activation="softmax")(new_model.output)

final_model = Model(inputs=new_model.input,outputs=x)

print final_model.summary()
final_model.compile(loss="categorical_crossentropy",optimizer="adam")

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = []
for f in os.listdir(os.path.join(dir_path,'data/training_data/nittany_lion')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/training_data/nittany_lion', f)):
        #print os.listdir(os.path.join(dir_path,'kinect _ eBay'))
        a = cv2.imread(os.path.join(dir_path, 'data/training_data/nittany_lion', f))
        #print os.path.join(dir_path, 'kinect _ eBay', f)
        #print a.shape
        b = cv2.resize(a, (224, 224))  - np.mean(np.mean(a, axis = 0), axis = 0)
        dataset.append((np.array([[1, 0, 0]]), b))
        cv2.imshow('image', b)
        cv2.waitKey(20)
for f in os.listdir(os.path.join(dir_path,'data/training_data/old_main')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/training_data/old_main', f)):
        a = cv2.imread(os.path.join(dir_path, 'data/training_data/old_main', f))
        b = cv2.resize(a, (224, 224)) - np.mean(a)
        dataset.append((np.array([[0, 1, 0]]), b))
#    print(b.shape)
        cv2.imshow('image', b)
        cv2.waitKey(20)

for f in os.listdir(os.path.join(dir_path,'data/training_data/beaver_stadium')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/training_data/beaver_stadium', f)):
        a = cv2.imread(os.path.join(dir_path, 'data/training_data/beaver_stadium', f))
        b = cv2.resize(a, (224, 224)) - np.mean(a)
        dataset.append((np.array([[0, 0, 1]]), b))
#    print(b.shape)
        cv2.imshow('image', b)
        cv2.waitKey(20)

# for f in os.listdir(os.path.join(dir_path,'random_stuff')):
#     if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'random_stuff', f)):
#         a = cv2.imread(os.path.join(dir_path, 'random_stuff', f))
#         b = cv2.resize(a, (224, 224)) - np.mean(a)
#         dataset.append((np.array([[0, 0, 1]]), b))
#     #    print(b.shape)
#         cv2.imshow('image', b)
#         cv2.waitKey(20)

random.shuffle(dataset)
print(dataset)

k = 0;
for x,y in dataset:
    k=k+1
    print k
    #print x.shape
    #print y.shape
    z = np.concatenate([y[np.newaxis]])
    #print z.shape
    final_model.fit(z, x)

model_yaml = final_model.to_yaml()
with open("model.yaml", "w") as yaml_file:
   yaml_file.write(model_yaml)
final_model.save_weights("weights.h5")
#
