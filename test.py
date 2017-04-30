#from keras.applications import vgg16
#from keras.models import Model
import os
import cv2
import numpy as np
import scipy.ndimage
import random
from keras.models import model_from_yaml
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

# # load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("weights.h5")
#print("Loaded model from disk...")
loaded_model.compile(loss="categorical_crossentropy",optimizer="adam")

dir_path = os.path.dirname(os.path.realpath(__file__))
total = 0
correct = 0

for f in os.listdir(os.path.join(dir_path,'data/testing_data/nittany_lion')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/testing_data/nittany_lion', f)):
        a = cv2.imread(os.path.join(dir_path, 'data/testing_data/nittany_lion', f))
        b = cv2.resize(a, (224, 224))  - np.mean(np.mean(a, axis = 0), axis = 0)
        out = loaded_model.predict(np.concatenate([b[np.newaxis]]), batch_size = 1, verbose = 0)
        total = total+1
        if out[0][0]>out[0][1] and out[0][0]>out[0][2]:
            correct = correct+1
            print total, out, 'Nittany Lion'
        else:
            print f, out


for f in os.listdir(os.path.join(dir_path,'data/testing_data/old_main')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/testing_data/old_main', f)):
        a = cv2.imread(os.path.join(dir_path, 'data/testing_data/old_main', f))
        b = cv2.resize(a, (224, 224))  - np.mean(np.mean(a, axis = 0), axis = 0)
        out = loaded_model.predict(np.concatenate([b[np.newaxis]]), batch_size = 1, verbose = 0)
        total = total+1
        if out[0][1]>out[0][0] and out[0][1]>out[0][2]:
            correct = correct+1
            print total, out, 'Old Main'
        else:
            print f, out


for f in os.listdir(os.path.join(dir_path,'data/testing_data/beaver_stadium')):
    if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, 'data/testing_data/beaver_stadium', f)):
        a = cv2.imread(os.path.join(dir_path, 'data/testing_data/beaver_stadium', f))
        b = cv2.resize(a, (224, 224))  - np.mean(np.mean(a, axis = 0), axis = 0)
        out = loaded_model.predict(np.concatenate([b[np.newaxis]]), batch_size = 1, verbose = 0)
        total = total+1
        if out[0][2]>out[0][0] and out[0][2]>out[0][1]:
            correct = correct+1
            print total, out, 'Beaver Stadium'
        else:
            print f, out



print ('Accuracy =', (correct*100/total))
# im = cv2.imread(os.path.join(dir_path, sys.argv[1]))#'pic_025.jpg'))
# #print im.shape
# #cv2.imshow('image', im)
# im2 = cv2.resize(im,(224, 224))# - np.mean(np.mean(im, axis=0), axis = 0);

#print loaded_model.summary()
# evaluate loaded model on test data

#print(np.concatenate([im2[np.newaxis]]));

#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
# print ('Nittany Lion: %d' % (out[0][0]*100))
# print ('Old Main: %d' % (out[0][1]*100))
#print ('Reject: %d' % (out[0][2]*100))
