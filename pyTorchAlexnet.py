import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from numpy.fft import rfft2, irfft2
from sklearn import svm

model = models.alexnet(pretrained=True)

new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

iteration = 1
correct = 0
incorrect = []

features = []
while(iteration <= 185):
    folder = ''
    if (iteration <= 64):
        folder = 'beaver_stadium'
    elif ((iteration > 64) and (iteration <= 124)):
        folder = 'nittany_lion'
    else:
        folder = 'old_main'
    img_pil = Image.open('data/training_data/'+folder+'/image'+str(iteration)+'.jpg')

    preprocess = transforms.Compose([
       transforms.Scale(256),
       transforms.CenterCrop(224),
       transforms.ToTensor()
    ])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)

    img_var = Variable(img_tensor)
    out = model(img_var)
    newOut = np.squeeze(out)

    features.append(np.squeeze(out.data.numpy()))

    print("iteration ", iteration, " done.")
    iteration += 1

labels = []
for i in range(1, 186):
    if (i <= 64):
        labels.append(0)
        #0 is Beaver Stadium
    elif ((i > 64) and (i <= 124)):
        labels.append(1)
        #1 is Nittany Lion
    else:
        labels.append(2)
        #2 is Old Main

clf = svm.SVC(verbose=True)
clf.fit(features, labels)

testIteration = 1
while(testIteration <= 82):
    folder = ''
    if (testIteration <= 26):
        folder = 'beaver_stadium'
    elif ((testIteration > 26) and (testIteration <= 52)):
        folder = 'nittany_lion'
    else:
        folder = 'old_main'
    img_pil = Image.open('data/testing_data/'+folder+'/image'+str(testIteration)+'.jpg')

# img_pil = Image.open('/Users/sahilmishra/Desktop/Pictures/Dataset/image178.jpg')
    preprocess = transforms.Compose([
       transforms.Scale(256),
       transforms.CenterCrop(224),
       transforms.ToTensor()
    ])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)

    img_var = Variable(img_tensor)
    out = model(img_var)
    # print type(out)
    # print out
    newOut = np.squeeze(out)
    pred = clf.predict(newOut.data.numpy())
    if (pred == 0):
        if(testIteration <= 26):
            correct += 1
        else:
            incorrect.append([testIteration, pred])
        print("beaver_stadium")
    elif (pred == 1):
        if ((testIteration > 26) and (testIteration <= 52)):
            correct += 1
        else:
            incorrect.append([testIteration, pred])
        print("nittany_lion")
    else:
        if (testIteration > 52):
            correct += 1
        else:
            incorrect.append([testIteration, pred])
        print("old_main")
    testIteration += 1
    print("testIteration: " + str(testIteration) + "done.")

accuracy = (correct/82) * 100
print("accuracy: ", accuracy)
print("incorrect: ", incorrect)
