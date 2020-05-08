# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

def euclidean_distance(train_images,train_labels,test_image):
    # return a array
    d = np.sum((train_images-test_image)**2,axis = 1)
    # print("hello")
    s = np.concatenate((d.reshape((len(train_labels),1)),train_labels.reshape((len(train_labels),1))),axis=1)
    return s
    

def find_majority(labels):
    a = {}
    for label in labels:
        if label in a:
            a[label] += 1
        else:
            a[label] = 1
    majority_label = max(a,key = a.get)
    return majority_label

def predict(k,train_images,train_labels,test_image):
    distance = euclidean_distance(train_images,train_labels,test_image)
    
    sort_distance = distance[np.argsort(distance[:,0])]
    
    # print(sort_distance[:k])
    k_labels = [label for (_,label) in sort_distance[:k]]
    # print(k_labels)
    return find_majority(k_labels)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist_data')

train_images = np.asarray(mnist.train.images)
train_labels = np.asarray(mnist.train.labels)
test_images = np.asarray(mnist.test.images)
test_labels = np.asarray(mnist.test.labels)

i = 0
total_correct = 0
for test_image in test_images:
    pred = predict(11, train_images, train_labels, test_image)
    if pred == test_labels[i]:
        total_correct += 1
    acc = (total_correct / (i+1)) * 100
    if i%20 ==0:
        print('test image['+str(i)+']', '\tpred:', pred, '\torig:', test_labels[i], '\tacc:', str(round(acc, 2))+'%')
        s = test_image.reshape((28,28))
        s = 255*s
        im = Image.fromarray(s.astype('uint8'))
        name = i
        im.save(str(i)+'.jpg')
    i += 1
