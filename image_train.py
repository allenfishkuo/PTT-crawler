# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:09:28 2020

@author: allen
"""
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
from tflearn.data_augmentation import ImageAugmentation
'''Setting up the env'''
  
TRAIN_DIR = './imgfilename/'
TEST_DIR = './test/'
IMG_SIZE = 256
LR = 1e-4
MODEL_NAME = 'image_classification-{}-{}.model'.format(LR, '6conv-basic') 
  

'''Labelling the dataset'''
def label_img(img): 

    # DIY One hot encoder 
    if img == '0': return [1, 0] 
    elif img == '1': return [0, 1] 
  
'''Creating the training data'''
def create_train_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
    training_data = [] 
  
    # tqdm is only used for interactive loading 
    # loading the training data 
    for file in os.listdir(TRAIN_DIR):
        file_list = TRAIN_DIR+file+'/'
        for img in tqdm(os.listdir(file_list)): 
            # labeling the images 
            label = label_img(file)      
            path = os.path.join(file_list, img)   
            # loading the image from the path and then converting them into 
            # greyscale for easier covnet prob 
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

            # resizing the image for processing them in the covnet
            try :
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) 
            except :
                continue
            # final step-forming the training data list with numpy array of the images 
            training_data.append([np.array(img), np.array(label)]) 
      
    # shuffling of the training data to preserve the random state of our data 
    shuffle(training_data) 
  
    # saving our trained data for further uses if required 
    np.save('train_data.npy', training_data) 
    return training_data 
  
'''Processing the given test data'''
# Almost same as processing the training data but 
# we dont have to label it. 

def process_test_data(): 
    testing_data = [] 
    for file in os.listdir(TEST_DIR):
        file_list = TEST_DIR+file+'/'
        for img in tqdm(os.listdir(file_list)): 
            path = os.path.join(file_list, img) 
            img_num = file
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            try :
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) 
            except :
                continue
            testing_data.append([np.array(img), img_num]) 
          
    shuffle(testing_data) 
    np.save('test_data.npy', testing_data) 
    return testing_data 

'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
#print(train_data)
test_data = process_test_data()
#print(test_data) 

train_data_imgs = [item[0] for item in train_data]
train_data_lbls = [item[1] for item in train_data]

print(train_data_imgs)
print(train_data_lbls)
# train_data = np.load('train_data.npy') 
# test_data = np.load('test_data.npy') 
'''Creating the neural network using tensorflow'''
# Importing the required libraries 

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block, batch_normalization 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
  
import tensorflow as tf 
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    # conv layer 1 w/max pooling
conv1 = conv_2d(convnet, 32, 2, activation='relu')
conv1 = max_pool_2d(conv1, 2)
    # conv layer 2 w/max pooling etc
conv2 = conv_2d(conv1, 64, 2, activation='relu')
conv2 = max_pool_2d(conv2, 2)

conv3 = conv_2d(conv2, 64, 2, activation='relu')
conv3 = max_pool_2d(conv3, 2)

conv4 = conv_2d(conv3, 128, 2, activation='relu')
conv4 = max_pool_2d(conv4, 2)

conv5 = conv_2d(conv4, 128, 2, activation='relu')
conv5 = max_pool_2d(conv5, 2)

conv6 = conv_2d(conv5, 256, 2, activation='relu')
conv6 = max_pool_2d(conv6, 2)

conv7 = conv_2d(conv6, 256, 2, activation='relu')
conv7 = max_pool_2d(conv7, 2)

conv8 = conv_2d(conv7, 512, 2, activation='relu')
conv8 = max_pool_2d(conv8, 2)
    # fully connected layer
fc1 = fully_connected(conv8, 1024, activation='relu')
fc1 = dropout(fc1, 0.8)
    # fc2
fc2 = fully_connected(fc1, 128, activation='relu')
fc2 = dropout(fc2, 0.8)
    # output layer for classification
output = fully_connected(fc2, 2, activation='softmax')
output = regression(output, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(output, tensorboard_dir='log')     # logs to temp file for tensorboard analysis


# Splitting the testing data and training data 
train = train_data[:] 
test = train_data[-500:] 
  
'''Setting up the features and lables'''
# X-Features & Y-Labels 

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test] 
  
'''Fitting the data into our model'''
# epoch = 5 taken 
model.fit(X, Y,n_epoch=50,validation_set=(test_x, test_y),snapshot_step=500,show_metric=True,run_id=MODEL_NAME)
model.save(MODEL_NAME) 

'''Testing the data'''
import matplotlib.pyplot as plt 
# if you need to create the data: 
# test_data = process_test_data() 
# if you already have some saved: 
test_data = np.load('test_data.npy',allow_pickle=True) 
  
fig = plt.figure() 
ans = []
for num, data in enumerate(test_data[:100]): 
    img_num = data[1] 
    img_data = data[0] 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1) 
  
    # model_out = model.predict([data])[0] 
    model.load('./'+str(MODEL_NAME))
    model_out = model.predict([data])[0]
    #print("model out:", model_out)
    ans.append(np.argmax(model_out))

print(ans)
    

