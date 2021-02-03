import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.keras.models import model_from_json

LR = 1e-3
MODEL_NAME = 'plastic-{}-{}.model'.format(LR, '6conv')
IMG_SIZE = 50
TEST_DIR = 'D:/programming/CNN/plastic_test/test'

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 20, activation='softmax')
convnet = regression(convnet, optimizer='adam', metric='accuracy', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model_loaded!')

# if you dont have this file yet
#test_data = process_test_data()
# if you already have it
#test_data = np.load('test_data.npy', allow_pickle=True)

def classify(img_src):
    #img_array = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    #img_array = cv2.imdecode(np.fromstring(img_src, np.uint8), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_src, (IMG_SIZE, IMG_SIZE))
    predictdata = np.array(new_array).reshape(IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict([predictdata])
    model_out = prediction[0]
    model_accuracy = prediction
        
    if np.argmax(model_out) == 0:
        str_label='Fissan'
    elif np.argmax(model_out) == 1:
        str_label='FlawlesslyU'
    elif np.argmax(model_out) == 2:
        str_label='Zonrox'
    elif np.argmax(model_out) == 3:
        str_label='CDOIdol'
    elif np.argmax(model_out) == 4:
        str_label='Funtastyktocino'
    elif np.argmax(model_out) == 5:
        str_label='tenderjuicy'
    elif np.argmax(model_out) == 6:
        str_label='ajinomoto'
    elif np.argmax(model_out) == 7:
        str_label='clearshampoo'
    elif np.argmax(model_out) == 8:
        str_label='7up'
    elif np.argmax(model_out) == 9:
        str_label='absolute'
    elif np.argmax(model_out) == 10:
        str_label='aquafina'
    elif np.argmax(model_out) == 11:
        str_label='blue'
    elif np.argmax(model_out) == 12:
        str_label='c2'
    elif np.argmax(model_out) == 13:
        str_label='cupnoodlesbatchoy'
    elif np.argmax(model_out) == 14:
        str_label='star margarine'
    elif np.argmax(model_out) == 15:
        str_label='foam cup'
    elif np.argmax(model_out) == 16:
        str_label='foam plate'
    elif np.argmax(model_out) == 17:
        str_label='styro foam'
    elif np.argmax(model_out) == 18:
        str_label='octagon'
    elif np.argmax(model_out) == 19:
        str_label='pipefittings'

    datas = []
    return {'class': np.argmax(model_out), 'confidence': prediction[0][np.argmax(model_out)] * 100}


