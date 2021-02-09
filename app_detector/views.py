import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from django.shortcuts import render 
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse, HttpResponseServerError
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import cv2
from django.http import FileResponse
from wsgiref.util import FileWrapper
from django.views.decorators import gzip
import base64
import uuid
import io
import zipfile
import requests
import json
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image
from core.config import cfg
import colorsys
import random
from tensorflow.keras.models import model_from_json
#from classification.cnn import classify

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
    
input_size = 416

# load json and create model
json_file = open('plastic.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
plastic_model = model_from_json(loaded_model_json)
plastic_model.load_weights("plastic.h5")



    
def local(request):
    return render(request, 'local.html')

class JSONImage(APIView):
    def post(self, request):
        image_file = request.FILES['image'].read()
        imageFound = False
        bboxes = []
        data = []
        datas = []

        try:
            #img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
            original_image      = cv2.imdecode(np.fromstring(image_file, np.uint8), cv2.IMREAD_UNCHANGED)
            gray_image          = cv2.imdecode(np.fromstring(image_file, np.uint8), cv2.IMREAD_GRAYSCALE)
            original_image_size = original_image.shape[:2]

            #path = "D:\\programming\\plastic_detection\\plastic_detection\\app_detector\\static\\"
            #cv2.imwrite(path + "test1.jpg", original_image)

            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            imageFound = True
        except:
            pass

        if imageFound:
            imgcv = image_data
            #results = tfnet.return_predict(imgcv)
            pred_bbox = plastic_model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            font = cv2.FONT_HERSHEY_SIMPLEX

            for box in bboxes:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                #test_img = cv2.imdecode(np.fromstring(image_file, np.uint8), cv2.IMREAD_UNCHANGED)[y:h, x:w]
                new_img = gray_image[y:h, x:w]
                #data_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                #url = 'http://localhost:8083/classify/testimg'
                #myobj = {'image': new_img}

                #headers = {'Content-Type': 'text/plain', 'Accept':'plain/plain'}
                #data = requests.post(url, data = myobj, headers=headers)
                #print(data_image, "----new img")
                #print(data, "=====")
                graphA = tf.Graph()
                with graphA.as_default():
                    from classification.cnn import classify
                    label = classify(gray_image)
                    data.append([box[0],box[1],box[2],box[3], label["confidence"], int(label["class"])])


            image = utils.draw_bbox(original_image, data, './data/classes/plastic_classification.names')
            #image = cv2.rectangle(original_image)
            
            #image = utils.draw_bbox(original_image, bboxes, '')
            currentDirectory = os.getcwd()
            full_path = os.path.dirname(os.path.realpath(__file__))
            imagePath = os.path.join(full_path, "static")
            path = os.path.join(imagePath, "test1.jpg")
            print(path, "path")
            cv2.imwrite(path, image)

            print(data)
            print(bboxes)

            #path = "D:\\programming\\plastic_detection\\plastic_detection\\app_detector\\static\\"
            #cv2.imwrite(path + "test3.jpg", test_img)
                
            datas = {'bbox': bboxes, 'label': data}

        return Response(data)

        

    







            