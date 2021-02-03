import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from django.shortcuts import render 
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse, HttpResponseServerError
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from classification.cnn import classify
import cv2
import numpy as np


class Predict(APIView):
	def post(self, request):
		image_file = request.FILES['image'].read()
		#original_image = cv2.imdecode(np.fromstring(image_file, np.uint8), cv2.IMREAD_UNCHANGED)
		img_array = cv2.imdecode(np.fromstring(image_file, np.uint8), cv2.IMREAD_GRAYSCALE)
		text = classify(img_array)

		return Response(text)