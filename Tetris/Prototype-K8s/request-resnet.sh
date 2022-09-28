#!/bin/bash
# set -e
#访问请求

# request-resnet.sh Dockerfile dog.jpg requirements.txt resnet simple-keras-rest-api run_keras_server.py
while true
do
	curl -X POST -F image=@/CIFAR10/dog.jpg 'http://localhost:31188/predict'

	curl
	-F "pic=@/CIFAR10/1.jpg; filename='1.jpg'"
	-F "pic=@/CIFAR10/2.jpg; filename='2.jpg'"
	-F "pic=@/CIFAR10/3.jpg; filename='3.jpg'"
	http://localhost:31188/predict
	
	echo "#################################### Request send ####################################"
        sleep 1
done
