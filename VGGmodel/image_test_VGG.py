from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import backend
import sys

import matplotlib.pyplot as plt
from PIL import Image
import h5py as h5py
import numpy as np
import tensorflow as tf

from loadMoment import *

testdir = "./ir/"
testFile = "query.txt"

def comparef(s):
    print(s)
    return s[2]

def dista(a, b):
    ans = 0
    for i in range(9):
        ans += (data[a][i] - data[b][i]) * (data[a][i] - data[b][i])
    return ans

def get_one_image(img_dir):
    x=[]
    image=Image.open(img_dir)
    image=image.resize([400,300])
    x.append(np.array(image))
    x=np.array(x)
    return x

def images_show(images):
    images_dir='./image/'
    plt.subplots(num='result pictures window',figsize=(8,6))
    for i in range(10):
        if (images[i][0][0] == 'n'):
            image=Image.open(images_dir+images[i][0])
        else:
            image=Image.open(testdir+images[i][0])
        image=image.resize([image.width*5,image.height*5])
        plt.subplot(2,5,1+i)
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('result pictures')
    plt.show()

num = 800
if len(sys.argv) > 1:
    num = int(sys.argv[1])

images_dir='./image/'
log_dir="./VGG_log/VGG_model_new_" + str(num) + ".h5"

images_cat=open("imagelist.txt")
testdir = "./ir/"
testFile = open(testFile)
#保存所有图像经过模型计算之后的数组
images_tested=[]

num_photos = 0

#重载模型
model = load_model(log_dir)
# 取上一层的结果
# model=Model(inputs=model.input,outputs=model.get_layer('fc2').output)

for line in images_cat.readlines():
    image_name=line.strip('\n')
    image_array=get_one_image(images_dir+image_name)
    prediction=model.predict(image_array)
    prediction=np.array(prediction,dtype='float32')
    images_tested.append([image_name,prediction])
    num_photos += 1
    print(num_photos)


for line in testFile.readlines():
    image_name=line.strip('\n')
    image_array=get_one_image(testdir+image_name)
    prediction=model.predict(image_array)
    prediction=np.array(prediction,dtype='float32')
    images_tested.append([image_name,prediction])
    num_photos += 1
    print(num_photos)

testFile.close()
testFile = open("query.txt")
test_result_file = open("test_result_file.txt", "w")
for test_file in testFile:
    test_file = test_file.split("\n")[0]
    if(test_file=='z'):
        break
    image_name=test_file
    if (image_name[0] == 'n'):
        image_array=get_one_image(images_dir + image_name)
    else:
        image_array=get_one_image(testdir + image_name)
    prediction=model.predict(image_array)
    prediction=np.array(prediction,dtype='float32')
    test_result=[]
    for sample in images_tested:
        distance=np.sqrt(np.sum(np.square(sample[1]-prediction)))
        distance.astype('float32')
        if (sample[0][0] != 'n'):
            test_result.append([sample[0],distance])
    #将结果排序
    test_result=np.array(test_result)
    test_result=test_result[np.lexsort(test_result.T)]
    # print(test_result)
    dis11 = float(test_result[10][1])
    i = 10
    trylist = [[test_result[i][0], test_result[i][1], dista(test_file, test_result[i][0])]]
    alreadylist = []
    for i in range(10):
        if float(test_result[i][1]) == dis11:
            trylist.append([test_result[i][0], test_result[i][1], dista(test_file, test_result[i][0])])
        else:
            alreadylist.append(test_result[i])
    for i in range(len(test_result) - 11):
        if float(test_result[11 + i][1]) == dis11:
            trylist.append([test_result[11 + i][0], test_result[11 + i][1], dista(test_file, test_result[11 + i][0])])
        else:
            break
    trylist = sorted(trylist, key = comparef)
    # print(trylist)
    for i in range(11 - len(alreadylist)):
        alreadylist.append(trylist[i])
    for i in range(11):
        print(alreadylist[i][0])
        test_result_file.write("'" + alreadylist[i][0] + "', ")
    test_result_file.write("\n")
    # images_show(alreadylist)

def testnow(namePic):
    test_file = namePic.split("\n")[0]
    image_name=test_file
    if (image_name[0] == 'n'):
        image_array=get_one_image(images_dir + image_name)
    else:
        image_array=get_one_image(testdir + image_name)
    prediction=model.predict(image_array)
    prediction=np.array(prediction,dtype='float32')
    test_result=[]
    for sample in images_tested:
        distance=np.sqrt(np.sum(np.square(sample[1]-prediction)))
        distance.astype('float32')
        if (sample[0][0] != 'n'):
            test_result.append([sample[0],distance])
    #将结果排序
    test_result=np.array(test_result)
    test_result=test_result[np.lexsort(test_result.T)]
    # print(test_result)
    dis10 = float(test_result[9][1])
    i = 9
    trylist = [[test_result[i][0], test_result[i][1], dista(test_file, test_result[i][0])]]
    alreadylist = []
    for i in range(9):
        if float(test_result[i][1]) == dis10:
            trylist.append([test_result[i][0], test_result[i][1], dista(test_file, test_result[i][0])])
        else:
            alreadylist.append(test_result[i])
    for i in range(len(test_result) - 10):
        if float(test_result[10 + i][1]) == dis10:
            trylist.append([test_result[10 + i][0], test_result[10 + i][1], dista(test_file, test_result[10 + i][0])])
        else:
            break
    trylist = sorted(trylist, key = comparef)
    # print(trylist)
    for i in range(10 - len(alreadylist)):
        alreadylist.append(trylist[i])
    for i in range(10):
        print(alreadylist[i][0])
    images_show(alreadylist)

# 测试单张图片
while (True):
    test_file=input('输入测试图片:')
    testnow(test_file)
