import os
import numpy as np
import tensorflow as tf
from PIL import Image
from random import sample as mysample

#训练图片的路径
train_dir="./image/"

#提取每张图片的信息和标签
def get_files(file_dir, nums = 99999):
    #用于存放各种数据和标签
    All_Image=[[] for i in range(10)]
    All_Label=[[] for i in range(10)]
    All_Image_New=[[] for i in range(10)]
    All_Label_New=[[] for i in range(10)]

    totalnum = 0

    #根据图片名称和标签进行划分
    for file in os.listdir(file_dir):
        name=file.split(sep='_')
        totalnum += 1
        if name[0]=='n03877845':
            All_Image[0].append(file_dir+file)
            All_Label[0].append(0)
        elif name[0]=='n02278980':
            All_Image[1].append(file_dir+file)
            All_Label[1].append(1)
        elif name[0]=='n11669921':
            All_Image[2].append(file_dir+file)
            All_Label[2].append(2)
        elif name[0]=='n01613177':
            All_Image[3].append(file_dir+file)
            All_Label[3].append(3)
        elif name[0]=='n01923025':
            All_Image[4].append(file_dir+file)
            All_Label[4].append(4)
        elif name[0]=='n04515003':
            All_Image[5].append(file_dir+file)
            All_Label[5].append(5)
        elif name[0]=='n04583620':
            All_Image[6].append(file_dir+file)
            All_Label[6].append(6)
        elif name[0]=='n03767203':
            All_Image[7].append(file_dir+file)
            All_Label[7].append(7)
        elif name[0]=='n07897438':
            All_Image[8].append(file_dir+file)
            All_Label[8].append(8)
        elif name[0]=='n10247358':
            All_Image[9].append(file_dir+file)
            All_Label[9].append(9)

    realNum = 0

    # 如果发现数目太多了，需要筛选一下
    if totalnum > nums:
        print("Now choose " + str(nums) + " from " + str(totalnum))
        for i in range(9):
            rate = float(nums / totalnum)
            chosen = mysample(range(len(All_Image[i])), int(len(All_Image[i]) * rate))
            # chosen = chosen.sort()
            # 现在找到的要选择的是index的标签在chosen中的元素
            for j in chosen:
                All_Image_New[i].append(All_Image[i][j])
                All_Label_New[i].append(All_Label[i][j])
                realNum += 1
            nums -= int(len(All_Image[i]) * rate)
            totalnum -= len(All_Image[i])

        # 现在处理第10个
        i = 9
        chosen = mysample(range(len(All_Image[i])), nums)
        for j in chosen:
            All_Image_New[i].append(All_Image[i][j])
            All_Label_New[i].append(All_Label[i][j])
            realNum += 1

    print("Real chosen " + str(realNum))
    #水平合并数组
    Image_List=np.hstack(All_Image_New[i] for i in range(10))
    Label_List=np.hstack(All_Label_New[i] for i in range(10))

    #转置后随机排列
    temp=np.array([Image_List,Label_List])
    temp=temp.transpose()
    np.random.shuffle(temp)

    Image_List=list(temp[:,0])
    Label_List=list(temp[:,1])
    Label_List=[int(i) for i in Label_List]

    #返回两个list
    return Image_List,Label_List

#产生用于训练批次
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #进行类型转换
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    #加入队列
    input_queue=tf.train.slice_input_producer([image,label])

    label=input_queue[1]
    image_content=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_content,channels=3)

    #图像大小resize一致并进行标准化处理
    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)

    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=16,capacity=capacity)

    label_batch=tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch

