import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import CNN_model as model
import image_preprocessing as image_P

import sys

def get_one_image(img_dir):
    image=Image.open(img_dir)
    image=image.resize([400,300])
    image_arr = np.array(image)

    return image_arr


N_CLASSES=10
IMG_W=400
IMG_H=300

BATCH_SIZE=8
CAPACITY=1024

MAX_STEP=400

if (len(sys.argv) > 1):
    MAX_STEP = int(sys.argv[1])

if (len(sys.argv) > 2):
    CAPACITY = int(sys.argv[2])

learning_rate=0.0001

def run_training():
    train_dir="./image/"
    logs_train_dir ="./log_" + str(MAX_STEP) + "_cap_" + str(CAPACITY)

    train,train_label=image_P.get_files(train_dir)
    train_batch,train_label_batch=image_P.get_batch(train,
                                                    train_label,
                                                    IMG_W,
                                                    IMG_H,
                                                    BATCH_SIZE,
                                                    CAPACITY)
    train_logits=model.inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss=model.losses(train_logits,train_label_batch)
    train_op=model.trainning(train_loss,learning_rate)
    train_acc=model.evaluation(train_logits,train_label_batch)

    summary_op=tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth = True))

    sess = tf.Session(config=config)
    train_writer=tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver=tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners(sess=sess)

    for step in np.arange(9999999):
        sess.run([train_op,train_loss,train_acc])
        
        print(step)
        
        if step == MAX_STEP:
            print("Finished")
            break

    log_dir = train_dir

    images_dir='./image/'
    

    images_cat=open("imagelist.txt")
    #保存所有图像经过模型计算之后的数组
    images_tested=[]

    num_photos = 0
    outfile = open("test-new-" + str(MAX_STEP) + "-" + str(CAPACITY) + ".txt", "w")

    filelines = images_cat.readlines()
    for line in filelines:
        image_name=line.strip('\n')
        image_array=get_one_image(images_dir+image_name)
        image_array=np.reshape(image_array,[1,300,400,3])                 

        xName =tf.placeholder(tf.float32,shape=[1,300,400,3])
        prediction=sess.run(train_logits,feed_dict={xName:image_array})
        prediction=np.array(prediction,dtype='float32')
        images_tested.append([image_name,prediction])

        num_photos += 1
        print("Test:" + str(num_photos))

        outfile.writelines(image_name)
        t = str(prediction)
        outfile.writelines(t)
        outfile.close()
        outfile = open("test-new-" + str(MAX_STEP) + "-" + str(CAPACITY) + ".txt", "a")

    outfile.close()
    outfile2 = open("nearesttest-" + str(MAX_STEP) + "-" + str(CAPACITY) + ".txt", "w")
    outfile2.write("result = {\n")
    outfile2.close()
    num_photos = 0
    for line in filelines:
        num_photos += 1
        print("Find Near:" + str(num_photos))
        image_name=line.strip('\n')
        image_array=get_one_image(images_dir+image_name)
        image_array=np.reshape(image_array,[1,300,400,3])                 
        outfile2 = open("nearesttest-" + str(MAX_STEP) + "-" + str(CAPACITY) + ".txt", "a")
        outfile2.write("'" + image_name + "': [\n")
        xName =tf.placeholder(tf.float32,shape=[1,300,400,3])
        prediction=sess.run(train_logits,feed_dict={xName:image_array})
        prediction=np.array(prediction,dtype='float32')
        
        test_result=[]
        for sample in images_tested:
            distance=np.sqrt(np.sum(np.square(sample[1]-prediction)))
            distance.astype('float32')
            test_result.append([sample[0],distance])
                            
        #将结果排序
        test_result=np.array(test_result)
        test_result=test_result[np.lexsort(test_result.T)]
        for i in range(11):
            outfile2.write("'" + test_result[i][0] + "', ")
        outfile2.write("],\n")
        outfile2.close()

    outfile2 = open("nearesttest-" + str(MAX_STEP) + "-" + str(CAPACITY) + ".txt", "a")
    outfile2.write("}\n")
    outfile2.close()



    sess.close()

run_training()