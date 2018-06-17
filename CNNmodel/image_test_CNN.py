import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import CNN_model as model

def get_one_image(img_dir):
    image=Image.open(img_dir)
    image=image.resize([400,300])
    image_arr = np.array(image)

    return image_arr

def images_show(images):
    images_dir='./image/'
    plt.subplots(num='result pictures window',figsize=(8,6))
    for i in range(10):
        image=Image.open(images_dir+images[i][0])
        image=image.resize([image.width*5,image.height*5])
        plt.subplot(2,5,1+i)
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('result pictures')
    plt.show()
    
def test(log_dir):
    images_dir='./image/'
    

    images_cat=open("imagelist.txt")
    #保存所有图像经过模型计算之后的数组
    images_tested=[]

    with tf.Graph().as_default():
        #重载模型
        x=tf.placeholder(tf.float32,shape=[1,300,400,3])
        p=model.inference(x,1,10)
        logits=tf.nn.softmax(p)

        sess=tf.Session()
        tf.get_variable_scope().reuse_variables()
        ckpt=tf.train.get_checkpoint_state(log_dir)
        saver=tf.train.Saver()

        num_photos = 0
        outfile = open("test.txt", "w")

        for line in images_cat.readlines():
            image_name=line.strip('\n')
            image_array=get_one_image(images_dir+image_name)
            image_array=np.reshape(image_array,[1,300,400,3])                 
  
            prediction=sess.run(logits,feed_dict={x:image_array})
            prediction=np.array(prediction,dtype='float32')
            images_tested.append([image_name,prediction])

            num_photos += 1
            print(num_photos)

            outfile.writelines(image_name)
            t = str(prediction)
            outfile.writelines(t)
        
        #测试单张图片
        while (True):
            test_file=input('输入测试图片:')
            if(test_file=='z'):
                break

            image_name=test_file
            image_array=get_one_image(images_dir+image_name)
            image_array=np.reshape(image_array,[1,300,400,3])
            prediction=sess.run(logits,feed_dict={x:image_array})
            prediction=np.array(prediction,dtype='float32')
            test_result=[]
            for sample in images_tested:
                distance=np.sqrt(np.sum(np.square(sample[1]-prediction)));
                distance.astype('float32')
                test_result.append([sample[0],distance])
                                
            #将结果排序
            test_result=np.array(test_result)
            test_result=test_result[np.lexsort(test_result.T)]
            for i in range(11):
                print(test_result[i][0])

            images_show(test_result)