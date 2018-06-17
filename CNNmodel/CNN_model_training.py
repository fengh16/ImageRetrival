import os
import numpy as np
import tensorflow as tf

import CNN_model as model
import image_preprocessing as image_P

#图片分类数
N_CLASSES=10
#图片尺寸
IMG_W=400
IMG_H=300

BATCH_SIZE=8
CAPACITY=4096

#最大迭代次数
MAX_STEP=1000000

#学习速率
learning_rate=0.0001

def run_training():
    #目录
    train_dir="./image/"
    logs_train_dir ="./log"

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

    sess=tf.Session()
    train_writer=tf.summary.FileWriter(logs_train_dir,sess.graph)
    #保存模型
    saver=tf.train.Saver()

    #初始化
    sess.run(tf.global_variables_initializer())

    #使用多线程
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break#线程停止
            _,temp_loss,temp_acc=sess.run([train_op,train_loss,train_acc])
            
            #每迭代50次打印一次结果
            if step%50 == 0:
                print('Step %d,train loss = %.2f,train accuracy = %.2f'%(step,temp_loss,temp_acc))
                summary_str=sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
            
            #每迭代200次或到达最后一次保存一次模型
            if step%200 == 0 or (step+1) == MAX_STEP:
                checkpoint_path=os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print('Failed!')
    finally:
        coord.request_stop()

    #结束线程
    coord.join(threads)

    sess.close()

run_training()