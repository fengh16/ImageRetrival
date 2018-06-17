# CNNmodel

一个自己写的CNN模型，用于图片分类。文件说明如下：

* `CNN_model.py`：CNN模型的主体部分
* `image_preprocessing.py`：用于对图像预处理，提取标签等操作
* `CNN_model_training.py`：用于训练并保存模型（保存到log文件夹下）
* `image_test_CNN.py`：用于加载训练数据，输入图片得到10张最相似图像的结果
* `CNN_model_training_test.py`：用于训练模型并且自动测试模型预测准确度，使用时第一个参数为训练时迭代轮数（默认为400轮），第二个参数为使用的图片数目（默认为1024）
* `image`文件夹：存放5613张图片的目录
* `imagelist.txt`：储存所有图片的名称索引