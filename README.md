### 项目说明

该项目用于CIFAR-10数据集的图像分类任务。采用Keras官方网址给出的ResNet模型。

#### 准备工作

- 下载CIFAR-10数据集文件，解压后放在当前目录下，下载的网址为：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 。
- 安装Python3,以及模块tensorflow, keras, python-opencv, numpy.
#### 脚本说明

- show_image.py: 用于展示CIFAR-10数据集中的某一张图片；
- load_data.py: 导入CIFAR-10数据集，用于模型训练；
- cifar10_model.py: 模型训练，训练后的模型保存在save_models文件夹。

#### 模型训练

运行cifar10_model.py，训练共200个epoch，视自己的情况决定是否开启GPU训练。

笔者自己训练模型的最终结果为：

train_loss: 0.1743

train_acc: 0.9825

val_loss: 0.4439

val_acc: 0.9128

最好的val_acc: 0.91410，第160个epoch.