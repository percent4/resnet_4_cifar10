# -*- coding: utf-8 -*-
import cv2
import pickle

# 读取文件
fpath = 'cifar-10-batches-py/data_batch_1'
with open(fpath, 'rb') as f:
    d = pickle.load(f, encoding='bytes')

data = d[b'data']
labels = d[b'labels']
data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

# 保存第image_no张图片
strings=['airplane', 'automobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']
image_no = 1000
label = strings[labels[image_no]]
image = data[image_no,:,:,:]
cv2.imwrite('%s.jpg' % label, image)
