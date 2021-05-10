'''
@File  :preprocessing.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 5:07 PM
@Desc  :time-domain raw data to 2-d matrix
'''

import scipy.io
import random
import numpy as np
import matplotlib.pyplot as plt


def read_matdata(path):
    raw_data = scipy.io.loadmat(path)  # 读取mat文件
    # print(data.keys())  # 查看mat文件中的所有变量
    #print(raw_data.keys())
    del raw_data['__header__'], raw_data['__version__'], raw_data['__globals__'], raw_data['X118RPM'], \
        raw_data['X118_BA_time']
    print(raw_data.keys())
    return raw_data
    # scipy.io.savemat('matData2.mat', {'matrix1': matrix1, 'matrix2': matrix2})  # 写入mat文件


def signal2image(path, img_number, img_size):
    raw_data = read_matdata(path)
    DE_data = raw_data['X118_DE_time']
    print(DE_data.shape[0])
    #print(min(DE_data))
    for i in range(img_number):
        begin_number = random.randint(0, DE_data.shape[0] - img_size + 1)
        img = DE_data[begin_number:begin_number + img_size]
        img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
        img = np.reshape(img, (int(img_size**0.5), int(img_size**0.5)))
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

def read_folder(folder_path):
    pass



data_path = '../../data/12kDriveEnd/B007_0.mat'
matrix_number = 15
matrix_size = 64 * 64
signal2image(data_path, matrix_number, matrix_size)
