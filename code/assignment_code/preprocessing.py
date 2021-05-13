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
import os
import cv2


def folder2img(save_path, folder_path, train_num, val_num, test_num, img_size):
    files = os.listdir(folder_path)
    if not os.path.exists(save_path + '/train'):
        os.makedirs(save_path + '/train')
    if not os.path.exists(save_path + '/val'):
        os.makedirs(save_path + '/val')
    if not os.path.exists(save_path + '/test'):
        os.makedirs(save_path + '/test')
    for file in files:
        #        print(type(os.path.splitext(file)[1]))
        if os.path.splitext(file)[1] == '.mat':
            file_path = folder_path + "/" + file
            raw_data = scipy.io.loadmat(file_path)
            for name in raw_data.keys():
                if "DE" in name:
                    column_DE = name
                    print(column_DE)
            data_DE = raw_data[column_DE]
            #           print(data_DE.shape[0])
            # print(min(DE_data))
            if ("_3" in os.path.splitext(file)[0]) or ("_0" in os.path.splitext(file)[0]):
                choices = [i for i in range(data_DE.shape[0] - img_size)]
                print(os.path.splitext(file)[0], "train")
                choice = random.sample(choices ,train_num)
                for i in choice:
                    begin_number = i
                    img = data_DE[begin_number:begin_number + img_size]
                    img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
                    img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
                    cv2.imwrite(save_path + '/train/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)

            elif ("_1") in os.path.splitext(file)[0]:
                choices = [i for i in range(data_DE.shape[0] - img_size)]
                print(os.path.splitext(file)[0], "val")
                choice = random.sample(choices, val_num)
                for i in choice:
                    begin_number = i
                    img = data_DE[begin_number:begin_number + img_size]
                    img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
                    img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
                    cv2.imwrite(save_path + '/val/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)

            elif ("_2") in os.path.splitext(file)[0]:
                choices = [i for i in range(data_DE.shape[0] - img_size)]
                print(os.path.splitext(file)[0], "test")
                choice = random.sample(choices, test_num)
                for i in choice:
                    begin_number = i
                    img = data_DE[begin_number:begin_number + img_size]
                    img = np.round((img - min(img)) / (max(img) - min(img)) * 255)
                    img = np.reshape(img, (int(img_size ** 0.5), int(img_size ** 0.5)))
                    cv2.imwrite(save_path + '/test/' + os.path.splitext(file)[0] + '_' + str(i) + '.png', img)

            else:
                print("wrong!", os.path.splitext(file)[0])

            print('Finished creating img files for ' + file)


data_path = '../../data/12kDriveEnd'
normal_path = '../../data/Normal_Baseline_Data'
img_save_path = '../../data/12kDriveEnd_img'
train_number = 8000  # 800
val_number = 1000  # 100
test_number = 1000 # 100
matrix_size = 64 * 64

folder2img(img_save_path, data_path, train_number, val_number, test_number, matrix_size)

# signal2image(data_path, matrix_number, matrix_size)
