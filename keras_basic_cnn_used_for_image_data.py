__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

## import library
import keras
# from keras.datasets import
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import pandas
from keras.backend.tensorflow_backend import set_session

## 파라미터 변수 설정
### file 저장 시 file 이름에 대한 time stamp 반영하기 위해 설정해줌
# (코드를 실행하는 시점 기준으로 파일에 반영)
time_stamp = time.strftime("%Y%m%d_%H%M")  #YYYYmmdd_HHMM 형태로 입력됨 (ex. 20170611_1602)

batch_size = 32 # 학습 batch size 설정
num_classes = 4 # 분류 해야 할 class 개수
epochs = 5000 # 학습 반복 회수

# import data (numpy 형태로 저장되어 있는 이미지 데이터셋 불러오기)
img_x = np.load("datasets/resized_img_180x120_X.npy")
img_y = np.load("datasets/resized_img_180x120_Y.npy")
# img_y 의 경우 class label인 0, 1, 2, ... 형태로 되어 있음
# CNN을 keras 를 이용하여 학습시, one-hot encoding이 되어야 함
# 예를 들어, 0,1,2 class label이 있을 때, 0은 [1,0,0]으로 변환, 1은 [0,1,0]으로 변환 ...
# 이것을 자동으로 도와주는 코드가 바로 아래 코드
img_y_cat = keras.utils.to_categorical(img_y, num_classes)

