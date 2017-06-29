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

