__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

# Keras를 이용한 CNN training 예시 코드를 담고 있다.
# training data 대상은 이미지 데이터 이다.
# time_stamp를 찍어 result파일에 반영하는 코드가 포함되어 있다.
# class label을 one-hot encoding 하는 코드가 들어 있다.
# sklearn을 이용하여 데이터셋을 train, valid, test 로 나누는 코드가 포함되어 있다.
# GPU가 여러개일 경우 1개만을 할당하여 사용하는 코드가 포함됨 (Keras backend가 tensorflow를 사용할 경우)
# CNN 네트워크 구조는 다음과 같이 설계되어 있다.
"""
Convolutional (feature map=32) - Convolutional (feature map=32) - Maxpooling - 
Convolutional (feature map=64) - Convolutional (feature map=64) - Maxpooling - 
Fully Connected (node=512) - Output (node=4)
 
Receptive field = 3 x 3,  Pooling field = 2 x 2
Dropout과 ReLU 적용
"""
# data augmentation 을 Keras 에서 제공하는 ImageDataGenerator 를 사용하는 코드가 포함되어 있다.
# 주석처리 되어 있지만 필요시 기존에 학습된 모델 weight를 불러들일 수 있는 코드가 포함되어 있다.



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


# split the data to train, val, test
train_x, test_x, train_y, test_y  = train_test_split(img_x, img_y_cat, test_size=0.3) #(split할 데이터의 x,y, split비율)
val_x, test_x, val_y, test_y  = train_test_split(test_x, test_y, test_size=0.5)


## keras에서 tensorflow를 backend로 사용시 GPU 할당에 있어 하나의 GPU만 사용하도록 설정할 필요가 있다.
## 단, PC에 GPU가 2개 이상 있는 경우에 해당

config = tf.ConfigProto() #tensorflow의 configuration 을 위한 class 할당
# config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 설정시 1개의 GPU에 대해 할당할 메모리 비율을 설정할 수 있다.
config.gpu_options.visible_device_list="1" # 할당할 GPU 번호 설정, 통상적으로 0 부터 시작한다.

set_session(tf.Session(config=config)) # keras 함수로, keras backend가 config 변수가 가지고 있는 설정을 사용할 수 있도록 한다.


## 모델 선언

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=img_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# initiate RMSprop optimizer
lr = 0.0001
opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)


# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# loss function은 다중 분류 모델이기 때문에 'categorical_crossentropy'를 사용하였다.
# metrics 파라미터에는 'accuracy', ... 등을 전달하여 기록하고자 하는 수치를 지정할 수 있다.


## best model 학습 과정 중에 저장
model_save_path = "results2/model_"+time_stamp+"_epoch{epoch:04d}_valL_{val_loss:.5f}_valA_{val_acc:.5f}.hdf5" # 저장할 파일 이름 설정 (time stamp 이용)
checkpoint = ModelCheckpoint(model_save_path, monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=True)
# checkpoint 방법: 저장파일이름, 모니터링 수치, 정보 출력 정도(높을 수록 출력 정보가 상세해진다), best 모델만을 저장할지 여부 선택, 웨이트만 저장할지 여부 선택
# 웨이트만 저장할 경우, 모델 형태를 별도로 코드로 다시 설계해 줘야 한다. 대신 모델을 모두 저장하는 것 보다 파일 용량이 절약된다.


# This will do preprocessing and realtime data augmentation:
data_aug = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=180,
                            width_shift_range=0.,
                            height_shift_range=0.,
                            shear_range=0.,
                            zoom_range=0.,
                            channel_shift_range=0.,
                            fill_mode='nearest',
                            cval=0.,
                            horizontal_flip=True,
                            vertical_flip=True,
                            rescale=None,
                            preprocessing_function=None,
                            data_format="channels_last")



# model.load_weights("results2/model_20170611_1602_epoch0002_valL_0.39835_valA_0.90625.hdf5") #[0.51194879445400865, 0.86813188123179008]


## 설계된 모델을 학습한다. Data Augmentation이 자동 수행되어 학습을 진행한다.
# Augmentation 된 데이터를 파일로 저장하지 않고 학습 진행
history = model.fit_generator(data_aug.flow(train_x, train_y, batch_size=batch_size, shuffle=True), steps_per_epoch=train_y.shape[0] // batch_size, epochs=epochs, validation_data=data_aug.flow(val_x, val_y, batch_size=batch_size, shuffle=True), validation_steps=val_y.shape[0] // batch_size, verbose=2, callbacks=[checkpoint])

# Augmentation 된 데이터를 파일로 저장하고 학습 진행
# model.fit_generator(data_aug.flow(train_x, train_y, batch_size=batch_size, shuffle=True, save_to_dir="datasets/train", save_format="png"), steps_per_epoch=train_y.shape[0] // batch_size, epochs=epochs, validation_data=data_aug.flow(test_x, test_y, batch_size=batch_size, shuffle=True, save_to_dir="datasets/val", save_format="png"), validation_steps=test_y.shape[0] // batch_size, verbose=2)


## 판다스를 이용하여 모델이 학습 완료된 이후, history 변수에 저장된 수치 기록들을 csv 파일로 저장 가능하다. (정확도, loss 등)
pandas.DataFrame(history.history).to_csv("results2/model_{}.csv".format(time_stamp))