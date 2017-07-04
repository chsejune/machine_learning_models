__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

## import libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import h5py


## keras 모델에서 기존에 학습했던 weight을 부분 적으로 로딩하여 사용하고 싶을 경우
## 즉 기존 모델에서 일부 layer만 구조 및 조건을 변경하여 사용하고 싶은 경우

## 우선 기존에 학습되었던 모델 설계를 알고 있어야 한다.
## 기존 학습 된 설계 모델

## 기존 모델 선언

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(160, 160, 1)))
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
model.add(Dense(3))
model.add(Activation('softmax'))


# 어떤 형태의 모델이 설계 되었는지 확인
model.summary()

# 아래와 같이 모델이 설계되어 있음을 확인할 수 있으며, 여기서 주요하게 살펴봐야 할 부분은 Layer 열의 이름들이다.
# 예를들어 첫번째 행의 layer 이름은 "conv2d_1" 이 된다.
# 해당 이름은 default 로 부여된 이름이며
# 이름은 사용자가 다른 이름으로 지정 가능하다. (지정 방법은 새로운 모델 설계 부분에서 설명하겠다.)
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 160, 160, 32)      320       
_________________________________________________________________
activation_1 (Activation)    (None, 160, 160, 32)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 158, 158, 32)      9248      
_________________________________________________________________
activation_2 (Activation)    (None, 158, 158, 32)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 79, 79, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 79, 79, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 79, 79, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 79, 79, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 77, 77, 64)        36928     
_________________________________________________________________
activation_4 (Activation)    (None, 77, 77, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 38, 38, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 38, 38, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 92416)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               47317504  
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 1539      
_________________________________________________________________
activation_6 (Activation)    (None, 3)                 0         
=================================================================
Total params: 47,384,035
Trainable params: 47,384,035
Non-trainable params: 0
_________________________________________________________________
"""

# 현재 모델은 3 class 에 대한 분류 모델임을 "dense_2" 레이어를 통해 알 수 있다.
# 만약 동일 모델에서 5 class 에 대한 분류를 수행하는 모델로 변경하고,
# 기존에 사용했던 weight를 기준으로 재학습 하고 싶은 경우
# (기존에 학습해 놓은 weight는 이미 저장되어 있다고 가정)


# 우선 새로운 모델 설계를 해준다.
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(160, 160, 1)))
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
model.add(Dense(3, name="new_class")) ## rlwhs
model.add(Activation('softmax'))



# 기존 모델을 설계 완료 하였으면, 기존 학습 weight를 불러들인다.
model.load_weights()