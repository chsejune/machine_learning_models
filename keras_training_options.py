__author__ = 'Sejune Cheon'

# 개발환경: PYTHON3

## keras 를 이용한 네트워크 학습 과정 중
## 학습 내용 기록, weight 저장, early stop 등
## 학습 옵션에 대해 정리해 본다.
## keras 에선 이를 Callback 이라 한다.

# import library
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import time
import numpy as np
import pandas

# 모델 학습 과정 기록을 위한 파일 생성시 time stamp 를 기준으로 파일을 만들어 주면 관리가 편하다.
time_stamp = time.strftime("%Y%m%d_%H%M")  #YYYYmmdd_HHMM 형태로 입력됨 (ex. 20170611_1602)

# keras가 포함하고 있는 데이터 샘플 중 하나인 MNIST 데이터셋을 로딩한다. (컴퓨터에 파일이 없으면 스스로 다운로드 한다.)
# 다운로드 경로는 보통 : C:\Users\user_id\.keras\datasets 이다.
(trainX, trainY), (testX, testY) = mnist.load_data()

# keras는 Y 값이 classification 목적일 경우 one hot encoding 형태로 바꿔주는 작업이 필요하다.
# np.unique() 함수를 이용해서 Y 데이터셋이 포함하고 있는 분류 항목 개수를 파악하였다.
trainX = trainX.reshape((-1, 28*28))
testX = testX.reshape((-1, 28*28))
trainY = keras.utils.to_categorical(trainY, np.unique(trainY).shape[0])
testY = keras.utils.to_categorical(testY, np.unique(testY).shape[0])


## 간단한 샘플 모델 선언
# 레이어 설계
l_input = Input(shape=(28*28,))
l_hidden = Dense(units=50, activation='relu')(l_input)
l_output = Dense(units=10, activation='softmax')(l_hidden)

# 모델 정의
model = Model(inputs=l_input, outputs=l_output)

# 모델 컴파일
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])


#### training 과정 기록 및 모니터링을 위한 옵션들... (Call backs) ####

## 모델 체크 포인트
# 중간 중간 best 모델 또는 학습 된 weight들을 저장할 수 있는 옵션 기능이다.
# 먼저 모델을 저장할 파일명을 정의한다. 정의 과정에 저장할 경로도 같이 설정한다.
model_save_path = "results/model_"+time_stamp+"_epoch{epoch:04d}_trainingL_{loss:.5f}.hdf5" # 저장할 파일 이름 설정 (time stamp 이용)
# {epoch:04d} - 파일명에 현재 epoch 을 반영할 수 있다.
# {val_loss:.5f} - validation loss 반영이 가능하다. (':' 뒤는 파일명에 반영될 숫자에 대한 포멧팅 형식이다. / 소숫점 다섯째 짜리 까지 반영)
# {val_acc:.5f} - validation accuracy 반영도 가능하다.
# {loss:.5f} - training loss
# {acc:.5f} - training accuracy

checkpoint = ModelCheckpoint(filepath=model_save_path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
# monitor - 학습 과정 중 중간 model weight 또는 모델을 저장 시 모니터링 할 값을 설정한다.
# verbose: verbosity mode, 0 or 1. (저장 되었음을 알려주는 로그를 출력할지 말지 결정한다.)


##


## 설정한 training 옵션(Callback)들을 반영하기 위해선 "callbacks" 파라미터에 list에 변수를 담아 모두 전달해 줘야 한다.
history = model.fit(trainX, trainY, epochs=20, verbose=2, validation_data=(testX, testY), callbacks=[checkpoint])


## 모델 학습이 완료된 후 history 기록을 하고 싶을 때
## 판다스를 이용하여 모델이 학습 완료된 이후, history 변수에 저장된 수치 기록들을 csv 파일로 저장 가능하다.
# (정확도, loss 등, compile 시 metrics 파라미터에 전달한 값들이 기록된다.)
pandas.DataFrame(history.history).to_csv("results/model_{}.csv".format(time_stamp))

