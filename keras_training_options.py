__author__ = 'Sejune Cheon'

# 개발환경: PYTHON3

## keras 를 이용한 네트워크 학습 과정 중
## 학습 내용 기록, weight 저장, early stop 등
## 학습 옵션에 대해 정리해 본다.

# import library
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import time

# 모델 학습 과정 기록을 위한 파일 생성시 time stamp 를 기준으로 파일을 만들어 주면 관리가 편하다.
time_stamp = time.strftime("%Y%m%d_%H%M")  #YYYYmmdd_HHMM 형태로 입력됨 (ex. 20170611_1602)


## 간단한 샘플 모델 선언
# 레이어 설계
l_input = Input(shape=(20,))
l_hidden = Dense(units=50, activation='relu')(l_input)
l_output = Dense(units=5, activation='softmax')(l_hidden)

# 모델 정의
model = Model(inputs=l_input, outputs=l_output)

# 모델 컴파일
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])





