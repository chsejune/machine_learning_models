__author__ = 'Sejune Cheon'

# 개발환경: PYTHON3

## keras 를 이용한 네트워크 학습 과정 중
## 학습 내용 기록, weight 저장, early stop 등
## 학습 옵션에 대해 정리해 본다.
## keras 에선 이를 Callback 이라 한다.

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


#### training 과정 기록 및 모니터링을 위한 옵션들... (Call backs) ####

## 모델 체크 포인트
# 중간 중간 best 모델 또는 학습 된 weight들을 저장할 수 있는 옵션 기능이다.
# 먼저 모델을 저장할 파일명을 정의한다. 정의 과정에 저장할 경로도 같이 설정한다.
model_save_path = "results/SA_pretraining_model_"+time_stamp+"_epoch{epoch:04d}_valL_{val_loss:.5f}.hdf5" # 저장할 파일 이름 설정 (time stamp 이용)
# {epoch:04d} - 파일명에 현재 epoch 을 반영할 수 있다.
# {val_loss:.5f} - validation loss 반영이 가능하다. (':' 뒤는 파일명에 반영될 숫자에 대한 포멧팅 형식이다. / 소숫점 다섯째 짜리 까지 반영)
# {val_acc:.5f} - validation accuracy 반영도 가능하다.

checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)




