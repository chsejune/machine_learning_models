__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

## import libraries
from keras.models import Sequential
from keras.models import load_model

## 현재 설계 완료된 또는 학습 완료된 모델이 있다고 가정 할때,
model = Sequential() # sequential 모델임을 가정 (sequential 아니여도 상관 없음)

# weight 저장법:
model.save_weights("filepath")

# model 설계구조+weight 저장법:
model.save("filepath")

# weight 불러오기:
# (weight 만 불러오기 때문에 모델 설계가 저장된 weight의 모델과 동일해야 한다. 모델 설계가 별도로 선행되어야 함)
# partial weight 불러오기는 "keras_cnn_load_partial_weights.py" 파일 참조
model.load_weights("filepath")

# model 설계구조+weight 불러오기:
model = load_model("filepath")




