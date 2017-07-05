__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

# keras 모델에서 중간단의 layer output 값을 출력하는 방법
# 이미 기존 설계 및 학습 완료된 모델이 있다고 가정
# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer 사이트 참조

# import library
import keras.backend as kb
from keras.models import load_model

# 모델 불러오기
model = load_model("filepath")

# intermediate layer의 output을 뽑기 위한 함수 만들기
get_feature_map = kb.function([model.layers[0].input, kb.learning_phase()], [model.layers[4].output])

# "learning_phase" is needed for dropout and batchnormalizaiton use.
# select option 0 - test mode, 1 - training mode


