__author__ = 'Sejune Cheon'

## 개발환경 Python3.5

# keras 모델에서 중간단의 layer output 값을 출력하는 방법
# 이미 기존 설계 및 학습 완료된 모델이 있다고 가정
# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer 사이트 참조

# import library
import keras.backend as kb
from keras.models import load_model

# 데이터가 trainX 로 load 되어 있다고 가정
trainX = [0,0,0,0]

# 모델 불러오기
model = load_model("filepath")

# intermediate layer의 output을 뽑기 위한 함수 만들기
get_inter_layer_output = kb.function([model.layers[0].input, kb.learning_phase()], [model.layers[4].output])
# "learning_phase" is needed for dropout and batchnormalizaiton use.
# select option 0 - test mode, 1 - training mode

# 사용법
output = get_inter_layer_output([trainX, 0])[0]  # learning phase 값으로 0 또는 1 전달
output = get_inter_layer_output([trainX, 0])[0].squeeze() # 혹시나 의미없는 차원수가 같이 출력될 경우 "squeeze" 함수를 써서 의미없는 차원수를 줄여준다.


"""
How can I obtain the output of an intermediate layer?

One simple way is to create a new Model that will output the layers that you are interested in:
-------------------------------------------------------------------------------------------------------------
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
-------------------------------------------------------------------------------------------------------------


Alternatively, you can build a Keras function that will return the output of a certain layer given a certain input, for example:
-------------------------------------------------------------------------------------------------------------
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
-------------------------------------------------------------------------------------------------------------


Similarly, you could build a Theano and TensorFlow function directly.

Note that if your model has a different behavior in training and testing phase (e.g. if it uses  Dropout, BatchNormalization, etc.), you will need to pass the learning phase flag to your function:
-------------------------------------------------------------------------------------------------------------
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]
-------------------------------------------------------------------------------------------------------------
"""

