__author__ = 'Sejune Cheon'
__version__ = 1.0

import tensorflow as tf

## tensorflow gpu 할당 tip (멀티 gpu 사용시)
config = tf.ConfigProto() #tensorflow의 configuration 을 위한 class 할당
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 설정시 1개의 GPU에 대해 할당할 메모리 비율을 설정할 수 있다.
config.gpu_options.visible_device_list="0" # 할당할 GPU 번호 설정, 통상적으로 0 부터 시작한다. / 0과 1번 GPU를 할당하고 싶으면 "0, 1" 으로 입력해주면 된다.
config.gpu_options.allow_growth = True # 처음 부터 GPU 메모리 할당을 전부 하는 것이 아니라, 필요에 따라서 메모리 할당량을 조절할 수 있다.

## session 할당
sess = tf.Session(config=config)
