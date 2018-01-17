__author__ = 'Sejune Cheon'
__version__ = 1.0

import tensorflow as tf

## tensorflow gpu 할당 tip
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.visible_device_list = '1'
tfconfig.gpu_options.allow_growth = True

## session 할당
sess = tf.Session(config=tfconfig)
