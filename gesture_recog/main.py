import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


from data_loader import *
from nn_def import *


def normalize(x,y):
	# TODO:  check range of IMU values
	x = tf.clip_by_value(x, -80.0, 80.0)
	mean = tf.reduce_mean(x, axis=0, keepdims=True)
	std = tf.math.reduce_std(x, axis=0, keepdims=True) + 1e-6
	x = (x-mean) / std
	return x, y


def filter(): 
	pass


def segment(): 
	pass


def extract_feature():  
	pass


def training_split(): 
	pass


def main(): 
	print('Initializing...\n')

	train_ds, test_ds = build_train_test_datasets(
    	train_subjects=(0,1,2,3,4,5,6),
    	test_subjects=(7,),
    	batch_size=64
	)

	model = build_imu_model(input_shape=(None, 3), num_classes=4)
	model.fit(train_ds, validation_data=test_ds, epochs=20)  # test_ds used as validation here
	model.evaluate(test_ds)

if __name__=='__main__':
	main()

