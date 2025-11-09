import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

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

	train_ds = build_dataset()
	model = build_imu_model()
	model.fit(train_ds, epochs=25)

if __name__=='__main__':
	main()

