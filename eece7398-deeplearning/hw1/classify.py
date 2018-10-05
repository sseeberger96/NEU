import tensorflow as tf
from keras.datasets import cifar10 
import cv2
import numpy as np
import sys


def train():

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# cv2.imshow('I did it!', x_train[4])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	print(x_train.shape)
	x_train = np.reshape(x_train, (x_train.shape[0], -1))
	print(x_train.shape)
	x_test = np.reshape(x_test, (x_test.shape[0], -1))
	y_train = np.squeeze(y_train)
	y_test = np.squeeze(y_test)

	












def predict():
	pass






if __name__ == '__main__':
	if sys.argv[1] == 'train':
		train()
	elif sys.argv[1] == 'predict':
		predict()
	else:
		print("\nNot a valid argument, please use an argument in one of the following formats...")
		print("python classify.py train\npython classify.py predict xxx.png\n")

