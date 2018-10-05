import tensorflow as tf
from keras.datasets import cifar10 
import cv2
import numpy as np
import sys



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # TBD: Consider preprocessing data to be between 0 and 1
# cv2.imshow('I did it!', x_train[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # print(x_train[4])

# print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_train = x_train.astype('float32')
x_train = x_train / 255.0
# print(x_train[1])
# print(x_train.shape[1])
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_train = np.squeeze(y_train)
# print(y_train[1])
y_test = np.squeeze(y_test)
n_input = int(x_train.shape[1])
n_neurons_second = 1000
n_neurons = 2000
n_labels = 10




def configure_layers(n_input, n_neurons, n_labels):
	x = tf.placeholder(tf.float32, [None, n_input]) 
	y_actual = tf.placeholder(tf.int64, [None])
	W = {
		'first_layer': tf.Variable(tf.random_normal([n_input, n_neurons])), 
		'second_layer': tf.Variable(tf.random_normal([n_neurons,n_neurons_second])),
		'out_layer': tf.Variable(tf.random_normal([n_neurons_second,n_labels]))
	}
	b = {
		'first_layer': tf.Variable(tf.zeros([n_neurons])), 
		'second_layer': tf.Variable(tf.zeros([n_neurons_second])),
		'out_layer': tf.Variable(tf.zeros([n_labels]))
	}

	return x, y_actual, W, b

def make_model(x, W, b):
	first = tf.matmul(x, W['first_layer']) + b['first_layer']
	mean1, variance1 = tf.nn.moments(first,axes=[0])
	layer_one = tf.nn.relu(tf.nn.batch_normalization(first,mean1,variance1,None,None,1e-12))

	second = tf.matmul(layer_one, W['second_layer']) + b['second_layer']
	mean2, variance2 = tf.nn.moments(second,axes=[0])
	layer_two = tf.nn.relu(tf.nn.batch_normalization(second,mean2,variance2,None,None,1e-12))
	y_pred = tf.matmul(layer_two,W['out_layer']) + b['out_layer']

	return y_pred

def get_loss(y_actual, y_pred):
	cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_actual, n_labels), logits=y_pred)) 
	return cross_entropy

def get_accuracy(y_actual,y_pred):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_actual)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

	# pred_labels = tf.cast(tf.argmax(y_pred,1), tf.int32) 
	# correct_prediction = tf.equal(pred_labels, tf.argmax(y_actual,1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy


def train():
	x, y_actual, W, b = configure_layers(n_input, n_neurons, n_labels)
	y_pred = make_model(x,W,b)
	loss = get_loss(y_actual,y_pred)
	accuracy = get_accuracy(y_actual,y_pred)

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(700):
		s = np.arange(x_train.shape[0]) 
		np.random.shuffle(s)
		xTr = x_train[s]
		yTr = y_train[s]
		batch_xs = xTr[:100]
		batch_ys = yTr[:100]
		loss_out, step, acc = sess.run([loss, train_step,accuracy], feed_dict={x: batch_xs, y_actual: batch_ys}) 
		if (i % 100) == 0: 
			print("Loss: ")
			print(loss_out)
			print("Training Accuracy: ")
			print(acc)
		# print(step)
	print("Testing Accuracy")
	print(sess.run(accuracy, feed_dict={x: x_test, y_actual: y_test}))



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

