import tensorflow as tf
from keras.datasets import cifar10 
import cv2
import numpy as np
import sys
import os


# Load in the CIFAR10 training data and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshape the training images into 1 x 3072 vectors, make the data type of the image data float32, 
# and normalize the image data points to be between 0 and 1
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_train = x_train.astype('float32')
x_train = x_train / 255.0

# Reshape the test images into 1 x 3072 vectors, make the data type of the image data float32, 
# and normalize the image data points to be between 0 and 1
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_test = x_test.astype('float32')
x_test = x_test / 255.0

# Reshape the training labels and test labels data to be vectors
# The training labels will be a 50000 x 1 vector and the test labels will be a 10000 x 1 vector
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# Define the number of input neurons as being equal to the vector size of one image (here, it will always be 3072)
n_input = int(x_train.shape[1])

# Define the number of neurons in the first hidden layer
n_neurons = 2000

# Define the number of neurons in the second hidden layer
n_neurons_second = 1000

# Define the number of labels for classifying
n_labels = 10

cur_dir = os.getcwd()
model_dir = str(cur_dir) + "/model/saved_model.ckpt"



# Function to initialize the input data placeholder, true label placeholder, 
# first hidden layer neurons, second hidden layer neurons, and output layer neurons 
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


# Function to make the actual neural network model 
# Performs the linear classification between layers (the matrix multiplication y = Wx + b)
# Also performs batch normalization and applies the ReLU activation function in each hidden layer
def make_model(x, W, b):
	first = tf.matmul(x, W['first_layer']) + b['first_layer']
	mean1, variance1 = tf.nn.moments(first,axes=[0])
	layer_one = tf.nn.relu(tf.nn.batch_normalization(first,mean1,variance1,None,None,1e-12))

	second = tf.matmul(layer_one, W['second_layer']) + b['second_layer']
	mean2, variance2 = tf.nn.moments(second,axes=[0])
	layer_two = tf.nn.relu(tf.nn.batch_normalization(second,mean2,variance2,None,None,1e-12))
	y_pred = tf.matmul(layer_two,W['out_layer']) + b['out_layer']

	return y_pred


# Function to return the loss of the network's output when compared to the true labels 
# Uses a softmax approach by computing the cross-entropy loss
def get_loss(y_actual, y_pred):
	cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_actual, n_labels), logits=y_pred)) 
	return cross_entropy


# Function to return the accuracy of the network's output when compared to the true labels 
def get_accuracy(y_actual,y_pred):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_actual)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

	return accuracy


# Main function for training the neural network 
def train():
	# Initialize the layers using the configure_layers function
	x, y_actual, W, b = configure_layers(n_input, n_neurons, n_labels)
	# Make and apply the actual model to determine the predicted output of the network
	y_pred = make_model(x,W,b)
	# Compute the loss of the network's output
	loss = get_loss(y_actual,y_pred)
	# Compute the accuracy of the network's output
	accuracy = get_accuracy(y_actual,y_pred)

	# Use the gradient descent optimizer to determine in what direction and by how much to  
	# adjust the weights and biases for the next training step
	train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)

	# Initialize and run the Tensorflow session
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Print the output header
	print("\nLoop | Training Loss | Training Accuracy (%) | Test Set Loss | Test Set Accuracy")
	print("--------------------------------------------------------------------------------")

	# Perform 1501 training iterations
	for i in range(1501):
		# Shuffle the training data before each iteration to always pick a random batch of 100 
		# images from the training data 
		s = np.arange(x_train.shape[0]) 
		np.random.shuffle(s)
		xTr = x_train[s]
		yTr = y_train[s]
		batch_xs = xTr[:100]
		batch_ys = yTr[:100]

		# Run the batch through the neural network, and return the loss, training step, and accuracy of the network for that batch
		loss_out, step, acc = sess.run([loss, train_step,accuracy], feed_dict={x: batch_xs, y_actual: batch_ys}) 

		# Consider every 100 iterations to be a training "epoch"
		# At this point, test the neural network on all of the testing data
		# Print both the training and testing results to the user
		if (i % 100) == 0: 
			test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: x_test, y_actual: y_test})
			print(" " + str(int((i / 100) + 1)) + "      " + str(loss_out) + "           " + str(("{0:.3f}".format(acc * 100))) + "             " + str(test_loss) + "        " + str(("{0:.3f}".format(test_acc * 100)))) 
		

def predict():
	pass
	# with tf.Session() as sess:


	# 	saver.restore(sess, model_dir)
	# 	print("Model restored.")
	# 	test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: x_test, y_actual: y_test})
	# 	print(test_loss)
	# 	print(test_acc)



if __name__ == '__main__':
	if sys.argv[1] == 'train':
		train()
	elif sys.argv[1] == 'predict':
		predict()
	else:
		print("\nNot a valid argument, please use an argument in one of the following formats...")
		print("python classify.py train\npython classify.py predict xxx.png\n")
		print("NOTE: The 'predict' method is actually not included as updated homework requirements stated that it was not necessary")

