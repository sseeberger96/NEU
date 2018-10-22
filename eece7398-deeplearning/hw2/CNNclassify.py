#
#   This program designs a neural network for classifying the image data in the CIFAR-10 dataset
# 	The neural network is designed to have two hidden layers, with 2000 neurons and 1000 neurons respectively
#	The network uses a softmax cross-entropy loss calculation for its loss function, and uses the Adam optimizer for performing training steps
#	The network also makes use of mini-batch training, batch normalization, and the ReLU activation function to perform training
#	NOTE: This neural network only uses a linear classifier between layers. No convolutional layers were utilized. 
#

import tensorflow as tf
from keras.datasets import cifar10 
import cv2
import numpy as np
import sys
import os


# x = tf.ones([3,3,3])
# x = tf.reshape(x,[-1,3,3,3])

# W = tf.ones([3,3,3,1])*2

# o = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# print(sess.run(x))
# print(x.shape)
# print(sess.run(W))
# print(W.shape)
# print(sess.run(o))


# Load in the CIFAR-10 training data and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Code for writing certain test images out to png images for later image testing
# cv2.imwrite('ex1.png', x_test[0])
# cv2.imwrite('ex2.png', x_test[5001])
# cv2.imwrite('ex3.png', x_test[10])
# cv2.imwrite('ex4.png', x_test[9000])
# cv2.imwrite('ex5.png', x_test[123])

# Reshape the training images into 1 x 3072 vectors, make the data type of the image data float32, 
# and normalize the image data points to be between 0 and 1
# x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_train = x_train.astype('float32')
x_train = x_train / 255.0

# Reshape the test images into 1 x 3072 vectors, make the data type of the image data float32, 
# and normalize the image data points to be between 0 and 1
# x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_test = x_test.astype('float32')
x_test = x_test / 255.0

# Reshape the training labels and test labels data to be vectors
# The training labels will be a 50000 x 1 vector and the test labels will be a 10000 x 1 vector
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# Define the number of input neurons as being equal to the vector size of one image (here, it will always be 3072)
n_input = 0

# Define the number of neurons in the first hidden layer
n_neurons = 1000

# Define the number of neurons in the second hidden layer
n_neurons_second = 1000

# Define the number of labels for classifying
n_labels = 10

# Define the directory and filename for saving the model
cur_dir = os.getcwd()
model_dir = str(cur_dir) + "/model/saved_model"



# Function to initialize the input data placeholder, true label placeholder, 
# first hidden layer neurons, second hidden layer neurons, and output layer neurons 
def configure_layers(n_input, n_neurons, n_labels):
	x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x") 
	y_actual = tf.placeholder(tf.int64, [None], name="y_actual")
	W = {
		'first_layer': tf.Variable(tf.random_normal([5, 5, 3, 32]), name="w1"), 
		'second_layer': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="w2"),
		'third_layer': tf.Variable(tf.random_normal([5, 5, 64, 128]), name="w3"), 
		'fourth_layer': tf.Variable(tf.random_normal([20*20*128, n_neurons]), name="w4"), 
		'fifth_layer': tf.Variable(tf.random_normal([n_neurons, n_neurons_second]), name="w5"), 
		'out_layer': tf.Variable(tf.random_normal([n_neurons_second,n_labels]), name="w6")
	}
	b = {
		'first_layer': tf.Variable(tf.zeros([32]), name="b1"), 
		'second_layer': tf.Variable(tf.zeros([64]), name="b2"),
		'third_layer': tf.Variable(tf.zeros([128]), name="b3"),
		'fourth_layer': tf.Variable(tf.zeros([n_neurons]), name="b4"),
		'fifth_layer': tf.Variable(tf.zeros([n_neurons_second]), name="b5"),
		'out_layer': tf.Variable(tf.zeros([n_labels]), name="b6"),
	}

	return x, y_actual, W, b


# Function to make the actual neural network model 
# Performs the linear classification between layers (the matrix multiplication y = Wx + b)
# Also performs batch normalization and applies the ReLU activation function in each hidden layer
def make_model(x, W, b):

	first = tf.nn.conv2d(x, W['first_layer'], strides=[1,1,1,1], padding="VALID")
	first = tf.nn.bias_add(first, b['first_layer'])
	mean1, variance1 = tf.nn.moments(first,axes=[0])
	layer_one = tf.nn.relu(tf.nn.batch_normalization(first,mean1,variance1,None,None,1e-12))

	second = tf.nn.conv2d(layer_one, W['second_layer'], strides=[1,1,1,1], padding="VALID")
	second = tf.nn.bias_add(second, b['second_layer'])
	mean2, variance2 = tf.nn.moments(second,axes=[0])
	layer_two = tf.nn.relu(tf.nn.batch_normalization(second,mean2,variance2,None,None,1e-12))

	third = tf.nn.conv2d(layer_two, W['third_layer'], strides=[1,1,1,1], padding="VALID")
	third = tf.nn.bias_add(third, b['third_layer'])
	mean3, variance3 = tf.nn.moments(third, axes=[0])
	layer_three = tf.nn.relu(tf.nn.batch_normalization(third, mean3, variance3, None,None,1e-12))

	layer_three = tf.reshape(layer_three, [-1, 20*20*128])

	fourth = tf.matmul(layer_three, W['fourth_layer'])
	fourth = tf.nn.bias_add(fourth, b['fourth_layer'])
	mean4, variance4 = tf.nn.moments(fourth,axes=[0])
	layer_four = tf.nn.relu(tf.nn.batch_normalization(fourth, mean4, variance4, None,None,1e-12))

	fifth = tf.matmul(layer_four, W['fifth_layer'])
	fifth = tf.nn.bias_add(fifth, b['fifth_layer'])
	mean5, variance5 = tf.nn.moments(fifth, axes=[0])
	layer_five = tf.nn.relu(tf.nn.batch_normalization(fifth, mean5, variance5, None,None,1e-12))

	y_pred = tf.matmul(layer_five,W['out_layer'])
	y_pred = tf.nn.bias_add(y_pred, b['out_layer'])

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
	# train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

	# Use the adam optimizer to determine in what direction and by how much to  
	# adjust the weights and biases for the next training step
	train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

	# Initialize and run the Tensorflow session
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Print the output header
	print("\nLoop | Training Loss | Training Accuracy (%) | Test Set Loss | Test Set Accuracy (%)")
	print("------------------------------------------------------------------------------------")

	# Perform 2000 training iterations
	for i in range(2001):
		# Shuffle the training data before each iteration to always pick a random batch of 128 
		# images from the training data 
		shuff = np.arange(x_train.shape[0]) 
		np.random.shuffle(shuff)
		xTr = x_train[shuff]
		yTr = y_train[shuff]
		batch_xs = xTr[:128]
		batch_ys = yTr[:128]

		# Run the batch through the neural network, and return the loss, training step, and accuracy of the network for that batch
		loss_out, step, acc = sess.run([loss, train_step, accuracy], feed_dict={x: batch_xs, y_actual: batch_ys}) 

		# Consider every 100 iterations to be a training "epoch"
		# At this point, test the neural network on all of the testing data
		# Print both the training and testing results to the user
		if (i % 100) == 0: 
			test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: x_test, y_actual: y_test})
			print(" " + str(int((i / 100) + 1)) + "      " + str(loss_out) + "            " + str(("{0:.3f}".format(acc * 100))) + "              " + str(test_loss) + "         " + str(("{0:.3f}".format(test_acc * 100)))) 
	 	
	# Save the model to the model folder 
	saver = tf.train.Saver()
	save_path = saver.save(sess, model_dir)	


# Main function for testing the neural network 
def test(image_path):
	# Create a new Tensorflow session 
	sess = tf.InteractiveSession()
	# Import the saved graph
	saver = tf.train.import_meta_graph(str(cur_dir) + "/model/saved_model.meta")
	# Restore the variable values from the last saved checkpoint
	saver.restore(sess,tf.train.latest_checkpoint(str(cur_dir) + "/model/"))

	# Create the graph for calling back placeholders and variables  
	graph = tf.get_default_graph()

	# Call back x and y_actual placeholders
	x = graph.get_tensor_by_name("x:0")
	y_actual = graph.get_tensor_by_name("y_actual:0")

	# Call back individual layer weight matrix variables
	w1 = graph.get_tensor_by_name("w1:0")
	w2 = graph.get_tensor_by_name("w2:0")
	w3 = graph.get_tensor_by_name("w3:0")

	# Call back individual layer bias vector variables
	b1 = graph.get_tensor_by_name("b1:0")
	b2 = graph.get_tensor_by_name("b2:0")
	b3 = graph.get_tensor_by_name("b3:0")

	# Recreate the weight and bias dictionaries using the individual layer call backs
	W = {'first_layer': w1, 'second_layer': w2, 'out_layer': w3}
	b = {'first_layer': b1, 'second_layer': b2, 'out_layer': b3}

	# Define method for making and applying the neural network model using the saved variables and recalled placeholders
	y_pred = make_model(x,W,b)
	# Define method for returning the loss of the saved neural network's results
	loss = get_loss(y_actual,y_pred)
	# Define method for returning the accuracy of the saved neural network's results
	accuracy = get_accuracy(y_actual,y_pred)

	print("Model restored\n")

	if image_path == None: 
		# Run the full test data set through the saved (i.e. already trained) neural network 
		test_loss, test_acc, y_pred = sess.run([loss, accuracy, y_pred], feed_dict={x: x_test, y_actual: y_test})
		
		# Print Results to the user
		print("Full Test Set Loss:         " + str(test_loss))
		print("Full Test Set Accuracy (%): " + str(("{0:.3f}".format(test_acc * 100))) + "\n")
	else: 
		test_image = cv2.imread(image_path,1)
		# cv2.imshow('image',test_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		test_image = np.reshape(test_image, (1, -1))
		test_image = test_image.astype('float32')
		test_image = test_image / 255.0

		pred = sess.run(tf.argmax(y_pred, 1), feed_dict={x: test_image})
		# print(pred)
		pred_index = int(pred[0])

		label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

		print("The neural network predicts that the input image is a....  " + label_list[pred_index] + "\n")
		print("NOTE: This function does not seem to be working properly. However, as it was no longer required for the homework assignment, I did not figure out a way to fix it.\n")



if __name__ == '__main__':
	if len(sys.argv) > 1: 
		if sys.argv[1] == 'train':
			train()
		elif sys.argv[1] == 'test':
			if len(sys.argv) == 3: 
				test(sys.argv[2])
			else: 
				test(None)
	else:
		print("\nNot a valid argument, please use an argument in one of the following formats...")
		print("python classify.py train\npython classify.py test\n")

