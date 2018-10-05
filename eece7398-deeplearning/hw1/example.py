import tensorflow as tf
from keras.datasets import mnist 
import cv2
import numpy as np
import sys


# Import data (Samples, 28, 28, 1)
(xTrain, yTrain), (xTest, yTest) = mnist.load_data() 
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1)) 
xTest = np.reshape(xTest, (xTest.shape[0], -1)) 
yTrain = np.squeeze(yTrain)
yTest = np.squeeze(yTest)

# Create the model
x = tf.placeholder(tf.float32, [None, 784]) 
y_ = tf.placeholder(tf.int64, [None])
# Variables
W = tf.Variable(tf.zeros([784, 10])) 
b = tf.Variable(tf.zeros([10]))
# Output
y = tf.matmul(x, W) + b


# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y)) 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Train
for _ in range(100):
	s = np.arange(xTrain.shape[0]) 
	np.random.shuffle(s)
	xTr = xTrain[s]
	yTr = yTrain[s]
	batch_xs = xTr[:100]
	batch_ys = yTr[:100]
	loss, step = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys}) 
	print(loss)
	# print(step)
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
print(sess.run(accuracy, feed_dict={x: xTest, y_: yTest}))