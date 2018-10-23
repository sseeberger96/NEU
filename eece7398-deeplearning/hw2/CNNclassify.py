#
#   This program designs a neural network for classifying the image data in the CIFAR-10 dataset
#   The neural network is designed to have five hidden layers. The structure of the network is as follows.. 
#       - Layer 1: conv(5x5x32) stride = 1 --> batch norm. --> dropout(0.8 keep prob) --> ReLU --> max pooling(2x2) stride = 2
#       - Layer 2: conv(3x3x64) stride = 1 --> batch norm. --> dropout(0.8 keep prob) --> ReLU
#       - Layer 3: conv(3x3x128) stride = 1 --> batch norm. --> dropout(0.8 keep prob) --> ReLU
#       - Layer 4: FC(1000 neurons) --> batch norm. --> ReLU 
#       - Layer 5: FC(100 neurons) --> batch norm. --> ReLU
#   The network uses a hinge loss calculation for its loss function, and uses the Adam optimizer for performing training steps
#   The network also makes use of mini-batch training
# 

import sys
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define CIFAR-10 classes
cifar10Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class seeNet (object):
    # Initialize Network Model
    def __init__ (self, training=1):
        nHNeurons = [1000, 100]
        nLabels = 10

        self.x = tf.placeholder(name="x", shape=[None, 32, 32, 3], dtype=tf.float32)
        self.yActual = tf.placeholder(name="yActual", shape=[None], dtype=tf.int64)

        # Define weights and biases
        W = {
        'first_layer': tf.get_variable(name="w1", shape=[5, 5, 3, 32]), 
        'second_layer': tf.get_variable(name="w2", shape=[3, 3, 32, 64]),
        'third_layer': tf.get_variable(name="w3", shape=[3, 3, 64, 128]), 
        'fourth_layer': tf.get_variable(name="w4", shape=[10*10*128, nHNeurons[0]]), 
        'fifth_layer': tf.get_variable(name="w5", shape=[nHNeurons[0], nHNeurons[1]]), 
        'out_layer': tf.get_variable(name="w6", shape=[nHNeurons[1],nLabels])
        }
        b = {
        'first_layer': tf.get_variable(name="b1", shape=[32]), 
        'second_layer': tf.get_variable(name="b2", shape=[64]),
        'third_layer': tf.get_variable(name="b3", shape=[128]),
        'fourth_layer': tf.get_variable(name="b4", shape=nHNeurons[0]),
        'fifth_layer': tf.get_variable(name="b5", shape=nHNeurons[1]),
        'out_layer': tf.get_variable(name="b6", shape=nLabels),
        }

        # First Hidden Layer
        layer_one = tf.nn.conv2d(self.x, W['first_layer'], strides=[1,1,1,1], padding="VALID")
        self.firstConvOutput = layer_one
        layer_one = tf.nn.bias_add(layer_one, b['first_layer'])
        if training == 1:
            mean1, variance1 = tf.nn.moments(layer_one, axes=[0])
            layer_one = tf.nn.batch_normalization(layer_one, mean1, variance1, None, None, 1e-3)
            layer_one = tf.nn.dropout(layer_one, 0.8)
        else: 
            mean1, variance1 = tf.nn.moments(layer_one, axes=[0, 1, 2])
            layer_one = tf.nn.batch_normalization(layer_one, mean1, variance1, None, None, 1e-3)
        layer_one = tf.nn.relu(layer_one)

        layer_one = tf.nn.max_pool(layer_one, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="VALID")

        # Second Hidden Layer
        layer_two = tf.nn.conv2d(layer_one, W['second_layer'], strides=[1,1,1,1], padding="VALID")
        layer_two = tf.nn.bias_add(layer_two, b['second_layer'])
        if training == 1:
            mean2, variance2 = tf.nn.moments(layer_two, axes=[0])
            layer_two = tf.nn.batch_normalization(layer_two, mean2, variance2, None, None, 1e-3)
            layer_two = tf.nn.dropout(layer_two, 0.8)
        else: 
            mean2, variance2 = tf.nn.moments(layer_two, axes=[0, 1, 2])
            layer_two = tf.nn.batch_normalization(layer_two, mean2, variance2, None, None, 1e-3)
        layer_two = tf.nn.relu(layer_two)

        # Third Hidden Layer
        layer_three = tf.nn.conv2d(layer_two, W['third_layer'], strides=[1,1,1,1], padding="VALID")
        layer_three = tf.nn.bias_add(layer_three, b['third_layer'])
        if training == 1:
            mean3, variance3 = tf.nn.moments(layer_three, axes=[0])
            layer_three = tf.nn.batch_normalization(layer_three, mean3, variance3, None, None, 1e-3)
            layer_three = tf.nn.dropout(layer_three, 0.8)
        else: 
            mean3, variance3 = tf.nn.moments(layer_three, axes=[0, 1, 2])
            layer_three = tf.nn.batch_normalization(layer_three, mean3, variance3, None, None, 1e-3)
        layer_three = tf.nn.relu(layer_three)

        layer_three = tf.reshape(layer_three, [-1, 10*10*128])

        # Fourth Hidden Layer
        layer_four = tf.matmul(layer_three, W['fourth_layer'])
        layer_four = tf.nn.bias_add(layer_four, b['fourth_layer'])
        if training == 1:
            mean4, variance4 = tf.nn.moments(layer_four, axes=[0])
            layer_four = tf.nn.batch_normalization(layer_four, mean4, variance4, None, None, 1e-3)
        else: 
            mean4, variance4 = tf.nn.moments(layer_four, axes=[0, 1])
            layer_four = tf.nn.batch_normalization(layer_four, mean4, variance4, None, None, 1e-3)
        layer_four = tf.nn.relu(layer_four)

        # Fifth Hidden Layer
        layer_five = tf.matmul(layer_four, W['fifth_layer'])
        layer_five = tf.nn.bias_add(layer_five, b['fifth_layer'])
        if training == 1:
            mean5, variance5 = tf.nn.moments(layer_five, axes=[0])
            layer_five = tf.nn.batch_normalization(layer_five, mean5, variance5, None, None, 1e-3)
        else: 
            mean5, variance5 = tf.nn.moments(layer_five, axes=[0, 1])
            layer_five = tf.nn.batch_normalization(layer_five, mean5, variance5, None, None, 1e-3)
        layer_five = tf.nn.relu(layer_five)

        # Output Layer
        y_pred = tf.matmul(layer_five,W['out_layer'])
        y_pred = tf.nn.bias_add(y_pred, b['out_layer'])

        # Define loss function and calculate loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(self.yActual, 10), logits=y_pred)
        # Add regularization to loss
        self.meanLoss = tf.reduce_mean(totalLoss) + 1e-5*tf.nn.l2_loss(W['first_layer']) + 1e-5*tf.nn.l2_loss(W['second_layer']) + 1e-5*tf.nn.l2_loss(W['third_layer'])+ 1e-5*tf.nn.l2_loss(W['fourth_layer'])+ 1e-5*tf.nn.l2_loss(W['fifth_layer'])+ 1e-5*tf.nn.l2_loss(W['out_layer'])

        # Define the Adam Optimizer
        adamOptimizer = tf.train.AdamOptimizer(1e-3)
        self.trainStep = adamOptimizer.minimize(self.meanLoss)

        # Determine accuracy
        correctPredictionFlag = tf.equal(tf.argmax(y_pred, 1), self.yActual)
        self.accuracy = tf.reduce_mean(tf.cast(correctPredictionFlag, tf.float32))

        # Determine network predicted class index
        self.prediction = tf.argmax(y_pred, 1)

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Setup directory for saving and loading the model
        if training == 1:
            if tf.gfile.Exists('./model'):
                tf.gfile.DeleteRecursively('./model')
            tf.gfile.MakeDirs('./model')
        else:
            self.saver.restore(self.sess, './model/saved_model')
            print("Model restored from file")


    # Function to train the network
    def train(self, xTrain, yTrain, xTest, yTest, numSteps=1000, batchSize=128):
        print('{0:>7} {1:>12} {2:>12} {3:>12} {4:>12}'.format('Loop', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
        for i in range(numSteps):
            # Shuffle the training data
            datInd = np.arange(xTrain.shape[0])
            np.random.shuffle(datInd)
            xTrain = xTrain[datInd]
            yTrain = yTrain[datInd]

            # Train in batches
            losses = []
            accuracies = []
            for j in range(0, xTrain.shape[0], batchSize):
                xBatch = xTrain[j:j + batchSize]
                yBatch = yTrain[j:j + batchSize]
                trainLoss, trainAccuracy, _ = self.sess.run([self.meanLoss, self.accuracy, self.trainStep], feed_dict={self.x: xBatch, self.yActual: yBatch})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
            avgTrainLoss = sum(losses) / len(losses)
            avgTrainAcc = sum(accuracies) / len(accuracies)

            # Test in batches
            losses = []
            accuracies = []
            for j in range(0, xTest.shape[0], batchSize):
                xBatch = xTest[j:j + batchSize]
                yBatch = yTest[j:j + batchSize]
                testLoss, testAccuracy = self.sess.run([self.meanLoss, self.accuracy], feed_dict={self.x: xBatch, self.yActual: yBatch})
                losses.append(testLoss)
                accuracies.append(testAccuracy)
            avgTestLoss = sum(losses) / len(losses)
            avgTestAcc = sum(accuracies) / len(accuracies)

            # User Log Output
            print('{0:>7} {1:>12.4f} {2:>12.4f} {3:>12.4f} {4:>12.4f}'.format(str(i+1)+"/"+str(numSteps), avgTrainLoss, avgTrainAcc*100, avgTestLoss, avgTestAcc*100))

        # Save the model
        savePath = self.saver.save(self.sess, './model/saved_model')
        print('Model saved in file: {0}'.format(savePath))

    # Testing function to determine the network's prediction for the class of an input image (or set of images)
    def predictOutput(self, imInput):
        pred, convLayer1 = self.sess.run([self.prediction, self.firstConvOutput],feed_dict={self.x: imInput})
        self.makeConvVisualization(convLayer1)
        return pred

    # Function to create the visualization for the result of the first conv layer
    def makeConvVisualization(self, convLayer):
        convLayer = np.squeeze(convLayer)

        # Normalize convolution output to be between 0 and 255 for proper visualization
        for m in range(0,32): 
            convLayer[:,:,m] = convLayer[:,:,m] - np.amin(convLayer[:,:,m])
            convLayer[:,:,m] = convLayer[:,:,m]/(np.amax(convLayer[:,:,m])/255)

        blackImgs = np.zeros((4,28,28))

        formIm = [0, 0, 0, 0, 0, 0] 

        for u in range(0,6): 
            for v in range(0,6): 
                if v == 0:
                    formIm[u] = np.squeeze(convLayer[:,:,(6*u)+v])
                else: 
                    if (6*u)+v > 31: 
                        formIm[u] = np.append(formIm[u], np.zeros((28,28)), axis=1)
                    else:
                        formIm[u] = np.append(formIm[u], np.squeeze(convLayer[:,:,(6*u)+v]), axis=1)

        outIm = formIm[0]
        for k in range(1,6):
            outIm = np.append(outIm, formIm[k], axis=0)

        # Write out the visualization image
        cv2.imwrite('CONV_rslt.png', outIm)

    # Function for getting the accuracy of the network when tested with a given set of labeled inputs
    def getAccuracy(self, imInput, imLabel):
        return self.sess.run([self.accuracy],feed_dict={self.x: imInput, self.y: imLabel})

# Function for reading in and preprocessing an input image
def readImage(inputImage, meanValue):
    im = cv2.imread(inputImage)
    imgHeight, imgWidth, _ = im.shape
    im = cv2.resize(im, (32, 32))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.asarray(im)
    im = np.expand_dims(im, axis=0)
    im = im.astype(np.float)
    im -= meanValue
    return im

# Function to import and preprocess the CIFAR-10 dataset
def getCifar10():
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    # Format input data
    xTrain = xTrain.astype(np.float)
    xTest = xTest.astype(np.float)
    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)

    # Normalize the data by subtracting the mean image
    meanImage = np.mean(xTrain, axis=0)
    xTrain -= meanImage
    xTest -= meanImage

    return (xTrain, yTrain), (xTest, yTest), meanImage

# Main function
if __name__ == '__main__':
    if len(sys.argv) > 1: 
        if sys.argv[1] == 'train':
            network = seeNet(training=1)
            (xDataTrain, yDataTrain), (xDataTest, yDataTest), meanIm = getCifar10()
            network.train(xDataTrain, yDataTrain, xDataTest, yDataTest, numSteps=25)
        elif sys.argv[1] == 'test':
            network = seeNet(training=0)
            (xDataTrain, yDataTrain), (xDataTest, yDataTest), meanIm = getCifar10()
            predictedClass = cifar10Classes[np.squeeze(network.predictOutput(readImage(sys.argv[2], meanIm)))]
            print("\nThe network predicts that this image is of a:  %s\n" % predictedClass)

    else:
        print("\nNot a valid argument, please use an argument in one of the following formats...")
        print("python CNNclassify.py train\npython CNNclassify.py test xxx.png\n")



