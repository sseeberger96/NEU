import sys
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cifar10Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class seeNet (object):
    def __init__ (self, isTrain=1):
        nHNeurons = [1000, 100]
        nLabels = 10


        self.x = tf.placeholder(name="x", shape=[None, 32, 32, 3], dtype=tf.float32)
        self.yActual = tf.placeholder(name="yActual", shape=[None], dtype=tf.int64)



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



        first = tf.nn.conv2d(self.x, W['first_layer'], strides=[1,1,1,1], padding="VALID")
        self.firstConvOutput = first
        first = tf.nn.bias_add(first, b['first_layer'])
        mean1, variance1 = tf.nn.moments(first, axes=[0])
        # scale1 = tf.Variable(tf.ones([1000]))
        # beta1 = tf.Variable(tf.zeros([1000]))
        layer_one = tf.nn.relu(tf.nn.batch_normalization(first, mean1, variance1, None, None, 1e-3))
        layer_one = tf.nn.dropout(layer_one, 0.8)
        layer_one = tf.nn.max_pool(layer_one, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="VALID")

        second = tf.nn.conv2d(layer_one, W['second_layer'], strides=[1,1,1,1], padding="VALID")
        second = tf.nn.bias_add(second, b['second_layer'])
        mean2, variance2 = tf.nn.moments(second, axes=[0])
        # scale2 = tf.Variable(tf.ones([1000]))
        # beta2 = tf.Variable(tf.zeros([1000]))
        layer_two = tf.nn.relu(tf.nn.batch_normalization(second, mean2, variance2, None, None, 1e-3))
        layer_two = tf.nn.dropout(layer_two, 0.8)

        third = tf.nn.conv2d(layer_two, W['third_layer'], strides=[1,1,1,1], padding="VALID")
        third = tf.nn.bias_add(third, b['third_layer'])
        mean3, variance3 = tf.nn.moments(third, axes=[0])
        # scale3 = tf.Variable(tf.ones([1000]))
        # beta3 = tf.Variable(tf.zeros([1000]))
        layer_three = tf.nn.relu(tf.nn.batch_normalization(third, mean3, variance3, None, None, 1e-3))
        layer_three = tf.nn.dropout(layer_three, 0.8)

        layer_three = tf.reshape(layer_three, [-1, 10*10*128])

        fourth = tf.matmul(layer_three, W['fourth_layer'])
        fourth = tf.nn.bias_add(fourth, b['fourth_layer'])
        mean4, variance4 = tf.nn.moments(fourth, axes=[0])
        # scale4 = tf.Variable(tf.ones([1000]))
        # beta4 = tf.Variable(tf.zeros([1000]))
        layer_four = tf.nn.relu(tf.nn.batch_normalization(fourth, mean4, variance4, None, None, 1e-3))

        fifth = tf.matmul(layer_four, W['fifth_layer'])
        fifth = tf.nn.bias_add(fifth, b['fifth_layer'])
        mean5, variance5 = tf.nn.moments(fifth, axes=[0])
        # scale5 = tf.Variable(tf.ones([1000]))
        # beta5 = tf.Variable(tf.zeros([1000]))
        layer_five = tf.nn.relu(tf.nn.batch_normalization(fifth, mean5, variance5, None, None, 1e-3))

        y_pred = tf.matmul(layer_five,W['out_layer'])
        y_pred = tf.nn.bias_add(y_pred, b['out_layer'])


        # Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(self.yActual, 10), logits=y_pred)
        self.meanLoss = tf.reduce_mean(totalLoss) + 1e-5*tf.nn.l2_loss(W['first_layer']) + 1e-5*tf.nn.l2_loss(W['second_layer']) + 1e-5*tf.nn.l2_loss(W['third_layer'])+ 1e-5*tf.nn.l2_loss(W['fourth_layer'])+ 1e-5*tf.nn.l2_loss(W['fifth_layer'])+ 1e-5*tf.nn.l2_loss(W['out_layer'])

        # Optimizer
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.trainStep = optimizer.minimize(self.meanLoss)

        # Correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(y_pred, 1), self.yActual)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Predict index
        self.predict = tf.argmax(y_pred, 1)

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if isTrain == 1:
            if tf.gfile.Exists('./model'):
                tf.gfile.DeleteRecursively('./model')
            tf.gfile.MakeDirs('./model')
        else:
            self.saver.restore(self.sess, './model/model.ckpt')


    def train(self, xTr, yTr, xTe, yTe, maxSteps=1000, batchSize=128):
        print('{0:>7} {1:>12} {2:>12} {3:>12} {4:>12}'.format('Loop', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
        for i in range(maxSteps):
            # Shuffle data
            s = np.arange(xTr.shape[0])
            np.random.shuffle(s)
            xTr = xTr[s]
            yTr = yTr[s]

            # Train
            losses = []
            accuracies = []
            for j in range(0, xTr.shape[0], batchSize):
                xBatch = xTr[j:j + batchSize]
                yBatch = yTr[j:j + batchSize]
                trainLoss, trainAccuracy, _ = self.sess.run([self.meanLoss, self.accuracy, self.trainStep], feed_dict={self.x: xBatch, self.yActual: yBatch})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
            avgTrainLoss = sum(losses) / len(losses)
            avgTrainAcc = sum(accuracies) / len(accuracies)

            # Test
            losses = []
            accuracies = []
            for j in range(0, xTe.shape[0], batchSize):
                xBatch = xTe[j:j + batchSize]
                yBatch = yTe[j:j + batchSize]
                testLoss, testAccuracy = self.sess.run([self.meanLoss, self.accuracy], feed_dict={self.x: xBatch, self.yActual: yBatch})
                losses.append(testLoss)
                accuracies.append(testAccuracy)
            avgTestLoss = sum(losses) / len(losses)
            avgTestAcc = sum(accuracies) / len(accuracies)

            # Log Output
            print('{0:>7} {1:>12.4f} {2:>12.4f} {3:>12.4f} {4:>12.4f}'.format(str(i+1)+"/"+str(maxSteps), avgTrainLoss, avgTrainAcc*100, avgTestLoss, avgTestAcc*100))

        savePath = self.saver.save(self.sess, './model/model.ckpt')
        print('Model saved in file: {0}'.format(savePath))

    def predictOutput(self, inputX):
        
        pred, convLayer1 = self.sess.run([self.predict, self.firstConvOutput],feed_dict={self.x: inputX})
        convLayer1 = np.squeeze(convLayer1)
        blackImgs = np.zeros((4,28,28))

        formIm = [0, 0, 0, 0, 0, 0] 

        for u in range(0,6): 
            for v in range(0,6): 
                if v == 0:
                    formIm[u] = np.squeeze(convLayer1[:,:,(6*u)+v])
                else: 
                    if (6*u)+v > 31: 
                        formIm[u] = np.append(formIm[u], np.zeros((28,28)), axis=1)
                    else:
                        formIm[u] = np.append(formIm[u], np.squeeze(convLayer1[:,:,(6*u)+v]), axis=1)

        # print(formIm)
        outIm = formIm[0]
        for k in range(1,6):
            outIm = np.append(outIm, formIm[k], axis=0)

        cv2.imwrite('ex2.png', outIm)


        return pred

    def getAcc(self, inputX, inputY):
        return self.sess.run([self.accuracy],feed_dict={self.x: inputX, self.y: inputY})

def readImage(inputImage, meanValue):
    img = cv2.imread(inputImage)
    h_img, w_img, _ = img.shape
    imgResize = cv2.resize(img, (32, 32))
    imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
    imgResizeNp = np.asarray(imgRGB)
    imgResizeNp = np.expand_dims(imgResizeNp, axis=0)
    imgResizeNp = imgResizeNp.astype(np.float)
    imgResizeNp -= meanValue
    # imgResizeNp = np.reshape(imgResizeNp, (imgResizeNp.shape[0], -1))
    return imgResizeNp

def getCifar10():
    (x1, y1), (x2, y2) = cifar10.load_data()

    # Format data
    x1 = x1.astype(np.float)
    x2 = x2.astype(np.float)
    y1 = np.squeeze(y1)
    y2 = np.squeeze(y2)

    # Normalize the data by subtract the mean image
    meanImage = np.mean(x1, axis=0)
    x1 -= meanImage
    x2 -= meanImage

    # Reshape data from channel to rows
    # x1 = np.reshape(x1, (x1.shape[0], -1))
    # x2 = np.reshape(x2, (x2.shape[0], -1))

    return (x1, y1), (x2, y2), meanImage

# Main function
if sys.argv[1] == "train":
    classify = seeNet(isTrain=1)
    (xTrain, yTrain), (xTest, yTest), mV = getCifar10()
    classify.train(xTrain, yTrain, xTest, yTest, maxSteps=25)

elif sys.argv[1] == 'test':
    classify = seeNet(isTrain=0)
    (xTrain, yTrain), (xTest, yTest), mV = getCifar10()
    print(cifar10Classes[np.squeeze(classify.predictOutput(readImage(sys.argv[2], mV)))])



