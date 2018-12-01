""" 
This file was taken directly from the the neural chatbot model created by Chip Huyen, which can be found here... 

https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot

This chatbot model itself was based off of the Google Translate Tensorflow model, which
is cited as follows... 

https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

The parameters in this file were simply modified to meet the requirements of the neural machine translator. 
Therefore, some parameters were deleted from the original file, some were added, and others simply had their 
values changed. 
"""

# parameters for processing the dataset
PROCESSED_PATH = 'data'
MODEL_PATH = 'model'

THRESHOLD = 1

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 256

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 18777

MAX_ITERATION = 31401

ENC_VOCAB = 41303
DEC_VOCAB = 18778


