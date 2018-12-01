""" This code produces a neural machine translation system, capable of translating 
input English text data into its corresponding Vietnamese text data. 

Note that the code used to produce this model is heavily based off of the neural chatbot
model created by Chip Huyen (much of the code is identical), which can be found here... 
https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot

This chatbot model itself was based off of the Google Translate Tensorflow model, which
is cited as follows... 

https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the main function needed to run the neural machine translator. Options are... 

python NMT.py train 
python NMT.py test 
python NMT.py translate

For training, testing on a full dataset, and translating user input sentences, respectively. 
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import NMTModel
import config
import data

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True in the testing and translating modes. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained model if there is a saved model. """

    checkpoint = tf.train.get_checkpoint_state(os.path.dirname(config.MODEL_PATH + '/checkpoint'))
    if os.path.isfile(os.path.join(config.MODEL_PATH, 'checkpoint')):
        print("Loading saved NMT model from folder... ")
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Model loaded successfully!")
    else: 
        print("No saved model. Initializing a new model.")

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    # print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def getBLEU(ref, hyp):
    """ Function to get the BLEU score for a single testing iteration
    Returns the BLEU score for a given reference sentence (ref) and hypothesis 
    sentence output by the neural network (hyp)
    Utilizes smoothing method 1
    """
    smoothingFunc = SmoothingFunction()
    bleuScore = sentence_bleu(ref, hyp, smoothing_function=smoothingFunc.method1)
    return bleuScore

def train():
    """ 
    Function to train the NMT, given the English and Vietnamese training datasets
    """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # In training mode, we need to create the backward path, so forwrad_only is False
    model = NMTModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        iteration = model.global_step.eval()
        total_loss = 0
        print('Training... ')
        while iteration < config.MAX_ITERATION:
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % 100 == 0:
                if tf.gfile.Exists(config.MODEL_PATH):
                    tf.gfile.DeleteRecursively(config.MODEL_PATH)
                tf.gfile.MakeDirs(config.MODEL_PATH)
                saver.save(sess, os.path.join(config.MODEL_PATH, 'saved_model'), global_step=model.global_step)

            if iteration % 1000 == 0:
                print('Iteration {}: Loss {}, Time {}'.format(iteration, total_loss/1000, time.time() - start))
                start = time.time()
                total_loss = 0

                sys.stdout.flush()


def test():
    """ 
    Function to test the NMT, given the English and Vietnamese test datasets
    Calculates the average BLEU score for the whole test dataset
    """
    inv_enc_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    enc_file = open(os.path.join(config.PROCESSED_PATH, 'test_ids.enc'), 'r')
    dec_file = open(os.path.join(config.PROCESSED_PATH, 'test_ids.dec'), 'r')

    enc_lines = enc_file.read().splitlines()
    dec_lines = dec_file.read().splitlines()


    model = NMTModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()
    bleuScores = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        max_length = config.BUCKETS[-1][0]
        print("Testing full test dataset... ")
        for i in range(len(enc_lines)):
            encode = [int(id_) for id_ in enc_lines[i].split()]
            target = [int(id_) for id_ in dec_lines[i].split()]
            target = target[1:target.index(config.EOS_ID)]
            target = [inv_dec_vocab[word] for word in target]

            if len(encode) <= max_length:
                bucket_id = _find_right_bucket(len(encode))
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(encode, [])], 
                                                                                bucket_id,
                                                                                batch_size=1)
                _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                               decoder_masks, bucket_id, True)
                response = _construct_response(output_logits, inv_dec_vocab)
                response = response.split()

                bleu = getBLEU([target], response)

                bleuScores.append(bleu)

        meanBLEU = np.mean(bleuScores)

        sampleEncOutput = " ".join([tf.compat.as_str(inv_enc_vocab[enc]) for enc in encode])
        sampleDecOutput = " ".join(response)
        print("\nSample Test Translation... ")
        print("   Encoded Text: %s" % sampleEncOutput)
        print("   Decoded Text: %s" % sampleDecOutput)

        print("\nThe testing is complete! Average BLEU Score: %f\n" % meanBLEU)


def translate():
    """ 
    Translate function for the NMT
    This function prompts the user to input a sentence in English
    The function then uses the previously trained NMT model to predict 
    what the input sentence should be when translated to Vietnamese
    The function prints the predicted translation out to the user
    """
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = NMTModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        max_length = config.BUCKETS[-1][0]
        print("\nWelcome to the English-Vietnamese Neural Machine Translator!")
        print("Type a sentence in English and hit 'enter' to translate it to Vietnamese.")
        print("Hit 'enter' without typing anything to exit the Neural Machine Translator")
        print("**Note: The maximum sentence length is %d words\n" % max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            # Get encoder token ids for the input sentence.
            encIds = data.sentence2id(enc_vocab, str(line))
            if (len(encIds) > max_length):
                print('Error: The maximum sentence length is %d words' % max_length)
                line = _get_user_input()
                continue

            bucket_id = _find_right_bucket(len(encIds))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(encIds, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)

            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print('Vietnamese Translation: %s' % response)


def main():
    encoderVocabFile = os.path.join(config.PROCESSED_PATH, 'vocab.enc')
    if not os.path.isfile(encoderVocabFile):
        print('No vocabulary data found. Processing new vocabulary data.')
        data.process_data()
        print('New vocabulary data processed!')
    else: 
        print('Vocabulary data loaded from folder! Folder name: %s' % config.PROCESSED_PATH)

    # Create model folder for saving model if it does not currently exist
    data.make_dir(config.MODEL_PATH)

    if len(sys.argv) > 1: 
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'test':
            test()
        elif sys.argv[1] == 'translate':
            translate()

    else:
        print("\nNot a valid argument, please use an argument in one of the following formats...")
        print("python NMT.py train\npython NMT.py test\npython NMT.py translate\n")

if __name__ == '__main__':
    main()