""" This code produces a neural machine translation system, capable of translating 
input English text data into its corresponding Vietnamese text data. 

Note that the code used to produce this model is heavily based off of the neural chatbot
model created by Chip Huyen, which can be found here... 
https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot

This chatbot model itself was based heavily off of the Google Translate Tensorflow model, which
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

from model import ChatBotModel
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
    """ Restore the previously trained parameters if there are any. """
    # ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    # if ckpt and ckpt.model_checkpoint_path:
    #     print("Loading parameters for the Chatbot")
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     print("Initializing fresh parameters for the Chatbot")
    model_path = os.path.join(config.CPT_PATH, 'saved_model-31400')
    if os.path.isfile(os.path.join(config.CPT_PATH, 'checkpoint')):
        print("Loading saved model... ")
        saver.restore(sess, model_path)
    else: 
        print("No saved model. Initializing a new model.")


def train():
    """ Train the NMT """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # in train mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
                                                                           bucket_id,
                                                                           batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % 100 == 0:
                print('Iteration {}: Loss {}, Time {}'.format(iteration, total_loss/100, time.time() - start))
                start = time.time()
                total_loss = 0

                if tf.gfile.Exists(config.CPT_PATH):
                    tf.gfile.DeleteRecursively(config.CPT_PATH)
                tf.gfile.MakeDirs(config.CPT_PATH)

                saver.save(sess, os.path.join(config.CPT_PATH, 'saved_model'), global_step=model.global_step)

                sys.stdout.flush()

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

def chat():
    """ in test mode, we don't to create the backward path
    """
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome to TensorBro. Say something. Enter to exit. Max length is', max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = data.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                line = _get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()


def getBLEU(ref, hyp):
    smoothing = SmoothingFunction()
    bleuScore = sentence_bleu(ref, hyp, smoothing_function=smoothing.method1)
    return bleuScore


def test(): 
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    enc_file = open(os.path.join(config.PROCESSED_PATH, 'test_ids.enc'), 'r')
    # dec_file = open(os.path.join(config.PROCESSED_PATH, 'test.dec'), 'r')
    dec_file = open(os.path.join(config.PROCESSED_PATH, 'test_ids.dec'), 'r')

    enc_lines = enc_file.read().splitlines()
    dec_lines = dec_file.read().splitlines()

    # print(enc_lines[0])
    # print(dec_lines[0])

    # print(enc_lines[0].split())
    # print(dec_lines[0].split())

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()
    bleuScores = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        max_length = config.BUCKETS[-1][0]
        for i in range(len(enc_lines)):
            encode = [int(id_) for id_ in enc_lines[i].split()]
            # target = dec_lines[i].split()
            target = [int(id_) for id_ in dec_lines[i].split()]
            target = target[1:target.index(config.EOS_ID)]
            target = [inv_dec_vocab[word] for word in target]



            if len(encode) <= max_length:
                bucket_id = _find_right_bucket(len(encode))
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
        print("Average BLEU: %f" % meanBLEU)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat', 'test'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(config.PROCESSED_PATH, 'vocab.enc')):
        # data.prepare_raw_data()
        data.process_data()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()
    elif args.mode == 'test':
        test()

if __name__ == '__main__':
    main()