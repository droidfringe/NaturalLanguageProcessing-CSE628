import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    # Function for default configuration, which is also the best model
    # Just power 5 is used insted of power 3
    # The model is trainied for 2000 steps
    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():

            # Set trainable to False for experiment 2c
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable=True)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """

            # Define placeholders
            self.train_inputs = tf.placeholder(dtype=tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(dtype=tf.float32,shape=(Config.batch_size, parsing_system.numTransitions()))
            # no need to specify shape of test_inputs
            # the tensor is being reshaped afterwards
            self.test_inputs = tf.placeholder(dtype=tf.int32)

            # define prediction
            self.numFeatures = Config.embedding_size*Config.n_Tokens
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1])

            weights_input = tf.Variable(tf.random_normal(shape=(self.numFeatures,Config.hidden_size),stddev=0.1))
            biases_input = tf.Variable(tf.random_normal(shape=(1,Config.hidden_size),stddev=0.1))
            weights_output = tf.Variable(tf.random_normal(shape=(Config.hidden_size,parsing_system.numTransitions()), stddev=0.1))
            # Used random normal initialization because we are using L2 loss for weights (Gaussian prior)
            #weights_output = tf.Variable(tf.random_uniform(shape=(Config.hidden_size, parsing_system.numTransitions()), minval=-0.01,maxval=0.01))
            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)

            # Loss function to not consider infeasible transitions (i.e. label = -1)
            # Since the number of -1 in each training example varies, we need to go over each example in batch
            # and select those indices where label is not -1. For these indices, select the predicted values and compute
            # the cross entropy loss
            # This method was very slow (even one iteration did not complete)
            # Hence did not use this method

            # cross_entropy = tf.Variable(0.0, dtype=tf.float32)
            # tf.assign(cross_entropy, 0.0)
            # for i in range(Config.batch_size):
            #     current_y = tf.gather(self.train_labels, i)
            #     selected_indices = tf.where(tf.not_equal(current_y,-1))
            #     selected_y = tf.reshape(tf.gather(current_y,selected_indices),[1,-1])
            #     current_prediction = tf.gather(self.prediction, i)
            #     selected_y_prediction = tf.reshape(tf.gather(current_prediction, selected_indices),[1,-1])
            #     # tf.assign_add(cross_entropy, tf.nn.softmax_cross_entropy_with_logits(labels=selected_y,logits=selected_y_prediction))
            #     cross_entropy = cross_entropy + tf.nn.softmax_cross_entropy_with_logits(labels=selected_y,logits=selected_y_prediction)

            #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.prediction))


            # Used sparse_cross_entropy_with_logits to handle -1 in labels
            # This method still considers non-feasible transitions for computing loss
            # This is not 100% in accordance with paper, but is faster
            sparse_labels = tf.argmax(self.train_labels, axis=1)
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=self.prediction))
            l2_loss = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(train_embed)
            self.loss = cross_entropy + Config.lam*l2_loss

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For experiment 2e - Apply unclipped gradients
            # self.app = optimizer.apply_gradients(grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()




    # For experiment 1 - two hidden layers
    '''
    def build_graph(self, graph, embedding_array, Config):

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            # define placeholders
            self.train_inputs = tf.placeholder(dtype=tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(dtype=tf.float32,
                                               shape=(Config.batch_size, parsing_system.numTransitions()))
            # no need to specify shape of test_inputs
            # the tensor is being reshaped afterwards
            self.test_inputs = tf.placeholder(dtype=tf.int32)

            # define prediction
            self.numFeatures = Config.embedding_size * Config.n_Tokens
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1])
            weights_input = tf.Variable(tf.random_normal(shape=(self.numFeatures, Config.hidden_size), stddev=0.1))
            biases_input = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size), stddev=0.1))
            weights_hidden = tf.Variable(tf.random_normal(shape=(Config.hidden_size, Config.hidden_size_2), stddev=0.1))
            biases_hidden = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size_2), stddev=0.1))
            weights_output = tf.Variable(
                tf.random_normal(shape=(Config.hidden_size_2, parsing_system.numTransitions()), stddev=0.1))

            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_output)

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.prediction))
            l2_loss = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_hidden) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(
                biases_input) + tf.nn.l2_loss(biases_hidden) + tf.nn.l2_loss(train_embed)
            self.loss = cross_entropy + Config.lam * l2_loss

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    '''

    # For experiment 1 - three hidden layers

    '''
    def build_graph(self, graph, embedding_array, Config):

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            # define placeholders
            self.train_inputs = tf.placeholder(dtype=tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(dtype=tf.float32,
                                               shape=(Config.batch_size, parsing_system.numTransitions()))
            # no need to specify shape of test_inputs
            # the tensor is being reshaped afterwards
            self.test_inputs = tf.placeholder(dtype=tf.int32)

            # define prediction
            self.numFeatures = Config.embedding_size * Config.n_Tokens
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1])
            weights_input = tf.Variable(tf.random_normal(shape=(self.numFeatures, Config.hidden_size), stddev=0.1))
            biases_input = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size), stddev=0.1))
            weights_hidden = tf.Variable(tf.random_normal(shape=(Config.hidden_size, Config.hidden_size_2), stddev=0.1))
            biases_hidden = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size_2), stddev=0.1))
            weights_hidden_2 = tf.Variable(tf.random_normal(shape=(Config.hidden_size_2, Config.hidden_size_3), stddev=0.1))
            biases_hidden_2 = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size_3), stddev=0.1))
            weights_output = tf.Variable(
                tf.random_normal(shape=(Config.hidden_size_3, parsing_system.numTransitions()), stddev=0.1))

            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_hidden_2, biases_hidden_2, weights_output)

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.prediction))
            l2_loss = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_hidden) + tf.nn.l2_loss(weights_hidden_2) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(
                biases_input) + tf.nn.l2_loss(biases_hidden) + tf.nn.l2_loss(biases_hidden_2) + tf.nn.l2_loss(train_embed)
            self.loss = cross_entropy + Config.lam * l2_loss

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_hidden_2, biases_hidden_2, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    '''

    # For experiment 2b - no skip connections between words, pos and label embeddings
    '''
    def build_graph(self, graph, embedding_array, Config):

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            # define placeholders
            self.train_inputs = tf.placeholder(dtype=tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(dtype=tf.float32,
                                               shape=(Config.batch_size, parsing_system.numTransitions()))
            # no need to specify shape of test_inputs
            # the tensor is being reshaped afterwards
            self.test_inputs = tf.placeholder(dtype=tf.int32)

            # define prediction
            self.numFeatures = Config.embedding_size * Config.n_Tokens
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1])

            embedding_tr = tf.transpose(train_embed)
            word_embeddings = tf.transpose(tf.nn.embedding_lookup(embedding_tr, np.arange(18*Config.embedding_size)))
            pos_embeddings = tf.transpose(tf.nn.embedding_lookup(embedding_tr, 18*Config.embedding_size +np.arange(18*Config.embedding_size)))
            label_embeddings = tf.transpose(tf.nn.embedding_lookup(embedding_tr, 36*Config.embedding_size +np.arange(12*Config.embedding_size)))
            # As per paper, initialize weights in range [-0.01,0.01]
            # Used larger sigma as loss was not changing with smaller sigma

            words_weights_input = tf.Variable(tf.random_normal(shape=(18*Config.embedding_size, Config.hidden_size), stddev=0.2))
            words_biases_input = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size), stddev=0.2))
            pos_weights_input = tf.Variable(tf.random_normal(shape=(18*Config.embedding_size, Config.hidden_size), stddev=0.2))
            pos_biases_input = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size), stddev=0.2))
            label_weights_input = tf.Variable(tf.random_normal(shape=(12*Config.embedding_size, Config.hidden_size), stddev=0.2))
            label_biases_input = tf.Variable(tf.random_normal(shape=(1, Config.hidden_size), stddev=0.2))

            weights_output = tf.Variable(tf.random_normal(shape=(Config.hidden_size, parsing_system.numTransitions()), stddev=0.2))
            self.prediction = self.forward_pass(word_embeddings,pos_embeddings, label_embeddings,words_weights_input,words_biases_input,pos_weights_input,pos_biases_input,label_weights_input,label_biases_input,weights_output)


            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_labels, logits=self.prediction))
            l2_loss = tf.nn.l2_loss(words_weights_input) + tf.nn.l2_loss(pos_weights_input) + tf.nn.l2_loss(label_weights_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(words_biases_input) + tf.nn.l2_loss(pos_biases_input) + tf.nn.l2_loss(label_biases_input) + tf.nn.l2_loss(train_embed)
            self.loss = cross_entropy + Config.lam * l2_loss

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])

            embedding_test = tf.transpose(test_embed)
            word_embeddings_te = tf.transpose(tf.nn.embedding_lookup(embedding_test, np.arange(18*Config.embedding_size)))
            pos_embeddings_te = tf.transpose(tf.nn.embedding_lookup(embedding_test, 18*Config.embedding_size+np.arange(18*Config.embedding_size)))
            label_embeddings_te = tf.transpose(tf.nn.embedding_lookup(embedding_test, 36*Config.embedding_size+np.arange(12*Config.embedding_size)))

            self.test_pred = self.forward_pass(word_embeddings_te, pos_embeddings_te, label_embeddings_te, words_weights_input,
                              words_biases_input, pos_weights_input, pos_biases_input, label_weights_input,
                              label_biases_input, weights_output)
            # self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()
    '''

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        # 5 for best model
        polynomial_power = 3
        hidden = tf.pow(tf.matmul(embed,weights_input) + biases_input,polynomial_power)
        # out = tf.pow(tf.matmul(hidden,weights_output),polynomial_power)
        out = tf.matmul(hidden, weights_output)

        # Relu
        # hidden = tf.nn.relu(tf.matmul(embed,weights_input) + biases_input)
        # out = tf.matmul(hidden,weights_output)

        # Sigmoid
        # hidden = tf.nn.sigmoid(tf.matmul(embed,weights_input) + biases_input)
        # out = tf.matmul(hidden, weights_output)

        # Tanh
        # hidden = tf.nn.tanh(tf.matmul(embed,weights_input) + biases_input)
        # out = tf.matmul(hidden,weights_output)
        return out


    # For experiment 1 - 2 Hidden layers
    '''
    def forward_pass(self, embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_output):
        polynomial_power = 3
        hidden = tf.pow(tf.matmul(embed,weights_input) + biases_input,polynomial_power)
        hidden2 = tf.pow(tf.matmul(hidden, weights_hidden) + biases_hidden, polynomial_power)
        out = tf.matmul(hidden2, weights_output)
        return out
    '''

    # For experiment 1 - 3 Hidden layers
    '''
    def forward_pass(self, embed, weights_input, biases_input, weights_hidden, biases_hidden, weights_hidden_2, biases_hidden_2, weights_output):
        polynomial_power = 3
        hidden = tf.pow(tf.matmul(embed,weights_input) + biases_input, polynomial_power)
        hidden2 = tf.pow(tf.matmul(hidden, weights_hidden) + biases_hidden, polynomial_power)
        hidden3 = tf.pow(tf.matmul(hidden2, weights_hidden_2) + biases_hidden_2, polynomial_power)
        out = tf.matmul(hidden3, weights_output)
        return out
    '''

    # For experiment 2b
    '''
    def forward_pass(self, word_embeddings,pos_embeddings, label_embeddings,words_weights_input,words_biases_input,pos_weights_input,pos_biases_input,label_weights_input,label_biases_input,weights_output):
        polynomial_power = 3
        hidden = tf.pow(tf.matmul(word_embeddings, words_weights_input) + words_biases_input, polynomial_power)
        hidden = hidden + tf.pow(tf.matmul(pos_embeddings, pos_weights_input) + pos_biases_input, polynomial_power)
        hidden = hidden + tf.pow(tf.matmul(label_embeddings, label_weights_input) + label_biases_input, polynomial_power)
        # out = tf.pow(tf.matmul(hidden,weights_output),polynomial_power)
        out = tf.matmul(hidden, weights_output)
        return out
    '''

def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    stack0 = c.getStack(0)
    stack1 = c.getStack(1)
    stack2 = c.getStack(2)
    buffer0 = c.getBuffer(0)
    buffer1 = c.getBuffer(1)
    buffer2 = c.getBuffer(2)
    lc1_s1 = c.getLeftChild(stack0,1)
    lc2_s1 = c.getLeftChild(stack0,2)
    rc1_s1 = c.getRightChild(stack0,1)
    rc2_s1 = c.getRightChild(stack0,2)
    lc1_s2 = c.getLeftChild(stack1,1)
    lc2_s2 = c.getLeftChild(stack1,2)
    rc1_s2 = c.getRightChild(stack1,1)
    rc2_s2 = c.getRightChild(stack1,2)
    lc1_lc1_s1 = c.getLeftChild(lc1_s1,1)
    rc1_rc1_s1 = c.getRightChild(rc1_s1,1)
    lc1_lc1_s2 = c.getLeftChild(lc1_s2, 1)
    rc1_rc1_s2 = c.getRightChild(rc1_s2, 1)
    words = [stack0,stack1,stack2,buffer0,buffer1,buffer2,lc1_s1,lc2_s1,rc1_s1,rc2_s1,lc1_s2,lc2_s2,rc1_s2,rc2_s2,lc1_lc1_s1,rc1_rc1_s1,lc1_lc1_s2,rc1_rc1_s2]
    #posTags = [posDict[c.getPOS(w)] for w in words]
    #labels = [labelDict[c.getLabel(words[i])] for i in range(6,18)]
    posTags = [getPosID(c.getPOS(w)) for w in words]
    labels = [getLabelID(c.getLabel(words[i])) for i in range(6,18)]
    # print(words)
    # print(posTags)
    # print(labels)
    words2 = [c.getWord(w) for w in words]
    features = [getWordID(w) for w in words2]
    # features = []
    # for w in words2:
    #     try:
    #         features.append(wordDict[w])
    #     except KeyError:
    #         features.append(wordDict[Config.UNKNOWN])
    features.extend(posTags)
    features.extend(labels)
    # print(features)
    return features

def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()
    print('DependencyParser: genTrainExamples: start')
    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
    # for i in range(1000):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])
            # print('DependencyParser: genTrainExamples: is c None?', c is None)
            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        # label.append(0.)
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
            # print('iter ',i, 'is equal? ', trees[i].equal(c.tree))
            # print(trees[i].print_tree())
            # print(c.tree.print_tree())
    print('DependencyParser: genTrainExamples: complete')
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array

def load_embeddings_one_hot(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(wordDict)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    # One hot representation for POS tags
    for i in range(len(posDict)):
        embedding_array[i+len(wordDict)] = np.zeros(Config.embedding_size)
        embedding_array[i + len(wordDict),i] = 1.0

    # One hot representation for Labels
    for i in range(len(labelDict)):
        embedding_array[i+len(wordDict)+len(posDict)] = np.zeros(Config.embedding_size)
        embedding_array[i + len(wordDict)+len(posDict),i] = 1.0

    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)
    print(len(wordDict), len(posDict), len(labelDict))
    print(posDict)
    print('Label dict')
    print(labelDict)
    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    # For experiment 2c, used one-hot embeddings for POS and Labels
    # embedding_array = load_embeddings_one_hot(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    # Save time in computing features
    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    # with open('features.pkl', 'wb') as fid:
    #    pickle.dump([trainFeats, trainLabels], fid)

    # with open('features_with0.pkl', 'rb') as fid:
    #     trainFeats, trainLabels = pickle.load(fid)
    print "Done."


    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter

    with tf.Session(graph=graph) as sess:
        # tf.set_random_seed(1234)
        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

