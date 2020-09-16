# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:49:19 2020

@author: Rakin Shahriar
"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

class RBM(object):
    def __init__(self, visibleDimensions, epochs = 60, hiddenDimensions=50, ratingValues = 10,learningRate=0.001,batchSize=100):
        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def Train(self, X):
        ops.reset_default_graph()
        self.MakeGraph()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        for epoch in range(self.epochs):
            np.random.shuffle(X)
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                self.sess.run(self.update, feed_dict = {self.X: trX[i:i+self.batchSize]})
                
            print("Trained epoch ", epoch)
            
    def GetRecommendations(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights)+self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)
        feed = self.sess.run(hidden, feed_dict={self.X: inputUser})
        rec = self.sess.run(visible, feed_dict = {hidden : feed})
        return rec[0]
    
    def MakeGraph(self):
        tf.set_random_seed(0)
        tf.disable_eager_execution()
        self.X = tf.placeholder(tf.float32,[None, self.visibleDimensions], name = "X")
        maxWeight = -4.0 * np.sqrt(6.0/(self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight))
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions],tf.float32, name="hiddenBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions],tf.float32, name="visibleBias"))
        hProb0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random_uniform(tf.shape(hProb0))))
        forward = tf.matmul(tf.transpose(self.X), hSample)
        v = tf.matmul(hSample,tf.transpose(self.weights)) + self.visibleBias
        vMask = tf.sign(self.X)
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues])
        vMask3D = tf.reduce_max(vMask3D, axis = [2], keepdims=True)
        v = tf.reshape(v,[tf.shape(v)[0],-1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D)
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1])
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights)+self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
        weightUpdate = self.weights.assign_add(self.learningRate*(forward-backward))
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(self.X - vProb, 0))
        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
        