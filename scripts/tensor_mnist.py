import argparse
import sys
import numpy as np
import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Get data from tensorflow examples
from tensorflow.examples.tutorials.mnist import input_data

def run():
	print('Started')
	mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
	print('2')
	# Nodes for imput images 
	# None means any size - number of input images is the number of rows
	# It can be any number based on train, test split. So, none is given.
	# 784 is the columns - dimensional vectors got from 28*28 pixel image
	x = tf.placeholder(tf.float32, shape=[None, 784])

	# Weights and bias variables. Variables are modifiable.
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	# Implement out model
	y = tf.nn.softmax(tf.matmul(x,W)+b)

	# Placeholder to input the correct answers:
	y_ = tf.placeholder(tf.float32, shape=[None,10])

	# Cross entropy function - this may not be stable - dont use
#	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

	# Using Gradient Descent
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# Using interactive session
	sess = tf.InteractiveSession()

	# Initialize all variables we created
	tf.global_variables_initializer().run()

	# Run our algorithm
	# On each iteration, batch of 100 random images comes from training data 
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Make Predictions
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

	# Accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Find accuracy on test data
	print(sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels}))





