from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def main(_):
	# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

 #  # Create the model
	# x = tf.placeholder(tf.float32, [None, 784])
	# W_conv1 = weight_variable([5, 5, 1, 32])
	# b_conv1 = bias_variable([32])
	
	# x_image = tf.reshape(x, [-1,28,28,1])
	
	# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	# h_pool1 = max_pool_2x2(h_conv1)
	
	# W_conv2 = weight_variable([5, 5, 32, 64])
	# b_conv2 = bias_variable([64])
	
	# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	# h_pool2 = max_pool_2x2(h_conv2)
	
	# W_fc1 = weight_variable([7 * 7 * 64, 1024])
	# b_fc1 = bias_variable([1024])
	
	# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	# keep_prob = tf.placeholder(tf.float32)
	# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# W_fc2 = weight_variable([1024, 10])
	# b_fc2 = bias_variable([10])
	
	# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

 #  # Define loss and optimizer
	# y_ = tf.placeholder(tf.float32, [None, 10])
	
	# sess = tf.InteractiveSession()
	a = [[[1],[3],[2]], [[5],[6],[7]]]
	b = [[[1],[3],[2]], [[5],[6],[7]]]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()