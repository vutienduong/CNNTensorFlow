from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool3d(x):
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def print_type_name(x):
	print(type(x).__name__)

# read input and create a 4D tensor
def read_input():
	# example_ni1 = os.path.join(path, file_name)
	mri_input_file = '/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/mris.txt'
	mri_train = []
	mri_test  = []

	path = "/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/resizeMRI"
	count = 1

	f = open(mri_input_file, 'r')
	for line in f:
		line = line.strip('\n')
		example_ni1 = os.path.join(path, line)
		n1_img = nib.load(example_ni1)
		image_data = n1_img.get_data()

		if count < 31:
			mri_train.append(image_data)
		else:
			mri_test.append(image_data)

		count = count + 1
	f.closed
	
	mri_train = np.array(mri_train)
	mri_test = np.array(mri_test)

	print ("load OK ")
	return mri_train, mri_test

# read label input and create a label list
def read_label_input():
	mri_input_file = '/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/mris_label.txt'
	label_train = []
	label_test = []
	count = 1

	f = open(mri_input_file, 'r')
	for line in f:
		line = int(line)

		if line == 1:
			line = [1, 0]
		else:
			line = [0, 1]

		if count < 31:
			label_train.append(line)
		else:
			label_test.append(line)

		count = count + 1
	f.closed

	label_train = np.array(label_train)
	label_test = np.array(label_test)
	return label_train, label_test

def print_shape(strg, var_name):
	print (strg + str(var_name.get_shape()))

def data_iterator(features, labels):
    # """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(labels))
       
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        batch_size = 10
        for batch_idx in range(0, len(features), batch_size):
            # images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            # images_batch = images_batch.astype("float32")

            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


def main(_):
	mri_train, mri_test = read_input()
	label_train, label_test = read_label_input()

	iter_ = data_iterator(mri_train, label_train)

	# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	
  # Create the model
	# x = tf.placeholder(tf.float32, [None, 784])
	x = tf.placeholder(tf.float32, [None, 30, 30, 20])
	
	W_conv1 = weight_variable([5, 5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1, 30, 30, 20, 1])
	
	print (x_image.get_shape())

	h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
	print_shape("h_conv1: ", h_conv1)

	h_pool1 = max_pool3d(h_conv1)
	print_shape("h_pool1: ", h_pool1)
	
	W_conv2 = weight_variable([5, 5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	
	h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
	print_shape("h_conv2: ", h_conv2)

	h_pool2 = max_pool3d(h_conv2)
	print_shape("h_pool2: ", h_pool2)

	W_fc1 = weight_variable([8 * 8 * 5 * 64, 512])
	b_fc1 = bias_variable([512])
	
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 5 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([512, 2])
	b_fc2 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])
	
	sess = tf.InteractiveSession()
	
	# Train
	tf.initialize_all_variables().run()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	for i in range(200):
		# batch = mnist.train.next_batch(50)
		xval, y_val = iter_.next()
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:xval, y_: y_val, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: xval, y_: y_val, keep_prob: 0.5})
		print ("run %d ..."%i)
	print("test accuracy %g"%accuracy.eval(feed_dict={x: mri_test, y_: label_test, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()