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

from os import listdir
from os.path import isfile, join

from autoencoder_duong.my_conv_autoencoder import *
import autoencoder_duong.read_data_from_analyze as rdfa

FLAGS = None

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def conv3dStr(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool3d(x):
	return tf.nn.max_pool3d(x, ksize=[1, 5, 5, 5, 1], strides=[1, 5, 5, 5, 1], padding='SAME')

def print_type_name(x):
	print(type(x).__name__)


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
        batch_size = 2
        for batch_idx in range(0, len(features), batch_size):
            # images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            # images_batch = images_batch.astype("float32")
            if batch_idx + batch_size >= len(features):
        		batch_idx=0
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


def main(_):
	num_train_ite = 50# 405
	num_test_ite = 20 #102
	psize = 5

	mypath = rdfa.get_data_dir('all MRI 1523')
	adni, adni_label = rdfa.read_data2(mypath, ['AD', 'NC'], [80, 0, 20], False)
	mri_train = adni['train'][0];
	mri_test = adni['test'][0];

	print("train set size", np.shape(mri_train))
	print(np.shape(mri_test))

	label_train = adni_label['train'];
	label_test = adni_label['test'];

	iter_ = data_iterator(mri_train, label_train)
	iter2_ = data_iterator(mri_test, label_test)	

	# Create the model
	x = tf.placeholder(tf.float32, [None, 79, 95, 68])
	x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
	#initial = tf.truncated_normal([psize, psize, psize, 1, 32], stddev=0.1)

	# NEU DUNG AUTOENCODER
	W_conv1 = weight_variable([5, 5, 5, 1, 32])

	if 1:
		ini_weight, ini_weight2 = test_simple_ae(mri_train, mri_test)

		print(np.shape(ini_weight)) #weight encode => Transpose (e.g 2 x 2 x 2 * 5, size (5,8)=> Transpose, size (2,2,2,5))
		print(np.shape(ini_weight2)) # weight decode (e.g size (2,2,2, 5)

		ini_weight_t = np.transpose(ini_weight, (1, 2, 3, 0))
		#print ("ini_weight_t transpose: ", ini_weight_t)

		ini_weight_t = np.reshape(ini_weight_t, (psize,psize,psize, 1, 32))
		#print ("ini_weight_t reshape: ", ini_weight_t)

		#W_conv1 =  tf.Variable(ini_weight_t)
		W_conv1 = ini_weight_t #constant

	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1, 79, 95, 68, 1])

	h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
	print_shape("h_conv1: ", h_conv1)

	h_pool1 = max_pool3d(h_conv1)
	print_shape("h_pool1: ", h_pool1)
	#exit()

	last_size = 16 * 19 * 14 * 32
	W_fc1 = weight_variable([last_size, 256])
	b_fc1 = bias_variable([256])
	
	h_pool1_flat = tf.reshape(h_pool1, [-1, last_size])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([256, 2])
	b_fc2 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])
	all_acc = []
	sess = tf.InteractiveSession()
	

	# Train
	tf.global_variables_initializer().run()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())

	for i in range(num_train_ite):
		xval, y_val = iter_.next()
		if i%1 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:xval, y_: y_val, keep_prob: 1.0})
			all_acc.append(train_accuracy)
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: xval, y_: y_val, keep_prob: 0.5})
	print ("run %d ..."%i)
	#print("test accuracy %g"%accuracy.eval(feed_dict={x: mri_test, y_: label_test, keep_prob: 1.0}))
	print("train accuracy %g"%np.mean(all_acc))

	resl = []
	for j in range(num_test_ite):
		xxtest, y_test = iter2_.next()
		#print(y_test)
		acc = accuracy.eval(feed_dict={ x: xxtest, y_: y_test, keep_prob: 1.0})
		resl.append(acc)
	print("test accuracy %g"%np.mean(resl))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()