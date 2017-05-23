# chay voi data 1523


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib
import math


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

def bias_variable_notrainable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, trainable=False)	

def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')

def conv3dStr(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool3d(x):
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')

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
        batch_size = 1
        for batch_idx in range(0, len(features), batch_size):
            # images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            # images_batch = images_batch.astype("float32")
            if batch_idx + batch_size >= len(features):
        		batch_idx=0

        		print('RE-SHUFFLE BATCHES')

        		# shuffle again
        		idxs = np.arange(0, len(labels))
		        np.random.shuffle(idxs)
		        shuf_features = features[idxs]
		        shuf_labels = labels[idxs]

            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch

def getActivations(layer,stimuli, sess, x, keep_prob, saved_name=None):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,79, 95, 68],order='F'),keep_prob:1.0})
    plotNNFilter(units, saved_name)

def plotNNFilter(units=None, saved_name=None):
	if units==None:
		units = np.load(join(rdfa.get_stored_temp_dir(), 'units2.npy'))

	print(np.shape(units))
	print(type(units).__name__)
	print(units.shape[3])

	if saved_name != None:
		saved_name = join(rdfa.get_stored_temp_dir(), saved_name)

	#np.save(join(rdfa.get_stored_temp_dir(), 'units.npy', units)
	np.save(saved_name, units)

	slice_num = math.ceil(units.shape[3] / 2)
	filters = units.shape[4]
	plt.figure(1, figsize=(20,20))
	n_columns = 6
	n_rows = math.ceil(filters / n_columns) + 1
	for i in range(filters):
		ax = plt.subplot(n_rows, n_columns, i+1)
		plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,slice_num,i], interpolation="nearest", cmap="gray")
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

def main(_):
	# plotNNFilter()
	# exit()

	num_train_ite = 100 # 800 #405
	num_test_ite = 200 #102
	psize = 5
	encode_dim = 8
	encode_dim2 = 8
	encode_dim3= 8
	saved_dir = rdfa.get_stored_temp_dir()

	#mri_train, mri_test, label_train, label_test = rdfa.read_npy_file_3d() # NON norm
	mri_train, mri_test, label_train, label_test = rdfa.read_npy_file_3d_norm() # norm
	one_slice = mri_train[500, :, :, 34];

	# NORMALIZATION
	# mri_train = mri_train/(np.amax(mri_train,axis=(1,2,3))[:,None, None, None] * 1.0)
	# mri_test = mri_test/(np.amax(mri_test,axis=(1,2,3))[:,None, None, None] * 1.0)

	# plt.hist(mri_train[1].flatten(), 1000, range=(0,3000), fc='k', ec='k')
	# plt.show()
	print(np.amax(mri_train[1:10],axis=(1,2,3)))

	# arr = np.asarray(one_slice)
	# plt.imshow(arr, cmap='gray')
	# plt.show()

	print("train, test size", np.shape(mri_train), np.shape(mri_test))


	iter_ = data_iterator(mri_train, label_train)
	iter2_ = data_iterator(mri_test, label_test)	

	# Create the model
	x = tf.placeholder(tf.float32, [None, 79, 95, 68])
	#x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))

	W_conv1 = weight_variable([5, 5, 5, 1, encode_dim])
	# NEU DUNG AUTOENCODER
	if 0:

		ini_w_name = 'ini_weight_t.npy' # shape: 5,5,5,1,32 
		ini_w_name2 = 'ini_weight2.npy' # shape: 5,5,5, 32 

		if 0:
			ini_weight_t = np.load(join(saved_dir, ini_w_name))
			#ini_weight_t = np.reshape(ini_weight_t, (psize,psize,psize, 1, encode_dim)) # for ini_weight2.npy
		else:
			ini_weight, ini_weight2 = test_simple_ae(mri_train, mri_test)
			print(np.shape(ini_weight)) #weight encode => Transpose (e.g 2 x 2 x 2 * 5, size (5,8)=> Transpose, size (2,2,2,5))
			print(np.shape(ini_weight2)) # weight decode (e.g size (2,2,2, 5)
			ini_weight_t = np.transpose(ini_weight, (1, 2, 3, 0))
			ini_weight_t = np.reshape(ini_weight_t, (psize,psize,psize, 1, encode_dim))

			#np.save(join(saved_dir, ini_w_name), ini_weight_t);
			#np.save(join(saved_dir, ini_w_name2), ini_weight2);
			#print ("ini_weight_t reshape: ", ini_weight_t)

		W_conv1 =  tf.Variable(ini_weight_t, trainable=False)
		#W_conv1 =  ini_weight_t

	b_conv1 = bias_variable([encode_dim])
	
	x_image = tf.reshape(x, [-1, 79, 95, 68, 1])
	
	print (x_image.get_shape())

	h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
	print_shape("h_conv1: ", h_conv1)

	h_pool1 = max_pool3d(h_conv1)
	print_shape("h_pool1: ", h_pool1)
	
	W_conv2 = weight_variable([5, 5, 5, encode_dim, encode_dim2])
	b_conv2 = bias_variable([encode_dim2])
	
	h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
	print_shape("h_conv2: ", h_conv2)

	h_pool2 = max_pool3d(h_conv2)
	print_shape("h_pool2: ", h_pool2)

	W_conv3 = weight_variable([3, 3, 3, encode_dim2, encode_dim3])
	b_conv3 = bias_variable([encode_dim3])
	
	h_conv3 = tf.nn.relu(conv3dStr(h_pool2, W_conv3) + b_conv3)
	print_shape("h_conv3: ", h_conv3)

	h_pool3 = max_pool3d(h_conv3)
	print_shape("h_pool3: ", h_pool3)

	# W_conv4 = weight_variable([3, 3, 3, 384, 256])
	# b_conv4 = bias_variable([256])
	# h_conv4 = tf.nn.relu(conv3d(h_pool3, W_conv4) + b_conv4)

	# W_conv5 = weight_variable([3, 3, 3, 256, 256])
	# b_conv5 = bias_variable([256])
	# h_conv5 = tf.nn.relu(conv3d(h_conv4, W_conv5) + b_conv5)

	# W_conv6 = weight_variable([3, 3, 3, 256, 256])
	# b_conv6 = bias_variable([256])
	# h_conv6 = tf.nn.relu(conv3d(h_conv4, W_conv6) + b_conv6)
	# print_shape("h_conv4: ", h_conv4)

	# h_pool4 = max_pool3d(h_conv6)
	# print_shape("h_pool4: ", h_pool4)

	num_hid_units = 400
	#W_fc1 = weight_variable([15 * 18 * 12 * encode_dim, num_hid_units])
	W_fc1 = weight_variable([4 * 5 * 3 * encode_dim3, num_hid_units])
	b_fc1 = bias_variable([num_hid_units])
	
	#h_pool2_flat = tf.reshape(h_pool1, [-1, 15 * 18 * 12 * encode_dim])
	h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 5 * 3 * encode_dim3])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	W_fc2 = weight_variable([num_hid_units, num_hid_units])
	b_fc2 = bias_variable([num_hid_units])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([num_hid_units, 2])
	b_fc2 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])
	all_acc = []
	sess = tf.InteractiveSession()
	
	log_file = join(rdfa.get_stored_temp_dir(), 'log.txt')
	log_pth = rdfa.get_stored_temp_dir()
	# f = open( log_file, 'w' )
	# f.write( 'dict = ' + repr(W_conv1) + '\n' )
	# f.close()

	# Train

	tf.global_variables_initializer().run()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

	print(tf.trainable_variables())

	#opt_vars = tf.trainable_variables()[9:13]
	
	#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=opt_vars)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())

	for i in range(num_train_ite):
		xval, y_val = iter_.next()
		train_accuracy = accuracy.eval(feed_dict={x:xval, y_: y_val, keep_prob: 1.0})
		#print('tf_reduce_max', tf.reduce_max(x))

		all_acc.append(train_accuracy)
		if i%50 == 0:
			print("step %d, training accuracy %g"%(i, np.mean(all_acc)))

			log_file = join(log_pth, 'logfile'+ str(i)+ '.txt')
			f = open( log_file, 'w' )
			#f.write( 'dict = ' + str(W_conv1) + '\n' )
			print(W_conv1.eval()[4,4,2,0,6])
			#f.write( 'dict = ' + repr(W_fc2.eval()) + '\n' )
			f.close()

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

	imageToUse = mri_train[1]

	print(type(sess).__name__)
	print(type(h_conv2).__name__)
	getActivations(h_conv2,imageToUse, sess, x, keep_prob, 'units2.npy')
	getActivations(h_conv3,imageToUse, sess, x, keep_prob, 'units3.npy')	

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
