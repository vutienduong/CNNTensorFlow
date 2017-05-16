# chay voi data 1523


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib
import tensorflow.contrib.slim as slim
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
	return tf.nn.max_pool3d(x, ksize=[1, 5, 5, 5, 1], strides=[1, 5, 5, 5, 1], padding='VALID')

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


def getActivations(layer,stimuli, sess, x, keep_prob):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,79, 95, 68],order='F'),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units=None):
	if units==None:
		units = np.load('/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/units.npy')

	print(np.shape(units))
	print(type(units).__name__)
	print(units.shape[3])
	#np.save('/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/units.npy', units)
	#exit()
	slice_num = int(units.shape[3] / 2)
	print(slice_num)
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

	num_train_ite = 50 # 800 #405
	num_test_ite = 200 #102
	psize = 5
	encode_dim = 32

	saved_dir = '/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/'

	mri_train, mri_test, label_train, label_test = rdfa.read_npy_file_3d_norm() # norm
	iter_ = data_iterator(mri_train, label_train)
	iter2_ = data_iterator(mri_test, label_test)
	tf.reset_default_graph()	

	# Create the model
	x = tf.placeholder(tf.float32, [None, 79, 95, 68], name="x-in")
	# x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))

	if 1:

		ini_w_name = 'ini_weight_t70.npy' # shape: 5,5,5,1,32 
		ini_w_name2 = 'ini_weight270.npy' # shape: 5,5,5, 32 
		list_keep_ind = [2,3,4,5,6,8,11,15,17,18,19,21,23,24,27,28,29,32,36,37,38,40,43,45,46,47,53,54,55,56,57,58];
		# list_keep_ind2 = range(1,17);
		# list_keep_ind3 = [3,4,9,11,14,15];

		if 1:
			ini_weight_t = np.load(join(saved_dir, ini_w_name))
			print(np.shape(ini_weight_t))
			ini_weight_t = ini_weight_t[:,:,:,:,list_keep_ind]
			# ini_weight_t = ini_weight_t[:,:,:,:,list_keep_ind2]
			# ini_weight_t = ini_weight_t[:,:,:,:,list_keep_ind3]
			# print(np.shape(ini_weight_t))
			#exit()
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
	
	#W_conv1 = weight_variable([5, 5, 5, 1, encode_dim])
	b_conv1 = bias_variable_notrainable([encode_dim])
	x_image = tf.reshape(x, [-1, 79, 95, 68, 1])

	h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
	print_shape("h_conv1: ", h_conv1)

	h_pool1 = max_pool3d(h_conv1)
	print_shape("h_pool1: ", h_pool1)

	num_hid_units = 400
	#W_fc1 = weight_variable([15 * 18 * 12 * encode_dim, num_hid_units])
	#exit()
	W_fc1 = weight_variable([15 * 18 * 12 * encode_dim, num_hid_units])
	b_fc1 = bias_variable([num_hid_units])
	
	#h_pool2_flat = tf.reshape(h_pool1, [-1, 15 * 18 * 12 * encode_dim])
	h_pool2_flat = tf.reshape(h_pool1, [-1, 15 * 18 * 12 * encode_dim])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	W_fc3 = weight_variable([num_hid_units, 200])
	b_fc3 = bias_variable([200])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc1, W_fc3) + b_fc3)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = slim.dropout(h_fc3, keep_prob)
	
	W_fc2 = weight_variable([200, 2])
	b_fc2 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])
	all_acc = []
	sess = tf.InteractiveSession()
	
	log_file = '/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/log.txt'
	log_pth = '/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/'
	# f = open( log_file, 'w' )
	# f.write( 'dict = ' + repr(W_conv1) + '\n' )
	# f.close()

	# Train

	tf.global_variables_initializer().run()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

	print(tf.trainable_variables())

	#opt_vars = tf.trainable_variables()[9:13]
	
	#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=opt_vars)
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
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
			print(W_conv1.eval()[4,4,2,0,2])
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

	#imageToUse = mri_train[1]

	#TEST
	class_path = '/home/ngoc/Desktop/_dataset/'
	#mri_name = 'old/AD/wADNI_003_S_1059_MR_MPR__GradWarp__B1_Correction__N3_Br_20070501173419666_S22301_I52811.hdr' # AD
	#ri_name = 'old/NC/wADNI_005_S_0223_MR_MPR__GradWarp__B1_Correction__N3_Br_20061212164202354_S11981_I32856.hdr' # NC

	#mri_name = 'old/AD/wADNI_003_S_1059_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20070122123023427_1_S25144_I37090.hdr' # AD
	mri_name = 'old/NC/wADNI_005_S_0223_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20061013152111299_1_S12249_I26257.hdr' # NC

	mri_img = nib.load(join(class_path, mri_name))
	imageToUse = mri_img.get_data()

	#print(np.shape(mri_train[1]))
	getActivations(h_conv1,imageToUse, sess, x, keep_prob)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
