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

FLAGS = None

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def conv2dNoStride(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

def max_pool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


def print_type_name(x):
	print(type(x).__name__)

# read input and create a 4D tensor
def read_nibabel(class_path,img_name):
	pet_img = nib.load(join(class_path, img_name))
	pet_data = pet_img.get_data()
	pet_data_trans = np.transpose(pet_data, (2, 0, 1)) # 79x95x68 => 68 x 79 x 95
	pet_data_trans = pet_data_trans[20:40] # remove some at begin and some slices at end
	return pet_data_trans

def read_data(data_path, class_names, each_set_portion ):
	# ex. class_names = ['AD', 'NC', 'MCI', 'LMCI']
	# ex. each_set_portion = [training, validate. testing] = [80, 10, 10]

	# 1. CREATE PET SET, MRI SET---------------------------
	## 1.A) CREATE ARRAY DATA
	pet_set = []
	mri_set = []
	
	count = 0
	for class_name in class_names:
		class_path = join(data_path, class_name)
		onlyfiles = [f for f in listdir(class_path) if isfile(join(class_path, f))]
		onlyfiles.sort() # 1. MRI hdr, 2) MRI img 3) PET hdr, 4) PET img
		
		pet_names = onlyfiles[1::4]
		mri_names = onlyfiles[3::4]

		pet_set.append([])
		mri_set.append([])

		i=0 # asign first element is a list of 68 of [79x95]
		pet_set[count] = read_nibabel(class_path, pet_names[i])
		mri_set[count] = read_nibabel(class_path, mri_names[i])

		# them assign from 1
		for i in range(1, len(pet_names)):
			pet_data_trans = read_nibabel(class_path, pet_names[i])
			mri_data_trans = read_nibabel(class_path, mri_names[i]) # mri and pet have same shape

			pet_set[count]= np.concatenate((pet_set[count], pet_data_trans), axis=0)
			mri_set[count]= np.concatenate((mri_set[count], mri_data_trans), axis=0)
		count = count + 1

		print ("load OK ")
	#print(np.shape(pet_set))
	all_set = [pet_set, mri_set]
	#print (np.shape(all_set[0][0]))
	#print (np.shape(all_set[0][1]))

	# 2. DIVIDE TO TRAIN, VALIDATE, TEST----------------------
	# ex. train_set[0] is PET traning, 
	# 	  train_set[1] is MRI traning. 
	train_set = []
	validate_set = []
	test_set = []

	# ex. num_train = [15, 10, 25, 30] with class (AD, NC, MCI, LMCI)
	#     means 15 AD training images, 10 NC traning images ...
	sizes = list(np.shape(f)[0] for f in pet_set)
	num_train = 	[int(size*each_set_portion[0]/100.0) for size in sizes]
	num_validate = 	[int(size*each_set_portion[1]/100.0) for size in sizes]
	num_test = [size - train - validate for size,train,validate in zip(sizes, num_train, num_validate)]
	print (num_train, num_validate, num_test)

	# Add data to each set
	for type_ind in range(2): # 0 is PET, 1 is MRI. For each modal
		train_set.append([])
		validate_set.append([])
		test_set.append([])

		count2 = 0; # for 0, append
		each_class_set = all_set[type_ind][count2]
		train_set[type_ind]		= 	each_class_set[ 											: num_train[count2] 						]
		validate_set[type_ind]	= 	each_class_set[ num_train[count2] 							: num_train[count2] + num_validate[count2] 	]
		test_set[type_ind]		= 	each_class_set[ num_train[count2] + num_validate[count2] 	: 											]


		for count2 in range(1,len(all_set[type_ind])): # for each class from 1, concat
			each_class_set = all_set[type_ind][count2]
			train_set[type_ind] 	= np.concatenate((train_set[type_ind]	, 		each_class_set[ :num_train[count2] ]											), axis = 0)
			validate_set[type_ind] 	= np.concatenate((validate_set[type_ind], 		each_class_set[ num_train[count2] : num_train[count2] + num_validate[count2] ]	), axis = 0)
			test_set[type_ind] 		= np.concatenate((test_set[type_ind]	, 		each_class_set[ num_train[count2] + num_validate[count2] : ]					), axis = 0)
			

	#print (np.shape(train_set))
	#print (np.shape(validate_set))
	#print (np.shape(test_set))
	

	# 3. CREATE LABEL
	train_label = []
	validate_label = []
	test_label = []

	for i in range(len(class_names)):
		specify_row = [0]*len(class_names)
		specify_row[i] = 1
		train_label = train_label + [specify_row for _ in range(num_train[i])]
		validate_label = validate_label + [specify_row for _ in range(num_validate[i])]
		test_label = test_label + [specify_row for _ in range(num_test[i])]
		#print (num_train[i], num_validate[i], num_test[i])

	# 4. CONVERT TO NUMPY ARRAY-------------------------------
	train_set = np.array(train_set)
	validate_set = np.array(validate_set)
	test_set = np.array(test_set)

	train_label = np.array(train_label)
	validate_label = np.array(validate_label)
	test_label = np.array(test_label)	

	adni = {'train': train_set, 'validate': validate_set, 'test': test_set}
	adni_label = {'train': train_label, 'validate': validate_label, 'test': test_label}
	
	print (np.shape(train_set), np.shape(validate_set), np.shape(test_set))
	print (np.shape(train_label), np.shape(validate_label), np.shape(test_label))

	return adni, adni_label


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
        batch_size = 40
        for batch_idx in range(0, len(features), batch_size):
        	#print('batch_idx', batch_idx)
        	if batch_idx + batch_size >= len(features):
        		batch_idx=0
        	images_batch = shuf_features[batch_idx:batch_idx+batch_size]
        	labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
        	yield images_batch, labels_batch


def main(_):
	mypath = '/home/ngoc/Desktop/CNNTensorFlow-master/_dataset/'
	adni, adni_label = read_data(mypath, ['AD', 'NC'], [80, 10, 10])

	mri_train = adni['train'][0];
	mri_test = adni['test'][0];

	label_train = adni_label['train'];
	label_test = adni_label['test'];

	iter_ = data_iterator(mri_train, label_train)
	iter2_ = data_iterator(mri_test, label_test)

	# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	
  # Create the model
	# x = tf.placeholder(tf.float32, [None, 784])
	x = tf.placeholder(tf.float32, [None, 79, 95])
	
	W_conv1 = weight_variable([7, 7, 1, 96])
	b_conv1 = bias_variable([96])
	
	x_image = tf.reshape(x, [-1, 79, 95, 1])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_3x3(h_conv1)
	print (x_image.get_shape())
	print_shape("h_conv1: ", h_conv1)
	print_shape("h_pool1: ", h_pool1)
	
	
	W_conv2 = weight_variable([5, 5, 96, 256])
	b_conv2 = bias_variable([256])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	print_shape("h_conv2: ", h_conv2)
	print_shape("h_pool2: ", h_pool2)

	W_conv3 = weight_variable([5, 5, 256, 384])
	b_conv3 = bias_variable([384])
	h_conv3 = tf.nn.relu(conv2dNoStride(h_pool2, W_conv3) + b_conv3)
	#h_pool3 = max_pool_2x2(h_conv3)
	print_shape("h_conv3: ", h_conv3)
	#print_shape("h_pool2: ", h_pool3)

	W_conv4 = weight_variable([3, 3, 384, 384])
	b_conv4 = bias_variable([384])
	h_conv4 = tf.nn.relu(conv2dNoStride(h_conv3, W_conv4) + b_conv4)
	print_shape("h_conv4: ", h_conv4)

	W_conv5 = weight_variable([3, 3, 384, 256])
	b_conv5 = bias_variable([256])
	h_conv5 = tf.nn.relu(conv2dNoStride(h_conv4, W_conv5) + b_conv5)
	h_pool5 = max_pool_3x3(h_conv5)
	print_shape("h_conv5: ", h_conv5)	
	print_shape("h_pool5: ", h_pool5)	

	W_fc1 = weight_variable([2 * 3 * 256, 2048])
	b_fc1 = bias_variable([2048])
	
	h_pool5_flat = tf.reshape(h_pool5, [-1, 2 * 3 * 256])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
	print_shape("h_fc1: ", h_fc1)

	W_fc2 = weight_variable([2048, 2048])
	b_fc2 = bias_variable([2048])
	
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)	
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
	
	W_fc3 = weight_variable([2048, 2])
	b_fc3 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])
	
	sess = tf.InteractiveSession()
	
	# Train
	tf.global_variables_initializer().run()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	for i in range(500):
		# batch = mnist.train.next_batch(50)
		xval, y_val = iter_.next()
		if i%10 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:xval, y_: y_val, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: xval, y_: y_val, keep_prob: 0.5})
	#print ("run %d ..."%i)
	#print("test accuracy %g"%accuracy.eval(feed_dict={x: mri_test, y_: label_test, keep_prob: 1.0}))

	for j in range(10):
		xxtest, y_test = iter2_.next()
		print("test accuracy %g"%accuracy.eval(feed_dict={ x: xxtest, y_: y_test, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
