from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib

# Import data
from tensorflow.examples.tutorials.mnist import input_data
import autoencoder_duong.my_conv_autoencoder as myae

import tensorflow as tf

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

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

def max_pool_2x2_stride(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def print_type_name(x):
	print(type(x).__name__)

# read input and create a 4D tensor
def read_nibabel(class_path,img_name):
	pet_img = nib.load(join(class_path, img_name))
	pet_data = pet_img.get_data()
	pet_data_trans = np.transpose(pet_data, (2, 0, 1)) # 79x95x68 => 68 x 79 x 95
	# pet_data_trans = np.transpose(pet_data, (1, 0, 2)) # 79x95x68 => 95 x 79 x 68
	pet_data_trans = pet_data_trans[30:40] # remove some at begin and some slices at end
	return pet_data_trans

# include mri and PET
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
		print(onlyfiles[0])
		print(onlyfiles[1])
		print(onlyfiles[2])
		
		pet_names = onlyfiles[1::4]
		mri_names = onlyfiles[3::4]
		print("PET, MRI shape", np.shape(pet_names), np.shape(mri_names))
		exit()

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

		print ("load OK: ", class_name)
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


def read_mri_data(data_path, class_names, each_set_portion ):
	# ex. class_names = ['AD', 'NC', 'MCI', 'LMCI']
	# ex. each_set_portion = [training, validate. testing] = [80, 10, 10]

	# 1. CREATE PET SET, MRI SET---------------------------
	## 1.A) CREATE ARRAY DATA
	mri_set = []
	
	count = 0
	for class_name in class_names:
		class_path = join(data_path, class_name)
		onlyfiles = [f for f in listdir(class_path) if isfile(join(class_path, f))]
		onlyfiles.sort() # 1. MRI hdr, 2) MRI img 3) PET hdr, 4) PET img
		#print(onlyfiles[0])
		#print(onlyfiles[1])
		#print(onlyfiles[2])
		
		mri_names = onlyfiles[0::2]
		#print(mri_names[0])
		#print(mri_names[1])

		mri_set.append([])

		i=0 # asign first element is a list of 68 of [79x95]
		mri_set[count] = read_nibabel(class_path, mri_names[i])

		# them assign from 1
		for i in range(1, len(mri_names)):
			mri_data_trans = read_nibabel(class_path, mri_names[i]) # mri and pet have same shape
			mri_set[count]= np.concatenate((mri_set[count], mri_data_trans), axis=0)
		count = count + 1
		print ("load OK: ", class_name)

	# 2. DIVIDE TO TRAIN, VALIDATE, TEST----------------------
	# ex. train_set[0] is PET traning, 
	# 	  train_set[1] is MRI traning. 
	train_set = []
	validate_set = []
	test_set = []

	# ex. num_train = [15, 10, 25, 30] with class (AD, NC, MCI, LMCI)
	#     means 15 AD training images, 10 NC traning images ...
	sizes = list(np.shape(f)[0] for f in mri_set)
	num_train = 	[int(size*each_set_portion[0]/100.0) for size in sizes]
	num_validate = 	[int(size*each_set_portion[1]/100.0) for size in sizes]
	num_test = [size - train - validate for size,train,validate in zip(sizes, num_train, num_validate)]
	print (num_train, num_validate, num_test)

	# Add data to each set
	#train_set.append([])
	#validate_set.append([])
	#test_set.append([])

	count2 = 0; # for 0, append
	each_class_set = mri_set[count2]
	train_set		= 	each_class_set[ 											: num_train[count2] 						]
	validate_set	= 	each_class_set[ num_train[count2] 							: num_train[count2] + num_validate[count2] 	]
	test_set		= 	each_class_set[ num_train[count2] + num_validate[count2] 	: 											]


	for count2 in range(1,len(mri_set)): # for each class from 1, concat
		each_class_set = mri_set[count2]
		train_set 	= np.concatenate((train_set	, 		each_class_set[ :num_train[count2] ]											), axis = 0)
		validate_set 	= np.concatenate((validate_set, 		each_class_set[ num_train[count2] : num_train[count2] + num_validate[count2] ]	), axis = 0)
		test_set 		= np.concatenate((test_set	, 		each_class_set[ num_train[count2] + num_validate[count2] : ]					), axis = 0)
			

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
        		print("BATCH ID MAX")
        		batch_idx=0
        	images_batch = shuf_features[batch_idx:batch_idx+batch_size]
        	labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
        	yield images_batch, labels_batch


def main(_):
	myae.run_ae_2d()
	mypath = '/home/ngoc/Desktop/_dataset/all MRI 1523'
	adni, adni_label = read_mri_data(mypath, ['AD', 'NC'], [80, 10, 10])
	
	#saved_dir = '/home/ngoc/Desktop/CNNTensorFlow-master/autoencoder_duong/stored_temp_data/'
	#mri_train_name = 'middle_slice_train.npy'
	#mri_test_name = 'middle_slice_test.npy'
	#label_train_name = 'middle_slice_train_label.npy'
	#label_test_name = 'middle_slice_test_label.npy'

	mri_train = adni['train'];
	mri_test = adni['test'];

	label_train = adni_label['train'];
	label_test = adni_label['test'];

	
	if 0: # save
		np.save(join(saved_dir, mri_train_name), mri_train);
		np.save(join(saved_dir, mri_test_name), mri_test);
		np.save(join(saved_dir, label_train_name), label_train);
		np.save(join(saved_dir, label_test_name), label_test);

	if 0: #load
		mri_train = np.load(join(saved_dir, mri_train_name))
		mri_test = np.load(join(saved_dir, mri_test_name))
		label_train = np.load(join(saved_dir, label_train_name))
		label_test = np.load(join(saved_dir, label_test_name))

	#exit();


	if 0: # plot middle slice to see
		plt.figure(figsize=(10, 4))
		for i in range(20):
		    # display AD
		    ax = plt.subplot(4, 10, i + 1)
		    plt.imshow(mri_train[i+1].reshape(79, 95))
		    #plt.imshow(mri_train[i+1].reshape(79, 68))
		    plt.gray()
		    ax.get_xaxis().set_visible(False)
		    ax.get_yaxis().set_visible(False)
		    print(label_train[i+1])

		    # display NC
		    ax = plt.subplot(4, 10, i + 21)
		    plt.imshow(mri_train[i+300].reshape(79, 95))
		    #plt.imshow(mri_train[i+300].reshape(79, 68))
		    plt.gray()
		    ax.get_xaxis().set_visible(False)
		    ax.get_yaxis().set_visible(False)
		    print(label_train[i+300])

		plt.show()
		exit() # DEBUG

	
	print("mri_train, mri_test shape: ", np.shape(mri_train), np.shape(mri_test))
	print("label_train, label_test shape: ", np.shape(label_train), np.shape(label_test))
	dim1 = np.shape(mri_train)[1]
	dim2 = np.shape(mri_train)[2]

	train_ite = int(np.shape(mri_train)[0]/40)
	test_ite = int(np.shape(mri_test)[0]/40)
	print("traint_ite, test_ite: ", train_ite, test_ite)

	iter_ = data_iterator(mri_train, label_train)
	iter2_ = data_iterator(mri_test, label_test)

	# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	
  # Create the model
	# x = tf.placeholder(tf.float32, [None, 784])
	x = tf.placeholder(tf.float32, [None, dim1, dim2])
	x = tf.div(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x))) # normalize
	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1, dim1, dim2, 1]) # hard code
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_3x3(h_conv1)

	print (x_image.get_shape())
	print_shape("h_conv1: ", h_conv1)
	print_shape("h_pool1: ", h_pool1)
	exit()
	
	# W_conv2 = weight_variable([5, 5, 96, 256])
	# b_conv2 = bias_variable([256])
	# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	# h_pool2 = max_pool_2x2(h_conv2)
	# print_shape("h_conv2: ", h_conv2)
	# print_shape("h_pool2: ", h_pool2)

	# W_conv3 = weight_variable([5, 5, 256, 384])
	# b_conv3 = bias_variable([384])
	# h_conv3 = tf.nn.relu(conv2dNoStride(h_pool2, W_conv3) + b_conv3)
	# #h_pool3 = max_pool_2x2(h_conv3)
	# print_shape("h_conv3: ", h_conv3)
	# #print_shape("h_pool2: ", h_pool3)

	# W_conv4 = weight_variable([3, 3, 384, 384])
	# b_conv4 = bias_variable([384])
	# h_conv4 = tf.nn.relu(conv2dNoStride(h_conv3, W_conv4) + b_conv4)
	# print_shape("h_conv4: ", h_conv4)

	# W_conv5 = weight_variable([3, 3, 384, 256])
	# b_conv5 = bias_variable([256])
	# h_conv5 = tf.nn.relu(conv2dNoStride(h_conv4, W_conv5) + b_conv5)
	# h_pool5 = max_pool_3x3(h_conv5)
	# print_shape("h_conv5: ", h_conv5)	
	# print_shape("h_pool5: ", h_pool5)	

	W_fc1 = weight_variable([18 * 22 * 32, 256])
	b_fc1 = bias_variable([256])
	
	h_pool5_flat = tf.reshape(h_pool1, [-1, 18 * 22 * 32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
	print_shape("h_fc1: ", h_fc1)

	# W_fc2 = weight_variable([2048, 2048])
	# b_fc2 = bias_variable([2048])
	
	# h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)	
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# softmax
	W_fc3 = weight_variable([256, 2])
	b_fc3 = bias_variable([2])
	
	y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 2])

	sess = tf.InteractiveSession()
	
	# Train
	tf.global_variables_initializer().run()


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	for i in range(train_ite):
		# batch = mnist.train.next_batch(50)
		xval, y_val = iter_.next()
		if i%5 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:xval, y_: y_val, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: xval, y_: y_val, keep_prob: 0.5})
	#print ("run %d ..."%i)
	#print("test accuracy %g"%accuracy.eval(feed_dict={x: mri_test, y_: label_test, keep_prob: 1.0}))

	resl = []
	for j in range(test_ite):
		xxtest, y_test = iter2_.next()
		acc = accuracy.eval(feed_dict={ x: xxtest, y_: y_test, keep_prob: 1.0})
		print("test accuracy %g"%acc)
		resl.append(acc)
	print("AVG test accuracy %g"%np.mean(resl))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
