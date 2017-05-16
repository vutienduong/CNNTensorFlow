from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib

from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling2D
from keras.models import Model, Sequential

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
	num_train_ite = 405
	num_test_ite = 102
	psize = 5

	mypath = '/home/ngoc/Desktop/_dataset/all MRI 1523/'
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

	if 0:
		ini_weight, ini_weight2 = test_simple_ae(mri_train, mri_test)

		print(np.shape(ini_weight)) #weight encode => Transpose (e.g 2 x 2 x 2 * 5, size (5,8)=> Transpose, size (2,2,2,5))
		print(np.shape(ini_weight2)) # weight decode (e.g size (2,2,2, 5)

		ini_weight_t = np.transpose(ini_weight, (1, 2, 3, 0))
		#print ("ini_weight_t transpose: ", ini_weight_t)

		ini_weight_t = np.reshape(ini_weight_t, (psize,psize,psize, 1, 32))
		#print ("ini_weight_t reshape: ", ini_weight_t)

		#W_conv1 =  tf.Variable(ini_weight_t)
		W_conv1 = ini_weight_t #constant

	# CNN Training parameters
	batch_size = 2
	nb_classes = 2
	nb_epoch = 50

	# number of convolutional filters to use at each layer
	nb_filters = [32, 32]

	# level of pooling to perform at each layer (POOL x POOL)
	nb_pool = [3, 3]

	# level of convolution to perform at each layer (CONV x CONV)
	nb_conv = [5,5]

	# Pre-processing
	mri_train = mri_train.astype('float32')
	mri_train -= np.mean(mri_train)
	mri_train /=np.max(mri_train)

	mri_test = mri_test.astype('float32')
	mri_test -= np.mean(mri_test)
	mri_test /=np.max(mri_test)	


	model = Sequential()
	model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], input_shape=(1, 79, 95, 68, 1), activation='relu'))

	model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(128, init='normal', activation='relu'))

	model.add(Dropout(0.5))

	model.add(Dense(nb_classes,init='normal'))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

	# Split the data
	#mri_train, mri_test, label_train,label_test =  train_test_split(mri_train, label_train, test_size=0.2, random_state=4)


	# Train the model
	hist = model.fit(mri_train, label_train, validation_data=(mri_test,label_test),
	          batch_size=batch_size,nb_epoch = nb_epoch,show_accuracy=True,shuffle=True)


	#hist = model.fit(train_set, Y_train, batch_size=batch_size,
	#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
	#           shuffle=True)


	# Evaluate the model
	score = model.evaluate(mri_test,label_test, batch_size=batch_size, show_accuracy=True)
	print('Test score:', score[0])
	print('Test accuracy:', score[1]) 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()