from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import nibabel as nib

import tensorflow as tf

from os import listdir
from os.path import isfile, join

# print name of type of variable x
def print_type_name(x):
	print(type(x).__name__)

# read input and create a 4D tensor, dung cho read_only_middle_slice
def read_nibabel(class_path,img_name):
	pet_img = nib.load(join(class_path, img_name))
	pet_data = pet_img.get_data()
	#pet_data_trans = np.transpose(pet_data, (2, 0, 1)) # 79x95x68 => 68 x 79 x 95
	pet_data_trans = pet_data_trans[39:40] # remove some at begin and some slices at end
	return pet_data_trans

# read input and create a 4D tensor
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


		for i in range(len(pet_names)):
			pet_name = pet_names[i]
			mri_name = mri_names[i]

			pet_img = nib.load(join(class_path, pet_name))
			mri_img = nib.load(join(class_path, mri_name))

			pet_data = pet_img.get_data()
			mri_data = mri_img.get_data()

			#pet_data2 = pet_data[20:60, : , 20:60]
			#mri_data2 = mri_data[20:60, : , 20:60]

			pet_set[count].append(pet_data)
			mri_set[count].append(mri_data)	

		count = count + 1
		print ("load OK ")

	#print (np.shape(pet_set[0]))
	all_set = [pet_set, mri_set]

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
	#print (num_train, num_validate, num_test)

	# Add data to each set
	for type_ind in range(2): # 0 is PET, 1 is MRI. For each modal
		train_set.append([])
		validate_set.append([])
		test_set.append([])

		count2 = 0;
		#idxs = np.arange(0, len(each_class_set))
		#np.random.shuffle(idxs)
		#each_class_set = each_class_set[idxs]
		for each_class_set in all_set[type_ind]: # for each class
			train_set[type_ind] = train_set[type_ind] + each_class_set[ :num_train[count2] ]
			validate_set[type_ind] = validate_set[type_ind] + each_class_set[ num_train[count2] : num_train[count2] + num_validate[count2] ]
			test_set[type_ind] = test_set[type_ind] + each_class_set[ num_train[count2] + num_validate[count2] : ]
			count2 = count2 + 1;

	# 3. CREATE LABEL
	#train_label = [ [0]*len(class_names) for _ in xrange(len(train_set[0])) ];
	#validate_label = [ [0]*len(class_names) for _ in xrange(len(validate_set[0])) ];
	#test_label = [ [0]*len(class_names) for _ in xrange(len(test_set[0])) ];

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

	train_set = np.nan_to_num(train_set)
	validate_set = np.nan_to_num(validate_set)
	test_set = np.nan_to_num(test_set)

	train_label = np.array(train_label)
	validate_label = np.array(validate_label)
	test_label = np.array(test_label)	

	adni = {'train': train_set, 'validate': validate_set, 'test': test_set}
	adni_label = {'train': train_label, 'validate': validate_label, 'test': test_label}

	return adni, adni_label


def read_data2(data_path, class_names, each_set_portion, load_mode=False ):
	# ex. class_names = ['AD', 'NC', 'MCI', 'LMCI']
	# ex. each_set_portion = [training, validate. testing] = [80, 10, 10]

	# 1. CREATE PET SET, MRI SET---------------------------
	## 1.A) CREATE ARRAY DATA
	mri_set = []
	#load_mode = True #False is load from data, True is load from .npy
	load_file = 'mri1523.npy'
	load_lb_file = 'mri1523_lb.npy'
	if not load_mode:
		count = 0
		for class_name in class_names:
			class_path = join(data_path, class_name)
			onlyfiles = [f for f in listdir(class_path) if isfile(join(class_path, f))]
			onlyfiles.sort() # 1. MRI hdr, 2) MRI img 3) PET hdr, 4) PET img

			mri_names = onlyfiles[1::2]
			print(np.shape(mri_names))
			mri_set.append([])

			for i in range(len(mri_names)):
				mri_name = mri_names[i]
				mri_img = nib.load(join(class_path, mri_name))
				mri_data = mri_img.get_data()
				#mri_data2 = mri_data[20:60, : , 20:60]
				mri_set[count].append(mri_data)	
				#mri_set[count].append(mri_data.copy())	
			count = count + 1
			print ("load OK ")
		#print (np.shape(pet_set[0]))
		all_set = [mri_set]

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
		#print (num_train, num_validate, num_test)

		# Add data to each set
		for type_ind in range(1): # 0 is PET, 1 is MRI. For each modal
			train_set.append([])
			validate_set.append([])
			test_set.append([])

			count2 = 0;
			#idxs = np.arange(0, len(each_class_set))
			#np.random.shuffle(idxs)
			#each_class_set = each_class_set[idxs]
			for each_class_set in all_set[type_ind]: # for each class
				train_set[type_ind] = train_set[type_ind] + each_class_set[ :num_train[count2] ]
				validate_set[type_ind] = validate_set[type_ind] + each_class_set[ num_train[count2] : num_train[count2] + num_validate[count2] ]
				test_set[type_ind] = test_set[type_ind] + each_class_set[ num_train[count2] + num_validate[count2] : ]
				count2 = count2 + 1;

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

		train_set = np.nan_to_num(train_set)
		validate_set = np.nan_to_num(validate_set)
		test_set = np.nan_to_num(test_set)

		train_label = np.array(train_label)
		validate_label = np.array(validate_label)
		test_label = np.array(test_label)	


		adni = {'train': train_set, 'validate': validate_set, 'test': test_set}
		adni_label = {'train': train_label, 'validate': validate_label, 'test': test_label}

		#np.save(load_file, adni)
		#np.save(load_lb_file, adni_label)
	else:
		dpath = get_def_dir()
		load_file = join(dpath, load_file)
		load_lb_file = join(dpath, load_lb_file)
		adni = np.load(load_file)
		adni_label = np.load(load_lb_file)
	return adni, adni_label


# function: READ_ONLY_MIDDLE_SLICE
# use to read only middle slice of an image follow one of 3 direction: 1, 2, 3
# corresponding to (x, :, :), (:, x, :), (:, :, x)
def read_only_middle_slice(data_path, class_names, each_set_portion, direction ):
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

		# then assign from 1
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
	print("sizes :", sizes )
	exit() 

	num_train = 	[int(size*each_set_portion[0]/100.0) for size in sizes]
	num_validate = 	[int(size*each_set_portion[1]/100.0) for size in sizes]
	num_test = [size - train - validate for size,train,validate in zip(sizes, num_train, num_validate)]
	print ("num_train, num_validate, num_test ", num_train, num_validate, num_test)

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

def read_npy_file_3d(prefix=None):
	if prefix==None:
		prefix = '3d'

	saved_dir = join(get_def_dir(), 'autoencoder_duong/stored_temp_data/')
	saved_dir = join(saved_dir, prefix)

	mri_train_name = '3d_mri_train.npy'
	mri_test_name = '3d_mri_test.npy'
	label_train_name = '3d_label_train.npy'
	label_test_name = '3d_label_test.npy'

	mri_train = np.load(join(saved_dir, mri_train_name))
	mri_test = np.load(join(saved_dir, mri_test_name))
	label_train = np.load(join(saved_dir, label_train_name))
	label_test = np.load(join(saved_dir, label_test_name))

	return mri_train, mri_test, label_train, label_test

def read_npy_file_3d_norm():
	return read_npy_file_3d('3d_norm')

def get_def_dir():
	return '/home/duong/Desktop/CNNTensorflow/CNNTensorflow/'

def get_def_data_dir():
	return '/home/duong/Desktop/CNNTensorflow/_dataset'	

def get_data_dir(dir_name):
	return join(get_def_data_dir(), dir_name)

def get_specified_dir(sub_name):
	return join(get_def_dir(), sub_name)

def get_stored_temp_dir():
	return get_specified_dir('autoencoder_duong/stored_temp_data')