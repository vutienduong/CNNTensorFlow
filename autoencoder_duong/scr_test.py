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

from my_conv_autoencoder import *
import read_data_from_analyze as rdfa

def test():
	mypath = rdfa.get_data_dir('old/AD')
	#file_path = join(mypath, 'wADNI_003_S_1059_MR_MPR__GradWarp__B1_Correction__N3_Br_20070501173419666_S22301_I52811.hdr') #MRI
	file_path = join(mypath, 'wADNI_003_S_1059_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20070122123023427_1_S25144_I37090.hdr') # PET
	pet_img = nib.load(file_path)
	pet_data = pet_img.get_data()
	pet_data = np.array(pet_data)
	# pet_data = np.nan_to_num(pet_data)
	# rdfa.print_type_name(pet_data)
	# print(np.shape(pet_data))
	# pet_data2 = pet_data[50:52,:,10:14]
	# print(np.shape(pet_data2))
	#print(pet_data2)

	plt.figure(figsize=(10, 4))
	for i in range(40):
	    # display AD
	    ax = plt.subplot(4, 10, i + 1)
	    plt.imshow(pet_data[i, :, :])#.reshape(79, 95))
	    #plt.imshow(mri_train[i+1].reshape(79, 68))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # # display NC
	    # ax = plt.subplot(4, 10, i + 21)
	    # plt.imshow(mri_train[i+300].reshape(79, 95))
	    # #plt.imshow(mri_train[i+300].reshape(79, 68))
	    # plt.gray()
	    # ax.get_xaxis().set_visible(False)
	    # ax.get_yaxis().set_visible(False)
	    # print(label_train[i+300])

	plt.show()

def main(_):
	mypath = rdfa.get_data_dir('all MRI 1523')
	#adni, adni_label = rdfa.read_only_middle_slice(mypath, ['AD', 'NC'], [80, 0, 20], 1)
	#print(np.shape(adni['train']), np.shape(adni_label))
	test()
	
	


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
