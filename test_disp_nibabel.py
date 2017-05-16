import matplotlib.pyplot as plt
import nibabel as nib
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf

epi_img = nib.load('/home/ngoc/Desktop/CNNTensorFlow-master/_dataset/AD/wADNI_003_S_1059_MR_MPR__GradWarp__B1_Correction__N3_Br_20070501173419666_S22301_I52811.img')
epi_img_data = epi_img.get_data()
print epi_img_data.shape


mypath = '/home/ngoc/Desktop/CNNTensorFlow-master/_dataset/'
subpath = 'AD'

fullpath = join(mypath, subpath)

def print_type_name(x):
	print(type(x).__name__)

onlyfiles = [f for f in listdir(fullpath) if isfile(join(fullpath, f))]

onlyfiles.sort()
print (len(onlyfiles))

petf = onlyfiles[1::4]
mrif = onlyfiles[3::4]

#for i in range(4,6):
#	print i
	#print (onlyfiles[i])

print len(petf)
print len(mrif)


def test_np_array():
	pet_set = []
	mri_set = []
	for i in range (3):
		pet_set.append([])
		mri_set.append([])
		for j in range (2):
			pet_set[i].append([j+i, j*j])
			mri_set[i].append((j+i)*1.5)

	pet_set = np.array(pet_set)
	mri_set = np.array(mri_set)
	print pet_set
	print mri_set

	print pet_set[1]
	print mri_set[2][0]

def read_nibabel(class_path,img_name):
	pet_img = nib.load(join(class_path, img_name))
	pet_data = pet_img.get_data()
	pet_data_trans = np.transpose(pet_data, (2, 0, 1)) # 79x95x68 => 68 x 79 x 95
	pet_data_trans = pet_data_trans[14:49] # remove some at begin and some slices at end
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
	print ('num_train, num_validate, num_test')
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
	#print test_label[450:500]

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


def show_slices(img, slice_index):
	slice_0 = img[slice_index[0], :, :]
	slice_1 = img[:, slice_index[1], :]
	slice_2 = img[:, :, slice_index[2]]

	slices = [slice_0, slice_1, slice_2]

	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")
	
	plt.suptitle("Center slices for EPI image")  
	plt.show()



def show_slices2(img):
	slices =[]
	for i in range(60,68):
		slices.append(img[:, :, i])
	fig, axes = plt.subplots(5,6)
	for i, slice in enumerate(slices):
		axes[i/6][i%6].imshow(slice.T, cmap="gray", origin="lower")
	plt.suptitle("Center slices for EPI image")  
	plt.show()



def show_imgs(imgs):
	fig, axes = plt.subplots(10,7,squeeze=False)
	for i, slice in enumerate(imgs):
		axes[i/7][i%7].imshow(slice.T, cmap="gray", origin="lower")
	plt.suptitle("Center slices for EPI image")  
	plt.show()

# run for test
# test_np_array()
[adni, adni_label] = read_data(mypath, ['AD', 'NC'], [80, 10, 10])
#print (np.shape(train), np.shape(validate), np.shape(test))
print (np.shape(adni['train'][1]), np.shape(adni['validate'][1]), np.shape(adni['test'][1]))
#print (np.shape(adni_label['train']), np.shape(adni_label['validate']), np.shape(adni_label['test']))

#show_slices(adni['train'][1][3], [10, 10, 66])

test_img = adni['train'][1][3]
print(np.shape(test_img))
show_imgs(adni['train'][0][:68])
#print(test_img)
#show_slices(test_img, [10, 10, 66])


aa = [[[1,2,3,4],[5,6,7,8],[9,10,11,12]], [[21,22,23,24],[25,26,27,28],[29,30,31,32]]]
aa = np.array(aa)
print(aa[:,:,0])


#print(np.shape(aa))

bb = np.transpose(aa, (2,0,1))
print(bb[0,:,:])

#show_slices2(adni['train'][1][3])





# img=mpimg.imread('games.png')
# imgplot = plt.imshow(img)

# plt.plot(range(20),range(20))
