from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
import random

# Data loading and preprocessing
from read_data_from_analyze import *
mypath = get_def_data_dir()
adni, adni_label = read_data(mypath, ['AD', 'NC'], [80, 0, 20])

mri_train = adni['train'][0];
mri_test = adni['test'][0];
label_train = adni_label['train'];
label_test = adni_label['test'];

print (np.shape(mri_train))
print (np.shape(mri_test))
print (np.shape(label_train))
print (np.shape(label_test))

# NEED TO FIX
exit()
patch_train = []
patch_test = []

for i in range(5, 105):
	r1 = random.randrange(0, 72)
	r2 = random.randrange(0, 88)
	r3 = random.randrange(0, 59)
	#print(r1,r2,r3)
	patch_train.append(mri_train[i][r1:r1+7, r2:r2+7, r3:r3+7])

# Building the encoder
encoder = tflearn.input_data(shape=[None, 7, 7, 7])
encoder = tflearn.fully_connected(encoder, 256)
encoder = tflearn.fully_connected(encoder, 64)

# Building the decoder
decoder = tflearn.fully_connected(encoder, 256)
tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
tf.nn.conv3d_transpose(x, W, output_shape = [], strides=[1, 2, 2, 2, 1], padding='SAME')
tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


decoder = np.reshape(decoder, (decoder.get_shape()[0],7,7,7))
exit()
# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
          run_id="auto_encoder", batch_size=256)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)
print(encoding_model.predict([X[0]]))

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")
testX = tflearn.data_utils.shuffle(testX)[0]
# Applying encode and decode over test set
encode_decode = model.predict(testX)
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(testX[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()
