from __future__ import absolute_import, division, print_function

if 0:
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	import tensorflow as tf
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



if 0:
	""" Linear Regression Example """

	

	import tflearn

	# Regression data
	X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
	Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

	# Linear Regression graph
	input_ = tflearn.input_data(shape=[None])
	linear = tflearn.single_unit(input_)
	regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
		                        metric='R2', learning_rate=0.01)
	m = tflearn.DNN(regression)
	m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

	print("\nRegression result:")
	print("Y = " + str(m.get_weights(linear.W)) +
	      "*X + " + str(m.get_weights(linear.b)))

	print("\nTest prediction for x = 3.2, 3.3, 3.4:")
	print(m.predict([3.2, 3.3, 3.4]))
	# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]




if 0:
	from autoencoder_duong.autoencoder_test import autoencoder_test_print
	autoencoder_test_print()
	
	from autoencoder_duong.read_data_from_analyze import *
	mypath = '/home/ngoc/Desktop/CNNTensorFlow-master/_dataset/'
	adni, adni_label = read_data(mypath, ['AD', 'NC'], [80, 0, 20])

	# cach 1: save file .dat (platform dependent)
	adni['train'][0].tofile('adni.dat')
	adni_label['train'].tofile('adni_label.dat')
	
	adni_t = np.fromfile('adni.dat', dtype=float)
	adni_label_t = np.fromfile('adni_label.dat', dtype=float)

	print (	adni['train'][0] == adni_t)
	print (	adni_label['train'] == adni_label_t)

	# cach 2: save file .npy (platform independent)
	np.save ('adni2.npy', adni['train'][0])
	adni2 = np.load('adni2.npy')
	
	print (	adni['train'][0] == adni2)

if 1:
	from autoencoder_duong.autoencoder_t import *
	run_autoencoder()

