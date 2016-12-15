#In this file we simulate some polynomial data with noise
#and then fit a second order polynomial to the data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#create the data and add gaussian noise to the output Y
#note need the float 32 because numpy default is float64
#and tensorflow default is float32
X1 = np.random.rand(1,100).astype(np.float32)
X2 = X1**2
X = np.vstack((X1,X2))
Y = 2*X[0,:] + 0.2*X[1,:] + 1 + 0.05*np.random.normal(size=100)

#set up the tensorflow graph
W = tf.Variable(tf.random_uniform([2,1],0,4.0))
b = tf.Variable(tf.zeros([1]))


yhat = tf.matmul(tf.transpose(W),X) + b

#set up loss function to minimize
loss = tf.reduce_mean(tf.square(Y-yhat)) + 0.05*b*b + 0.05*tf.reduce_mean(tf.square(W)) 
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#run the gradient descent by repeatedly calling sess.run(train)
#this runs the operation specified by the train variable above
for step in xrange(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(W), sess.run(b), sess.run(loss))


print Y.shape
print X.shape

plt.plot(X[1,:], Y.T, 'ro', label='data')
plt.plot(X[1,:], sess.run(yhat).T, 'gx', label='linear regression fit')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.savefig('plot.png')
