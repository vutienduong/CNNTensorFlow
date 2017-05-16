from __future__ import print_function
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import random
import numpy as np
from read_data_from_analyze import *
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras import regularizers
import h5py
# create 3d patches from adni mri input, with size, e.g. 5 x 5 x 5
# INPUT default 
# + patch_size = 5
# + num_patch_each_img = 1000
# + num_img = 110
#
# OUTPUT default
# + patch_train = 88,000
# + patch_test = 22,000

def create_3d_patch(patch_size, num_patch_each_img, num_img, mri_train=None, mri_test=None):
    # Data loading and preprocessing
    verbose_flag = 0;
    if mri_train is None and mri_test is None:
        mypath = '/home/ngoc/Desktop/CNNTensorFlow-master/_dataset/'
        adni, adni_label = read_data(mypath, ['AD', 'NC'], [80, 0, 20])

        mri_train = adni['train'][0];
        mri_test = adni['test'][0];

        label_train = adni_label['train'];
        label_test = adni_label['test'];
        
    print (np.shape(mri_train), np.shape(mri_test))
    

    if verbose_flag:
        print ("-- SIZE OF TRAIN, TEST DATA --")
        print (np.shape(mri_train))
        print (np.shape(mri_test))
        #print (np.shape(label_train))
        #print (np.shape(label_test))

    patch_train = []
    patch_test = []

    if verbose_flag:
        for i in range(len(mri_train)):
            if np.any(np.isnan(mri_train[i])):
                print ("is nan train")
                #print (x_train[i])
                exit()

        for i in range(len(mri_test)):
            if np.any(np.isnan(mri_test[i])):
                print ("is nan")


    for i in range(0, 110): # TODO: hardcode
        for j in range(0, num_patch_each_img):
            r1 = random.randrange(0, 79 - patch_size) # TODO: hardcode
            r2 = random.randrange(0, 95 - patch_size) # TODO: hardcode
            r3 = random.randrange(0, 68 - patch_size) # TODO: hardcode
            patch = mri_train[i][r1:r1+patch_size, r2:r2+patch_size, r3:r3+patch_size]
            patch_train.append(patch)

    random.shuffle(patch_train)

    num_patches = num_patch_each_img * num_img
    num_train = int(0.8* num_patches)
    num_test = num_patches - num_train

    if verbose_flag:
        print ("-- Number of training patches. testing patches --")
        print (num_train, num_test)

    patch_test = patch_train[num_train:num_patches]
    patch_train = patch_train[0:num_train]

    if verbose_flag:
        print ("-- SIZE of patch_train, patch_test --")
        print (np.shape(patch_train), np.shape(patch_test))

    patch_train = np.array(patch_train)
    patch_test = np.array(patch_test)
    return patch_train, patch_test


def display_2d_img(img):
    plt.figure()
    plt.imshow(img)
    plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    plt.show()

def display_list_patch(patch_train):
    n=10
    plt.figure(figsize=(14, 20))
    for i in range(10):
        for j in range(5):
            # display original
            ax = plt.subplot(i, j, j*5+i)
            plt.imshow(patch_train[i][:,:,j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def display_list_patch2(patch_train):
    fig, axes = plt.subplots(10,5,squeeze=False)
    for i in range(10):
        for j in range(5):
            # display original
            axes[i][j].imshow(patch_train[i][:,:,j], cmap="gray", origin="lower")
            # plt.gray()
            #axes.get_xaxis().set_visible(False)
            #axes.get_yaxis().set_visible(False)
    plt.show()

def test_reshape(img):
    fig, axes = plt.subplots(2,1,squeeze=False)
    img2 = np.reshape(img, (125))
    img2 = np.reshape(img2, (5,5,5))
    axes[0][0].imshow(img[:,:,1], cmap="gray", origin="lower")
    axes[1][0].imshow(img2[:,:,1], cmap="gray", origin="lower")
    plt.show()

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def test_simple_ae(mri_train = None, mri_test = None):
    psize = 5
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 584 floats

    # this is our input placeholder
    input_img = Input(shape=(psize * psize * psize,))

    # add a Dense layer with a L1 activity regularizer
    # encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-4))(input_img)


    # relu + sigmoid
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(psize * psize * psize, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input=input_img, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train, x_test = create_3d_patch(psize, 1000, 110, mri_train, mri_test)
    #print(type(x_train).__name__)

    #print(x_train[1][:,:,1])
    # print len(x_train)

    if 0:
        for i in range(len(x_train)):
            if np.any(np.isnan(x_train[i])):
                print ("is nan train")
                exit()
        for i in range(len(x_test)):
            if np.all(np.isnan(x_test[i])):
                print ("is nan")

    # FIND Max element
    max_elem = 0
    for i in range(len(x_train)):
        temp_elem = np.amax(x_train[i])
        if temp_elem > max_elem:
            max_elem = temp_elem

    for i in range(len(x_test)):
        temp_elem = np.amax(x_test[i])
        if temp_elem > max_elem:
            max_elem = temp_elem

    if 0:
        print(max_elem)

    x_train = x_train.astype('float32') / max_elem
    x_test = x_test.astype('float32') / max_elem
    print(x_train[1][:,:,1])

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print (x_train.shape)
    print (x_test.shape)

    autoencoder.fit(x_train, x_train,
                    nb_epoch=5,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    autoencoder.save_weights('save_weights')

    all_weight = autoencoder.layers[1].get_weights()[0]
    all_weight2 = all_weight.transpose()
    all_weight2 = np.reshape(all_weight2, (encoding_dim,psize,psize,psize))

    all_weight3 = autoencoder.layers[2].get_weights()[0]
    all_weight3 = np.reshape(all_weight3, (encoding_dim,psize,psize,psize))    

    if 0:
        print(all_weight)
    
    #print (np.shape(all_weight2))
    #print (np.shape(all_weight3))

    if 0:
        for layer in autoencoder.layers:
            weights = layer.get_weights() # list of numpy arrays
            print(layer)
            print ("--weight shape")
            for weight in weights:
                print(np.shape(weight))
                print(weight)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt
    if 0:
        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = encoder.predict(x_test)
        decoded_imgs = decoder.predict(encoded_imgs)

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(psize, psize, psize)[:,:,2])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(psize, psize, psize)[:,:,2])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    return all_weight2, all_weight3

def test_variable_initial():
    var_a = tf.truncated_normal([5, 5, 1], stddev=0.1)
    #sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())
    #print(var_a)
    #sess.close()
    with tf.Session() as sess:
        result = sess.run([var_a])
        print(result)

#test_simple_ae()
#print_structure('save_weights')
#test_variable_initial()