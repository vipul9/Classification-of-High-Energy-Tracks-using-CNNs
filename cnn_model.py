
"""
Author: Vipul Pawar
Project: https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs
"""
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
#import skflow
from tensorflow.contrib import learn as skflow
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
#import plotly.plotly as py
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
TRAIN_DATASET_PATH = './training_dataset' # the dataset file or root folder path.
TEST_DATASET_PATH = './testing_dataset'

# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 1 # The 3 color channels, change to 1 if grayscale


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.png') or sample.endswith('.jpg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

# -----------------------------------------------
# CNN
# -----------------------------------------------


# Parameters
learning_rate = 0.001
total_img = 11000 # Total Number of images
batch_size = 50 # Batch size for training
batch_size_test = 50 # Batch size for testing

epoch_steps = total_img/batch_size
num_epoch = 40 # Epoch is one complete presentation of data
num_steps = int(num_epoch * epoch_steps)
num_steps_test = int(total_img/batch_size_test)
display_step = 100
dropout = 0.5 # Dropout, probability to keep units

# Build the data input
X_train, Y_train = read_images(TRAIN_DATASET_PATH, MODE, batch_size)
X_train_test, Y_train_test = read_images(TRAIN_DATASET_PATH, MODE, batch_size_test)
X_test, Y_test = read_images(TEST_DATASET_PATH, MODE, batch_size_test)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv3 = tf.layers.conv2d(conv2, 64, 1, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 512)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X_train, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X_train, N_CLASSES, dropout, reuse=True, is_training=False)
logits_train_test = conv_net(X_train_test,N_CLASSES, dropout, reuse=True, is_training=False)
# Create another graph for testing that reuse the same weights
logits_test_test = conv_net(X_test, N_CLASSES, dropout, reuse=True, is_training=False)
# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_train, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred_train = tf.equal(tf.argmax(logits_train_test, 1), tf.cast(Y_train_test, tf.int64),name="correct_pred_train")
true_train, score_train, abs_score_train = tf.cast(Y_train_test, tf.int64), tf.cast(logits_train_test[:,1],tf.float32), tf.argmax(logits_train_test,1)
accuracy_train = tf.reduce_mean(tf.cast(correct_pred_train, tf.float32),name="train_accuracy")

correct_pred_test = tf.equal(tf.argmax(logits_test_test, 1), tf.cast(Y_test, tf.int64),name="correct_pred_test")
true_test, score_test, abs_score_test = tf.cast(Y_test, tf.int64), tf.cast(logits_test_test[:,1],tf.float32), tf.argmax(logits_test_test,1)
accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32),name="test_accuracy")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()
sum_test = 0
sum_train = 0

def pred(y_true,y_score,isSignal):
    pred=list()
    if isSignal:
        for i in range(0,len(y_true)):
            if y_true[i]==1:
                pred.append(y_score[i])
    if not isSignal:
        for i in range(0,len(y_true)):
            if y_true[i]==0:
                pred.append(y_score[i])
    return pred
# Start training

def roc(y_true,y_score):
    thresholds = np.linspace(0,1,101)
    ROC = np.zeros((101,2))
    l=len(y_true)
    Y1 = np.zeros(l,dtype=bool)
    Y2 = np.zeros(l,dtype=bool)
    for i in range(0,l):
        Y1[i] = y_true[i] == 1
        Y2[i] = y_true[i] == 0
    for i in range(101):
        t = thresholds[i]
        TP_t = np.logical_and( y_score > t, Y1 ).sum()
        TN_t = np.logical_and( y_score <=t, Y2 ).sum()
        FP_t = np.logical_and( y_score > t, Y2 ).sum()
        FN_t = np.logical_and( y_score <=t, Y1 ).sum()
        FPR_t = float(FP_t) / float(FP_t + TN_t)
        ROC[i,0] = FPR_t
        TPR_t = float(TP_t) / float(TP_t + FN_t)
        ROC[i,1] = TPR_t
        if t>0.49 and t<0.51:
            print("t = "+"{:.4f}".format(t)+", True positive rate = "+"{:.4f}".format(TPR_t)+", True Negetive Rate =  "+"{:.4f}".format(1-FPR_t))
    return ROC[:,0],ROC[:,1]

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1,num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")
    y_true_train, y_score_train, y_abs_score_train, y_true_test, y_score_test, y_abs_score_test = list(),list(),list(),list(),list(),list()
    for step in range(1,num_steps_test+1):
        train_acc = sess.run(accuracy_train)
        test_acc = sess.run(accuracy_test)
        sum_train = float(sum_train) + float(train_acc)
        sum_test = float(sum_test) + float(test_acc)
        y_trueTrain,y_scoreTrain,y_abs_scoreTrain = sess.run([true_train,score_train,abs_score_train])
        y_trueTest, y_scoreTest,y_abs_scoreTest = sess.run([true_test,score_test,abs_score_test])
        y_true_train.extend(y_trueTrain)
        y_score_train.extend(y_scoreTrain)
        y_abs_score_train.extend(y_abs_scoreTrain)
        y_true_test.extend(y_trueTest)
        y_score_test.extend(y_scoreTest)
        y_abs_score_test.extend(y_abs_scoreTest)
        print("Step " + str(step) + ", Training Accuracy= " + \
                  "{:.4f}".format(train_acc) + ", Testing Accuracy= "+"{:.4f}".format(test_acc))
    average_train=float(sum_train/num_steps_test)
    average_test=float(sum_test/num_steps_test)
    print("Training Accuracy = "+ "{:.4f}".format(average_train) + ", Testing Accuracy = " + "{:.4f}".format(average_test))
    
    sig_pred_test=pred(y_true_test,y_score_test,isSignal=True)
    bg_pred_test=pred(y_true_test,y_score_test,isSignal=False)
    sig_pred_train=pred(y_true_train,y_score_train,isSignal=True)
    bg_pred_train=pred(y_true_train,y_score_train,isSignal=False)
    #FPR, TPR, _ = roc_curve(y_true_test,y_abs_score_test,pos_label=1)
    FPR,TPR = roc(y_true_test,y_score_test)
    TNR = 1-FPR
    FNR = 1-TPR
    tn,fp,fn,tp = confusion_matrix(y_true_test,y_abs_score_test).ravel()
    AUC = auc(FPR, TPR)
    plt.figure(1)
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig('ROC.png')
    
    plt.figure(2)
    bins = np.linspace(0, 1, 50)
    counts_sig,bin_edges_sig=np.histogram(sig_pred_train,50)
    bin_centres_sig = (bin_edges_sig[:-1]+bin_edges_sig[1:])/2
    menStd_sig = np.sqrt(counts_sig)
    width = 0.01
    plt.errorbar(bin_centres_sig,counts_sig,yerr=menStd_sig,fmt='o',label='Signal(Training Set)')
    counts_bg,bin_edges_bg=np.histogram(bg_pred_train,50)
    bin_centres_bg = (bin_edges_bg[:-1]+bin_edges_bg[1:])/2
    menStd_bg = np.sqrt(counts_bg)
    plt.errorbar(bin_centres_bg,counts_bg,yerr=menStd_bg,fmt='o',label = 'Background(Training Set)')
    plt.hist(sig_pred_test,label='Signal(Testing set)',bins=bins,histtype='step',color='blue')
    plt.hist(bg_pred_test,label='Background(Testing set)',bins=bins,histtype='step',color='red')
    #plt.hist(sig_pred_train,label='Signal Train',bins=bins,histtype='step',density=True)
    #plt.hist(bg_pred_train,label='Background Train',bins=bins,histtype='step',density=True)
    plt.legend(loc="best")
    plt.xlabel('Output',fontsize=18)
    plt.ylabel('Entries',fontsize=18)

    plt.savefig('histo.png')
    
    plt.figure(3)
    plt.plot(TPR,TNR, label = 'ROC curve')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')

    plt.savefig('signal_background.png')
    
    #plt.show()
    
    #saver.save(sess, '/home/dell/my_tf_model')
    coord.request_stop()
    coord.join(threads)
   
