
# coding: utf-8

# In[ ]:


## The network construction

import tensorflow as tf
import numpy as np
conv_filter_size = 3
BATCH_SIZE = 1


def conv_block(x, in_channels, channels_1, channels_2, keep_prob):
    stddev = tf.cast(tf.sqrt(tf.divide(2,((conv_filter_size**2 * in_channels)))), tf.float32)
    with tf.variable_scope('conv1'):
        shape = [conv_filter_size, conv_filter_size, in_channels, channels_1]
        w = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name='weights'), validate_shape=False)
        b = tf.Variable(tf.zeros([channels_1]), name = 'bias')
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', name='conv') + b)
        with tf.variable_scope('conv2'):
            shape = [conv_filter_size, conv_filter_size, channels_1, channels_2]
            w = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name='weights'))
            b = tf.Variable(tf.zeros([channels_2]), name = 'bias')
            conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, w, strides=[1, 1, 1, 1], padding='VALID', name='conv') + b)
            conv_2_dropped = tf.nn.dropout(conv_2, keep_prob)
            return conv_2_dropped


def upconv_block(x, x_skip, in_channels, skip_channels, channels_1, channels_2):
    output_shape = [tf.shape(x)[0], 2*tf.shape(x)[1], 2*tf.shape(x)[2], in_channels]
    stddev = tf.cast(tf.sqrt(tf.divide(2,((conv_filter_size**2 * in_channels)))), tf.float32)
    with tf.variable_scope('deconv'):
        shape = [2, 2, in_channels, in_channels]
        w = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name='weights'), validate_shape=False)
        b = tf.Variable(tf.zeros([in_channels]), name='bias', validate_shape=False)
        upconv = tf.nn.relu(tf.nn.conv2d_transpose(x, w, output_shape, strides=[1,2,2,1], padding='VALID', name='upconv') + b)
    with tf.variable_scope('crop_concat'):
        # crop and concatenate x_skip to x 
        shape_diff = tf.shape(x_skip)[1:-1] - tf.shape(upconv)[1:-1]
        offsets = shape_diff/2
        offsets = tf.cast(offsets, tf.int32)
        target_h = tf.shape(upconv)[1]
        target_w = tf.shape(upconv)[2]
        x_skip_cropped = x_skip[:, offsets[0]:offsets[0]+target_h, offsets[1]:offsets[1]+target_w, :]
        x_concatenated = tf.concat([upconv,x_skip_cropped], 3, name='concat')
    with tf.variable_scope('conv_block'):
        return conv_block(x_concatenated, in_channels+skip_channels, channels_1, channels_2, keep_prob=1)


def inference(image_batch):
    image_batch_pad = tf.pad(image_batch, [[0,0],[100,100],[100,100],[0,0]], mode='CONSTANT', name=None)
    with tf.variable_scope('contracting'):
        with tf.variable_scope('step1'):
            conv_block_1 = conv_block(image_batch_pad, BATCH_SIZE, 32, 64, keep_prob=1)
            conv_pool_1 = tf.nn.max_pool(conv_block_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope('step2'):
            conv_block_2 = conv_block(conv_pool_1, 64, 64, 128, keep_prob=1)
            conv_pool_2 = tf.nn.max_pool(conv_block_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope('step3'):
            conv_block_3 = conv_block(conv_pool_2, 128, 128, 256, keep_prob=1)
            conv_pool_3 = tf.nn.max_pool(conv_block_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope('step4'):
            conv_block_4 = conv_block(conv_pool_3, 256, 256, 512, keep_prob=0.5)
            conv_pool_4 = tf.nn.max_pool(conv_block_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        with tf.variable_scope('step5'):
            conv_block_5 = conv_block(conv_pool_4, 512, 1024, 1024, keep_prob=0.5)
    with tf.variable_scope('expanding'):
        with tf.variable_scope('step4'):
            upconv_block_1 = upconv_block(conv_block_5, conv_block_4, 1024, 512, 512, 512)
        with tf.variable_scope('step3'):
            upconv_block_2 = upconv_block(upconv_block_1, conv_block_3, 512, 256, 256, 256)
        with tf.variable_scope('step2'):
            upconv_block_3 = upconv_block(upconv_block_2, conv_block_2, 256, 128, 128, 128)
        with tf.variable_scope('step1'):
            upconv_block_4 = upconv_block(upconv_block_3, conv_block_1,128, 64, 64, 64)
    with tf.variable_scope('scoring'):
        with tf.variable_scope('conv'):
            in_channels = 64
            out_channels = 2 # Binary class segmentation
            shape = [1, 1, in_channels, out_channels]
            stddev = tf.sqrt(2.0 / (conv_filter_size**2 * in_channels))
            w = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name='weights_scoring'), validate_shape=False)
            b = tf.Variable(tf.zeros([out_channels]), name='bias')
            score = tf.nn.conv2d(upconv_block_4, w, strides=[1, 1, 1, 1], padding='VALID', name='conv') + b
        with tf.variable_scope('crop_2_data'): # We want predicted_segmentation.shape = input_image.shape
            #import pdb; pdb.set_trace()
            shape_diff = tf.shape(score)[1:-1] - tf.shape(image_batch)[1:-1]
            offsets = shape_diff/2
            offsets = tf.cast(offsets, tf.int32)
            target_h = tf.shape(image_batch)[1]
            target_w = tf.shape(image_batch)[2]
            prediction = score[:, offsets[0]:offsets[0]+target_h, offsets[1]:offsets[1]+target_w, :]
    return prediction

def loss(mask, prediction):
    mask_0 = 1-mask
    mask_1 = mask
    mask =  tf.cast(tf.concat([mask_0, mask_1], axis = 3), tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=mask))
    return loss

def loss_sum(loss):
    loss_sum = tf.summary.scalar('loss', loss)
    return loss_sum
    
def accuracy(prediction, mask):
    correct_prediction = tf.equal(prediction, mask)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def acc_sum(accuracy):
    acc_sum = tf.summary.scalar('accuracy', accuracy)
    return acc_sum 

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op


