
# coding: utf-8

# In[ ]:


import tensorflow as tf
from unet_2d import *
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel as nib
from nilearn import image
import time
import warnings
warnings.filterwarnings("ignore")

ckpt_dir = "checkpoint_directory"


# In[ ]:


#Input Data

im_dir = "image_directory"
msk_dir = "mask_directory"

im_list = os.listdir(im_dir)
msk_list = os.listdir(msk_dir)

im_list.sort()
msk_list.sort()

im_list = np.array([im_dir + x for x in im_list]) 
msk_list = np.array([msk_dir + x for x in msk_list])

# Train & Validation split

idx = np.arange(im_list.size)
np.random.shuffle(idx)

val_size = 50

train_idx = idx[:-val_size]
val_idx = idx[-val_size:]

train_im_list = im_list[train_idx]
train_msk_list = msk_list[train_idx]
val_im_list = im_list[val_idx]
val_msk_list = msk_list[val_idx]


# In[ ]:


def preprocess_image(img):
    path = "path" # Path to reference image
    ref_img = nib.load(path)
    affine = ref_img.get_affine()
    shape = ref_img.get_shape()
    im_res = image.resample_img(img, target_affine=affine, target_shape=shape, interpolation='continuous')
    return im_res


# In[ ]:


def load_train_batch_from_nifti_image(batch_size):
    idx = np.random.randint(train_im_list.size)
    im_path = train_im_list[idx]
    msk_path = train_msk_list[idx]
    im = nib.load(im_path)
    im_prep = preprocess_image(im)
    im_arr = im_prep.get_data()
    msk = nib.load(msk_path)
    msk_prep = preprocess_image(msk)
    msk_arr = msk_prep.get_data()
    # im and msk batches
    idxs = np.random.randint(im_arr.shape[0], size=batch_size)
    im_batch = im_arr[idxs, :,:]
    msk_batch = msk_arr[idxs, :,:]
    return im_batch, msk_batch


# In[ ]:


def fill_feed_dict(input_pl, mask_pl, im_batch, msk_batch, slice_counter):
    frame_im = im_batch[slice_counter,:,:]
    frame_msk = msk_batch[slice_counter,:,:]
    r,c = frame_im.shape
    frame_im = np.reshape(frame_im, (1,r,c,1))
    frame_msk = np.reshape(frame_msk, (1,r,c,1))
    feed_dict = {input_pl: frame_im, mask_pl: frame_msk}
    return feed_dict


# In[ ]:


def get_mask_from_prediction(pred):
    pred = np.squeeze(pred, axis=0)
    msk_pred = np.argmax(pred, axis=2)
    return msk_pred


# In[ ]:


def plot_scores():
    channels = 2
    layer = layer_dict['score']
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(layer[0,:,:,0])
    plt.subplot(122)
    plt.imshow(layer[0,:,:,1])
    plt.show()


# In[ ]:


restore = False
learning_rate = 1e-5
max_steps = 50

#Create a U-Net instance
with tf.Graph().as_default():
    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=(10, 1024, 1024, 1))
    mask_pl = tf.placeholder(tf.float32, shape=(10, 1024, 1024, 1))
    # Ops
    prediction_op = inference(input_pl)
    loss_op = loss(mask_pl, prediction_op) 
    train_op = training(loss_op, learning_rate)
    # Create session
    sess = tf.InteractiveSession()
    # Run the Op to initialize the variables.
    saver = tf.train.Saver()
    if restore == False:
        init = tf.global_variables_initializer()
        sess.run(init)
    elif restore == True:
        checkpoint_file = '{}unet_2d_model_1000.ckpt'.format(ckpt_dir) 
        print('Loading variables from checkpoint_file')
        saver.restore(sess, checkpoint_file)
    batch_size = 50
    # Train loop
    print('Training:')
    for step in range(max_steps):
        # Load image (and hence a batch of 50 images)
        im_batch, msk_batch = load_train_batch_from_nifti_image(batch_size=50)
        # Feed images from this batch
        for i in range(batch_size):
            start_time = time.time()
            input_dict = fill_feed_dict(input_pl, mask_pl, im_batch, msk_batch, i)
            activations, pred, loss = sess.run([train_op, prediction_op, loss_op], feed_dict=input_dict)
            duration = time.time() - start_time
            
            # Print after each epoch
            if i == 0:
                print('Step %s: loss = %s' % (step, loss))

            # Show images 
            '''
            if step % 5 == 0 and i == 0:
                #plot_scores()
                msk_pred = get_mask_from_prediction(pred)
                plt.figure(figsize=(10,10))
                plt.subplot(141)
                plt.imshow(input_dict[input_pl][0,:,:,0])
                plt.subplot(142)
                plt.imshow(input_dict[mask_pl][0,:,:,0])
                plt.subplot(143)
                plt.imshow(msk_pred)
                #plt.subplot(144)
                #plt.imshow(loss_im[0,:,:])
                plt.show()
            '''
            
            # Store parameter values in checkpoint file
            if step+1 == (max_steps/2) or step+1 == max_steps:
                if i == 0:
                    checkpoint_file = '{}/unet_2d_model_{}.ckpt'.format(ckpt_dir, str(step+1))
            

