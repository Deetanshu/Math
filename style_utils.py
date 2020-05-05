# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:22:46 2018

@author: deept
"""

#Cleaned up one.py to make it library like

#Imports:
print("[INFO] Importing dependencies... ")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import IPython.display
import cv2
print("[INFO] Dependencies imported.")
#Enabling Eager Execution:

#tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

# Layers to be extracted:

# Content layer for feature map:
content_layers = ['block5_conv2']

# Style layers:
style_layers = ['block1_conv1',
               'block2_conv1',
               'block3_conv1',
               'block4_conv1',
               'block5_conv1']
num_content_layers = 1
num_style_layers = 5
#Loading image - Two variants:
def load_img(path_to_img, isBW = False):
    max_dim = 512
    img = Image.open(path_to_img)
    if isBW:
        img = RGB_G(img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    print("[INFO] Image loaded.")
    plt.subplot(1, 2, 1)
    imshow(img, 'Content Image')
    return img


#Showing image: 
    
def imshow(img, title=None):
    out = np.squeeze(img, axis=0)
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)
    
    
#Data preprocessing:
def load_and_process_img(path_to_img, isBW = False):
    img = load_img(path_to_img, isBW)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


#Make the image from the internal representation again:
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x=np.squeeze(x,0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of dimension ""[1, height, width, channel] or [height,width, channel]")
    
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
        
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Fetching VGG-19:
def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights = 'imagenet')
    vgg.trainable=False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs+content_outputs
    return models.Model(vgg.input, model_outputs)


# Getting losses from various parts (style+content):
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a,a,transpose_a = True)
    return gram/tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def get_feature_representations(model, content_path, style_path, isBW = False):
    
    #Load image:
    content_image = load_and_process_img(content_path, isBW)
    style_image = load_and_process_img(style_path, isBW)
    
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

#Computing loss and gradients
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0
    
    weight_per_style_layer = 1.0/float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    weight_per_content_layer = 1.0/float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight
    
    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

#Optimization loop

def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight = 1e3,
                       style_weight = 1e-2,
                       isBW = False):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    
    style_features, content_features = get_feature_representations(model, content_path, style_path, isBW)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    init_image = load_and_process_img(content_path, isBW)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    iter_count = 1
    best_loss, best_img = float('inf'), None
    
    loss_weights = (style_weight, content_weight)
    cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
      }
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score,content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()
        
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        
        if i % display_interval == 0:
            start_time = time.time()
            
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))        
            print('Total loss: {:.4e}, ' 
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
      
    return best_img, best_loss, imgs

# Denoising images:
def denoise(PIL_img=None, img=None , filter_size=21, multiFilter = False, f2_size=5, f1_size=50):
    if PIL_img is not None:
        img = conv_p2cv(PIL_img)
    if img is not None and not multiFilter:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10,10,filter_size)
        return img
    elif img is not None and multiFilter:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, f1_size)
        img2 = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, f2_size)
        return img, img2

# Convert PIL to opencv:
def conv_p2cv(PIL_img):
    open_cv_image = np.array(PIL_img) 
    return cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

# Convert to RGB - Grayscale:
def RGB_G(PIL_img):
    image_file = PIL_img.convert('L')
    return image_file.convert('RGB')
    