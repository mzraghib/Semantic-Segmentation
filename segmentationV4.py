import tensorflow as tf
import os
import helperV3
from PIL import Image, ImageEnhance
import random
import numpy as np

# set variables
num_classes = 2
image_shape = (160, 576)
epochs = 180
batch_size = 5


#define paths
data_dir =  os.getcwd()
runs_dir = os.path.join(data_dir, 'runs')
vgg_path = os.path.join(data_dir, 'vgg')
graph_dir= data_dir + '\\graphs'  # change '\\' to '/' for linux
data_folder = os.path.join(data_dir, 'data_road\\training') # change '\\' to '/' for linux


# TF placeholders
correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
keep_prob = tf.placeholder(tf.float32)



def load_vgg(sess, vgg_path):
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    # 1x1 convolution of vgg layer 7
    layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # upsample
    layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer3a_out = tf.add(layer3a_in1, layer3a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label  = tf.reshape(correct_label, (-1,num_classes))
    # define loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    # initialize all the variables in one go
    #this adds an init_op to the graph
    sess.run(tf.global_variables_initializer())   
    
    
    LEARNING_RATE = 1e-3
    DROPOUT = 0.5
    
    print("Training...")
    print()
    for i in range(epochs):
        
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):

            if i > 10 :  # leave first few epochs unchanged
                #preprocessing
                for j in range(0,2):
                    r = random.uniform(0., 1.)
                    if(r > 0.3):
                        if(r > 0.6): # change contrast
                            print(np.shape(image[j]))
                            print('debug')
    
                            img = image[j].copy()
                            print(np.shape(img))
                            img2 = Image.fromarray(img)
                            scale_value= random.uniform(0.7, 1.3)
                            contrast = ImageEnhance.Contrast(img2)
                            contrast_applied=contrast.enhance(scale_value)
                            image[j] = np.array(contrast_applied)
                        # flip about vertical axis
                        else:
                            im = np.fliplr( image[j] )
                            image[j] = im
                            
                            im2 =  np.fliplr( label[j] )
                            label[j] = im2            
            
            
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: DROPOUT, learning_rate: LEARNING_RATE})
            print("Loss", loss)
        print()
        
        
    

def run():    
    # Create function to get batches
    get_batches_fn = helperV3.gen_batch_function(data_folder, image_shape)
    

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # Returns the three layers, keep probability and input layer from the vgg architecture
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # Returns the output logits, training operation and cost operation to be used
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        
        # Train the neural network
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # Save inference data 
        helperV3.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helperV3.save_inference_samples2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()