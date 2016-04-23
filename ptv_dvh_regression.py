#PTV DVH prediction
#input-CCN-FC(features)-output(vector of size 10)
#CCN for feature extraction from contours


#libraries
import numpy as np
import tensorflow as tf
import scipy.io as sio
import pickle
import math
from matplotlib import pyplot as plt


#data
file_Name = '/home/morteza/Documents/python_scripts/data_slice_pickle'
fileObject = open(file_Name,'r')  
data = pickle.load(fileObject)

num_slice = 2835
num_slice_ptv=0
ind_ptv = []
for i in range(num_slice):
    if data[i]['dose_mean'] != 0: 
         num_slice_ptv = num_slice_ptv+1
         ind_ptv.append(i)

contour_array = np.zeros((num_slice_ptv,100,100))
dvh_array = np.zeros((num_slice_ptv,10))

for i in range(num_slice_ptv):
    contour_array[i,:,:] = data[ind_ptv[i]]['contour']
    dvh_array[i,:] = data[ind_ptv[i]]['ptv_vol']*data[ind_ptv[i]]['dvh_ptv'] / 1e2 #/ 1e4


# parameters
m = 100
n = 100
learning_rate = 1e-3
batch_size = 50
display_step = 1
#drop_out = 0.5
num_features = 25
num_filters_layer_1 = 10
num_filters_layer_2 = 10


# tf graph input
x = tf.placeholder(tf.float32,[None,m,n])
dvh = tf.placeholder(tf.float32,[None,10])
#keep_prob = tf.placeholder(tf.float32)


# create model
def conv_net(_x, _weights, _biases):
 
     _x = tf.reshape(_x, shape=[-1,m,n,1])
     
     # Encoding
     # Convolution Layer 1
     conv1 = tf.nn.conv2d(_x, _weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
     conv1 = tf.add(conv1, _biases['bc1'])
     conv1 = tf.nn.relu(conv1)   #[-1,m,n,num_filters_layer_1]

     # Convolution Layer 2
     conv2 = conv1 #tf.nn.conv2d(conv1, _weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
     #conv2 = tf.add(conv2, _biases['bc2'])
     #conv2 = tf.nn.relu(conv2)   #[-1,m,n,num_filters_layer_2]

     #fully connected layer
     dense1 = tf.reshape(conv2, [-1, m*n*num_filters_layer_2]) 
     dense1 = tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])
     dense1 = tf.nn.relu(dense1)  #[-1,num_features]

     # output
     # fully connected layer
     out = tf.matmul(dense1, _weights['wd2'])  #[-1,10]
     out = tf.add(out, _biases['bd2'])
     out = tf.nn.relu(out)
     return {'output':out, 'features':dense1}


#define variables
weights = {'wc1': tf.Variable(0.1*tf.random_normal([5, 5, 1, num_filters_layer_1])), # 5x5 conv, 1 input, 32 outputs
           'wc2': tf.Variable(0.1*tf.random_normal([5, 5, num_filters_layer_1, num_filters_layer_2])), # 5x5 conv, 1 input, 32 outputs
           'wd1': tf.Variable(0.1*tf.random_normal([m*n*num_filters_layer_2, num_features])), # fully connected
           'wd2': tf.Variable(0.1*tf.random_normal([num_features, 10])), # fully connected
}

biases = {'bc1': tf.Variable(0.1*tf.random_normal([num_filters_layer_1])),
          'bc2': tf.Variable(0.1*tf.random_normal([num_filters_layer_2])),
          'bd1': tf.Variable(0.1*tf.random_normal([num_features])),
          'bd2': tf.Variable(0.1*tf.random_normal([10])),
}


#summary ops to collect the data
wc1_hist = tf.histogram_summary("weights",weights['wc1'])
bc1_hist = tf.histogram_summary("biases",biases['bc1'])


#reconstructed image
out = conv_net(x, weights, biases)
hat_dvh = out['output']


#features
feature_vec = out['features']
feature_hist = tf.histogram_summary("features",feature_vec)


#summary of the images
#recon_image_summary = tf.image_summary("image_recon",recon_image_4d)
#original_image = tf.image_summary("image_actual",tf.reshape(x,shape=[p,m,n,1]))


#regualrized ls cost
xx=np.zeros((10,10))
for i in range(9):
    xx[i,i] = 1
    xx[i,i+1] = -1

xx=tf.to_float(xx)

cost = tf.reduce_mean(tf.square(hat_dvh - dvh)) + tf.reduce_sum(tf.reduce_max(tf.matmul(xx,tf.transpose(hat_dvh)), reduction_indices=[0]))
dvh_norm = tf.reduce_mean(tf.square(dvh))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#summary of cost
cost_summary = tf.scalar_summary("mse_normalized",(cost/dvh_norm)**(0.5))


#merge all summaries
merged = tf.merge_all_summaries()


# Initializing the variables
init = tf.initialize_all_variables()


#launch the graph
sess = tf.Session()
sess.run(init)

writer = tf.train.SummaryWriter("/tmp/slice_contour_dvh_logs", sess.graph.as_graph_def(add_shapes=True))  

batch_x_test = contour_array[num_slice_ptv-100:num_slice_ptv,:,:]
batch_dvh_test = dvh_array[num_slice_ptv-100:num_slice_ptv,:]

for j in range(1000):
     
     #test phase
     summary_str = sess.run(merged, feed_dict={x:batch_x_test, dvh:batch_dvh_test})
     writer.add_summary(summary_str,j)

     s=0
     count = 29
     for i in range(count):	
         
         batch_x = contour_array[i*batch_size:(i+1)*batch_size-1,:,:]
         batch_dvh = dvh_array[i*batch_size:(i+1)*batch_size-1,:]
         sess.run(optimizer, feed_dict={x:batch_x, dvh:batch_dvh})
         loss_train = sess.run(cost, feed_dict={x:batch_x, dvh:batch_dvh}) 
         norm_dvh = sess.run(tf.reduce_mean(tf.square(dvh)), feed_dict={dvh:batch_dvh})
         train_err = (loss_train / (norm_dvh+1e-5))**(0.5)
         s = s + train_err

         if i % display_step == 0: # and batch_d != 0:
                 print "Iter" + str(i) + ", avg. train error= " + "{:.6f}".format(train_err) 
 
     #dvh_orig = batch_dvh_test[1,:]
     dvh_pred = sess.run(hat_dvh, feed_dict={x:batch_x_test, dvh:batch_dvh_test})
     test_err = sess.run((cost/dvh_norm)**(0.5), feed_dict={x:batch_x_test, dvh:batch_dvh_test}) 
     print "************************************************************"
     print "epoch=" + str(j) + ", avg. train error= " + "{:.6f}".format(s/count) + ", test error= " + "{:.6f}".format(test_err)  
     print batch_dvh_test[1,:]
     print dvh_pred[1,:] 
     print "************************************************************"


print "Finished!"

writer.close()

#tensorboard --logdir=/tmp/slice_contour_dvh_logs
#rm -r /tmp/slice_contour_dvh_logs

