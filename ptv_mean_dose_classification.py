#PTV mean dose prediction
#input-CCN-FC(features)-output(scalar)
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
n_classes = 5
ind_ptv = []
for i in range(num_slice):
    if data[i]['dose_mean'] != 0: 
         num_slice_ptv = num_slice_ptv+1
         ind_ptv.append(i)

contour_array = np.zeros((num_slice_ptv,100,100))
dose_label = np.zeros((num_slice_ptv,n_classes))
for i in range(num_slice_ptv):
    contour_array[i,:,:] = data[ind_ptv[i]]['contour']
    #dose_array[i,:] = data[ind_ptv[i]]['ptv_vol']*data[ind_ptv[i]]['dose_mean'] / 1e4
    ind_class = data[ind_ptv[i]]['dose_label']  #binary array
    dose_label[i,ind_class-1] = 1


# parameters
m = 100
n = 100
k = 1 #max pooling
learning_rate = 1e-3
batch_size = 25
display_step = 1
#drop_out = 0.5


filter_size_m = 100
filter_size_n = 100
num_filters_layer_1 = 10   #10
num_filters_layer_2 = 10   #10
num_features = 5


# tf graph input
x = tf.placeholder(tf.float32,[None,m,n])
label = tf.placeholder(tf.float32,[None,n_classes])
#keep_prob = tf.placeholder(tf.float32)


# create model
def conv_net(_x, _weights, _biases):
 
     _x = tf.reshape(_x, shape=[-1,m,n,1])
     
     # Encoding
     # Convolution Layer 1
     conv1 = tf.nn.conv2d(_x, _weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
     conv1 = tf.add(conv1, _biases['bc1'])
     conv1 = tf.nn.relu(conv1)   #[-1,m-filter_size_m+1,n-filter_size_n+1,num_filters_layer_1]

     #max-pooling
     conv1_pool = tf.nn.avg_pool(conv1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')  #[-1,(m-filter_size_m+1)/k,(n-filter_size_n+1)/k,num_filters_layer_1]

     # Convolution Layer 2
     conv2 = conv1_pool #tf.nn.conv2d(conv1, _weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
     #conv2 = tf.add(conv2, _biases['bc2'])
     #conv2 = tf.nn.relu(conv2)   #[-1,m,n,num_filters_layer_2]???!!!

     #fully connected layer
     dense1 = tf.reshape(conv2, [-1, ((m-filter_size_m+1)/k)*((n-filter_size_n+1)/k)*num_filters_layer_2]) 
     dense1 = tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])
     dense1 = tf.nn.relu(dense1)  #[-1,num_features]

     # output
     # fully connected layer
     out = tf.matmul(dense1, _weights['wd2'])  #[-1,1]
     out = tf.add(out, _biases['bd2'])
     return {'output':out, 'features':dense1, 'feature_map':conv2}


#define variables
weights = {'wc1': tf.Variable(0.01*tf.random_normal([filter_size_m, filter_size_n, 1, num_filters_layer_1])), # 5x5 conv, 1 input, 32 outputs
           'wc2': tf.Variable(0.01*tf.random_normal([3, 3, num_filters_layer_1, num_filters_layer_2])), # 5x5 conv, 1 input, 32 outputs
           'wd1': tf.Variable(0.01*tf.random_normal([((m-filter_size_m+1)/k)*((n-filter_size_n+1)/k)*num_filters_layer_2, num_features])), # fully connected
           'wd2': tf.Variable(0.01*tf.random_normal([num_features, n_classes])), # fully connected
}

biases = {'bc1': tf.Variable(0.000001*tf.random_normal([num_filters_layer_1])),
          'bc2': tf.Variable(0.000001*tf.random_normal([num_filters_layer_2])),
          'bd1': tf.Variable(0.000001*tf.random_normal([num_features])),
          'bd2': tf.Variable(0.000001*tf.random_normal([n_classes])),
}


#features & output
out = conv_net(x, weights, biases)
hat_label_dose = out['output']
feature_vec = out['features']
xx=out['feature_map'][0,:,:,0]
feature_map = tf.reshape(xx,shape=[1,(m-filter_size_m+1)/k,(n-filter_size_n+1)/k,1])

lambda_c = 2.5
lambda_f = 1
lambda_o = 1
#cross entropy cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hat_label_dose,label)) + lambda_c*tf.reduce_mean(tf.square(weights['wc1'])) + lambda_f*tf.reduce_mean(tf.square(weights['wd1'])) + lambda_o*tf.reduce_mean(tf.square(weights['wd2']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#summary ops to collect the data
cost_summary = tf.scalar_summary("cross_entropy_loss",cost)
wc1_hist = tf.histogram_summary("weights",weights['wc1'])
bc1_hist = tf.histogram_summary("biases",biases['bc1'])
feature_hist = tf.histogram_summary("features",feature_vec)
feature_map_summary = tf.image_summary("feature_map",feature_map)
#original_image = tf.image_summary("image_actual",tf.reshape(x,shape=[p,m,n,1]))

merged = tf.merge_all_summaries()


#evaluate model
correct_pred = tf.equal(tf.argmax(hat_label_dose,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()


#launch the graph
sess = tf.Session()
sess.run(init)

writer = tf.train.SummaryWriter("/tmp/slice_contour_mean_classification_logs", sess.graph.as_graph_def(add_shapes=True))  

batch_x_test = contour_array[num_slice_ptv-100:num_slice_ptv,:,:]
batch_d_test = dose_label[num_slice_ptv-100:num_slice_ptv,:]

for j in range(2000):

     s=0
     count = 59
     for i in range(count):	
         
         batch_x = contour_array[i*batch_size:(i+1)*batch_size-1,:,:]
         batch_d = dose_label[i*batch_size:(i+1)*batch_size-1,:]
         sess.run(optimizer, feed_dict={x:batch_x, label:batch_d})
         loss_train = sess.run(cost, feed_dict={x:batch_x, label:batch_d}) 
         train_err = loss_train
         s = s + train_err

         if i % (batch_size-1) == 0:
                 #print "Iter" + str(i) + ", mean dose= " + "{:.6f}".format((dose)**(0.5)) + ", train_error= " + "{:.6f}".format(train_err) 
                 summary_str = sess.run(merged, feed_dict={x:batch_x, label:batch_d})
                 writer.add_summary(summary_str,j)
 
     label_dose_pred = sess.run(tf.nn.softmax(hat_label_dose), feed_dict={x:batch_x_test})
     test_acc = sess.run(accuracy, feed_dict={x:batch_x_test, label:batch_d_test}) 
     test_cost = sess.run(cost, feed_dict={x:batch_x_test, label:batch_d_test}) 
     print "***********************************************************"
     print "epoch=" + str(j) + ", avg_cross_entropy= " + "{:.3f}".format(s/count)  + ", test_acc= " + "{:.3f}".format(test_acc) + ", test_cost= " + "{:.3f}".format(test_cost) 
     print batch_d_test[0:4,:]
     print label_dose_pred[0:4,:]
     print "***********************************************************"

print "Finished!"

writer.close()

#tensorboard --logdir=/tmp/slice_contour_mean_classification_logs
#rm -r /tmp/slice_contour_mean_classification_logs

