#OAR Diff-DVH prediction
#CNN with 3 CL, 2 FL and average pooling
#3 input channels corresponding to PTV, Rectum, and Bladder

#libraries
import numpy as np
import tensorflow as tf
import scipy.io as sio
import math
from matplotlib import pyplot as plt
import pickle

#data
file_Name = '/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/data/data_slice_pickle'
fileObject = open(file_Name,'r')  
data = pickle.load(fileObject)

m=100
n=100
p=11
num=2835
output_size = 10
mesh_tnsr = np.zeros((num,m,n,p))  #np.zeros((num,50,50,33))
diff_dvh_tnsr = np.zeros((num,output_size))

#ind_shuff = np.arange(num)
#np.random.shuffle(ind_shuff)
#print ind_shuff[0:5]
ii=0
cc=np.arange(num)
np.random.shuffle(cc)
for i in cc:

     mesh_tnsr[ii,:,:,0:11] = data[i]['contour_ptv'] + data[i]['contour_rectum'] + data[i]['contour_bladder']
     #mesh_tnsr[ii,:,:,11:22] = data[i]['contour_rectum']
     #mesh_tnsr[ii,:,:,22:33] = data[i]['contour_bladder']
     dd_rec=data[ii]['diff_dvh_rectum']
     #print np.size(dd_rec)
     dd_blad=data[ii]['diff_dvh_bladder']
     diff_dvh_tnsr[ii,:] = (dd_rec + 1e-3) / (1 + output_size*1e-3) 
     #diff_dvh_tnsr[i,:,1] = dd_blad #[0:5:1]
     ii = ii+1

############################training data########################
mesh_tnsr_train = mesh_tnsr[1:2500,:,:,:] 
diff_dvh_tnsr_train = diff_dvh_tnsr[1:2500]
############################test data############################
mesh_tnsr_test = mesh_tnsr[2500:2835,:,:,:] #335 test cases
diff_dvh_tnsr_test = diff_dvh_tnsr[2500:2835,:]
##################################################################
        

#print np.shape(mesh_tnsr)
#print np.sum(diff_dvh_tnsr)
#print np.sum(diff_dvh_tnsr,0)

#plt.imshow(diff_dvh_tnsr)
#plt.show()
#plt.imshow(mesh_tnsr[1,:,:,4])
#plt.show()
#wait=input('press enter to continue')


# parameters
#learning_rate = 1e-3
batch_size = 100
display_step = 1

drop_out = 1 #0.1
k=2 #2	
num_features = 30
num_filters_layer_1 = 20
num_filters_layer_2 = 20
num_filters_layer_3 = 20

lam=0.01

mm=(m-5+1)/k
nn=(n-5+1)/k

mmm=(mm-4+1)/k
nnn=(nn-4+1)/k

mmmm=(mmm-3+1)/k
nnnn=(nnn-3+1)/k

# tf graph input
x = tf.placeholder(tf.float32,[None,m,n,p])
diff_dvh = tf.placeholder(tf.float32,[None,output_size])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

# create model
def conv_net(_x, _weights, _biases, _dropout):
 
     #_x = tf.reshape(_x, shape=[-1,m,n,p])
     
     # Encoding
     # Convolution Layer 1
     conv1 = tf.nn.conv2d(_x, _weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
     conv1 = tf.add(conv1, _biases['bc1'])
     conv1 = tf.nn.relu(conv1)   #[-1,m-5+1,n-5+1,num_filters_layer_1]

     #average pooling
     conv1_pool = tf.nn.avg_pool(conv1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')  #[-1,(m-5+1)/k,(n-5+1)/k,num_filters_layer_1]

     # Convolution Layer 2
     conv2 = tf.nn.conv2d(conv1_pool, _weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
     conv2 = tf.add(conv2, _biases['bc2'])
     conv2 = tf.nn.relu(conv2)   #[-1,mm-4+1,nn-4+1,num_filters_layer_2]

     #average pooling
     xx = tf.nn.avg_pool(conv2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')  #[-1,(mm-4+1)/k,(nn-4+1)/k,num_filters_layer_2]
     conv2d_pool = xx; #tf.reshape(xx, shape=[-1,mmm,nnn,num_filters_layer_2])

     # Convolution Layer 3
     conv3 = tf.nn.conv2d(conv2d_pool, _weights['wc3'], strides=[1, 1, 1, 1], padding='VALID')
     conv3 = tf.add(conv3, _biases['bc3'])
     conv3 = tf.nn.relu(conv3)   #[-1,mmm-3+1,nnn-3+1,num_filters_layer_3]

     #average pooling
     conv3_pool = tf.nn.avg_pool(conv3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')  #[-1,(mmm-3+1)/k,(nnn-3+1)/k,num_filters_layer_3]

     #fully connected layer
     dense1 = tf.reshape(conv3_pool, [-1, mmmm*nnnn*num_filters_layer_3]) 
     dense1 = tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])
     dense1 = tf.nn.relu(dense1)  #[-1,num_features]

     # Apply Dropout
     conv3 = tf.nn.dropout(dense1, _dropout)
     dense1 = dense1 / _dropout

     # output
     # fully connected layer
     out = tf.matmul(dense1, _weights['wd2'])  #[-1,output_size]
     out = tf.add(out, _biases['bd2'])
     #out = tf.nn.softmax(out)
     return {'output':out, 'features':dense1, 'output1':conv1, 'output2':conv2, 'output3':conv3}


#define variables
weights = {'wc1': tf.Variable(0.1*tf.random_normal([5, 5, p, num_filters_layer_1])), 
           'wc2': tf.Variable(0.1*tf.random_normal([4, 4, num_filters_layer_1, num_filters_layer_2])), 
           'wc3': tf.Variable(0.1*tf.random_normal([3, 3, num_filters_layer_2, num_filters_layer_3])), 
           'wd1': tf.Variable(0.1*tf.random_normal([mmmm*nnnn*num_filters_layer_3, num_features])), # fully connected
           'wd2': tf.Variable(0.1*tf.random_normal([num_features, output_size])), # fully connected
}

biases = {'bc1': tf.Variable(0.001*tf.random_normal([num_filters_layer_1])),
          'bc2': tf.Variable(0.001*tf.random_normal([num_filters_layer_2])),
          'bc3': tf.Variable(0.001*tf.random_normal([num_filters_layer_3])),
          'bd1': tf.Variable(0.001*tf.random_normal([num_features])),
          'bd2': tf.Variable(0.001*tf.random_normal([output_size])),
}


#summary ops to collect the data
#wc1_hist = tf.histogram_summary("weights",weights['wc1'])
#bc1_hist = tf.histogram_summary("biases",biases['bc1'])

#wc2_hist = tf.histogram_summary("weights",weights['wc2'])
#bc2_hist = tf.histogram_summary("biases",biases['bc2'])

#wc2_hist = tf.histogram_summary("weights",weights['wc3'])
#bc2_hist = tf.histogram_summary("biases",biases['bc3'])


#output
out = conv_net(x, weights, biases, keep_prob)
hat_diff_dvh = out['output'] + 1e-5

#features
feature_vec = out['features']
feature_hist = tf.histogram_summary("features",feature_vec)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hat_diff_dvh,diff_dvh)) #tf.reduce_mean(tf.square(hat_diff_dvh-diff_dvh)) 

reg = tf.reduce_mean(tf.square(weights['wc1'])) + tf.reduce_mean(tf.square(weights['wc2'])) + tf.reduce_mean(tf.square(weights['wc3'])) + tf.reduce_mean(tf.square(weights['wd1'])) + tf.reduce_mean(tf.square(weights['wd2']))

log_diff_dvh = tf.log(diff_dvh) 
cost_norm = -tf.reduce_mean(tf.reduce_sum(tf.mul(diff_dvh,log_diff_dvh), reduction_indices=1)) + 1e-5

#diff_dvh_norm = tf.reduce_mean(tf.square(diff_dvh))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost + lam*reg)

#summary of cost
cost_summary = tf.scalar_summary("cross_entropy_cost", tf.log(cost/cost_norm-1))  #(cost/diff_dvh_norm)**(0.5)


#merge all summaries
merged = tf.merge_all_summaries()


# Initializing the variables
init = tf.initialize_all_variables()


#launch the graph
sess = tf.Session()
sess.run(init)

writer = tf.train.SummaryWriter("/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/output/logs_2d_copy", sess.graph.as_graph_def(add_shapes=True))  

batch_x_test = mesh_tnsr_test
batch_diff_dvh_test = diff_dvh_tnsr_test

step_size = 1e-2
for j in range(1000):
     
     step_size = step_size*0.999 #/(j+1)  #0.001/(j/10+1)

     #test phase
     summary_str = sess.run(merged, feed_dict={x:batch_x_test, diff_dvh:batch_diff_dvh_test, keep_prob:1})
     writer.add_summary(summary_str,j)

     s=0
     count = 25
     dd=np.arange(count)
     #np.random.shuffle(dd)
     for i in dd:	
         
         batch_x = mesh_tnsr_train[i*batch_size:(i+1)*batch_size,:,:,:]
         batch_diff_dvh = diff_dvh_tnsr_train[i*batch_size:(i+1)*batch_size,:] 
         sess.run(optimizer, feed_dict={x:batch_x, diff_dvh:batch_diff_dvh, keep_prob:drop_out, learning_rate:step_size})
         loss_train = sess.run((cost/cost_norm)-1, feed_dict={x:batch_x, diff_dvh:batch_diff_dvh, keep_prob:1}) #(cost/diff_dvh_norm)**(0.5)
         s = s + loss_train

         #if i % display_step == 0: 
                 #print "Iter" + str(i) + ", loss train= " + "{:.6f}".format(loss_train) 
 
     #test phase
     diff_dvh_pred = sess.run(tf.nn.softmax(hat_diff_dvh), feed_dict={x:batch_x_test, diff_dvh:batch_diff_dvh_test, keep_prob:1})  
     test_err = sess.run((cost/cost_norm)-1, feed_dict={x:batch_x_test, diff_dvh:batch_diff_dvh_test, keep_prob:1}) 

     print "************************************************************"
     print "epoch=" + str(j) + ", train error= " + "{:.6f}".format(s/count) + ", test error= " + "{:.6f}".format(test_err) 
     #print batch_diff_dvh_test[1,:]
     #print diff_dvh_pred[1,:] 
     #print "------------------------------------------------------------"
     #print batch_diff_dvh_test[2,:]
     #print diff_dvh_pred[2,:] 
     #print "------------------------------------------------------------"
     #print batch_diff_dvh_test[3,:]
     #print diff_dvh_pred[3,:] 
     #print "************************************************************"

#features
feature_map=sess.run(feature_vec, feed_dict={x:batch_x_test, diff_dvh:batch_diff_dvh_test, keep_prob:1})

#save the variables
data_output = {'diff_dvh_pred':diff_dvh_pred, 'mesh_batch':batch_x_test, 'diff_dvh_true':batch_diff_dvh_test, 'feature_map':feature_map}
sio.savemat('/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/output/output_2d_copy.mat',data_output)

print "Finished!"

writer.close()

#tensorboard --logdir=/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/output/logs_2d_copy
#./bazel-bin/tensorflow/tensorboard/tensorboard --logdir=/home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/output/logs_2d_copy
#rm -r /home/morteza/Dropbox/python_codes/prostate-data-3D-CNN-may2016/output/logs_2d_copy

