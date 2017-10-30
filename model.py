#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import time
import os
import urllib.request
from os.path import isfile, join
from tensorflow.python.util import nest



##################################
def _read_file(filename): # read a single sample chrom file, each file contains 1 line which represents chrom of one sample
	fo = open(filename, "r")
	line = list(fo.readline())
	line_int = [ord(x) for x in line]
	fo.close()
	print("read file length: ", len(line_int))
	return line_int

def _get_chrom(dirpath,chr): # read all samples for selected chrom under specific dir (train, dev, test), return a list of all sample seq
	#allsample_dir = dirpath + '/' + chr
	allsample_dir = dirpath
	samplefiles = [join(allsample_dir, f) for f in os.listdir(allsample_dir) if isfile(join(allsample_dir, f))]
	data = [ _read_file(f) for f in samplefiles[0:9]]
	return data

def _prepare_raw_data(trainpath, devpath, testpath, chr, num_class): # read train, dev, test data and coded as one hot
	train_data = tf.one_hot(_get_chrom(trainpath,chr), num_class) # sample number x seq length x class number
	print("train data size: ",train_data.get_shape())
	#train_data = tf.unstack(train_data, axis=1)
	dev_data = tf.one_hot(_get_chrom(devpath,chr), num_class)
	print("dev data size: ",dev_data.get_shape())
	#dev_data = tf.unstack(dev_data, axis=1)
	test_data = tf.one_hot(_get_chrom(testpath,chr), num_class)
	print("test data size: ",test_data.get_shape())
	#test_data = tf.unstack(test_data, axis=1)
	return {'train':train_data, 'dev':dev_data ,'test':dev_data }
	#return train_data, dev_data, test_data
	
def prepare_data(rawdata_path, chr, num_class,batch_size, num_steps, name=None): # /work/yd44/imputation/sample_perl
	trainpath = rawdata_path + '/' + chr
	devpath = rawdata_path + '/dev/' + chr
	testpath = rawdata_path + '/test/' + chr
	train, valid, test = _prepare_raw_data(trainpath, devpath, testpath, chr, num_class)

def generate_batch(data, batch_size, num_steps): # generate batches from input data, discard the data that smaller than batch_size x num_step
	rows, columns, depth = map(lambda i: i.value, data.get_shape())
	num_slice = columns // num_steps
	num_batch = rows // batch_size
	
	total_batch = num_slice * num_batch
	chop_data = data[num_batch*batch_size, num_slice*num_steps,:]
	for i in range(total_batch):
		x = chop_data[i*batch_size:(i+1)*batch_size, i * num_steps:(i + 1) * num_steps, :]
		y = chop_data[i*batch_size:(i+1)*batch_size, i * num_steps+1:(i + 1) * num_steps+1, :]
		yield (x, y)

def generate_epoch(data, num_epoch, batch_size, num_steps):
	for i in range(num_epoch):
		yield generate_batch(data.random_shuffle(), batch_size, num_steps)
               
        
  
"""
Train the network
"""

def train_network(g, rawdata_path, chr = 'chr1', num_epochs=2, num_steps = 20, batch_size = 32, num_class = 4, verbose = True, save=False):
    tf.set_random_seed(2345)
    #rawdata_path = '/work/yd44/imputation/sample_perl'

    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        train, valid, test = prepare_data(rawdata_path, chr, num_class,batch_size, num_steps, name=None)
        
        training_losses = []
        for epoch in enumerate(genarate_epochs(input, num_epoch, batch_size, num_steps)):
        	avg_cost = 0.0
        	steps = 0
        	training_state = None
        	e = batch_size
        	
        	for batch in epoch:
        		for slice in batch:
        			for X, Y in slice:
        				steps += 1
        				feed_dict={g['x']: X, g['y']: Y}
        				
        				if training_state is not None:
        					feed_dict[g['init_state']] = training_state
        					
        				training_loss_, training_state, _ = sess.run([g['total_loss'],
        													g['final_state'],
        													g['train_step']],
        													feed_dict)
        				if verbose:
        					print("Average training loss for Epoch", idx, ":", training_loss/steps)
        					
        				training_losses.append(training_loss/steps)
        
        if isinstance(save, str):
        	g['saver'].save(sess, save)

    return training_losses

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

# def zeros(num_layer, batch_size, state_size):
# 	for i in range(num_layers):
	

def build_lstm_graph(
    state_size = 128,
    num_classes = 4,
    batch_size = 32,
    num_steps = 100,
    num_layers = 2,
    learning_rate = 1e-3,
    build_with_dropout=False):

    reset_graph()

    input_x = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes], name='input_placeholder')
    input_y = tf.placeholder(tf.int32, [batch_size, num_steps, num_classes], name='labels_placeholder')   
    #input_y = tf.cast(input_y, tf.int32)
    
    #input_x = tf.unstack(input_x, axis=1) ##
    
    dropout = tf.constant(0.9)
	   
    cells = []
    for i in range(num_layers):
    	cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    	if build_with_dropout:
    		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout)
    	cells.append(cell)
        	
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    init_state = multi_cell.zero_state(batch_size, tf.float32)
      
#     lstm0_c = tf.placeholder(tf.float32, [batch_size, state_size])
#     lstm0_h = tf.placeholder(tf.float32, [batch_size, state_size])
#     lstm1_c = tf.placeholder(tf.float32, [batch_size, state_size])
#     lstm1_h = tf.placeholder(tf.float32, [batch_size, state_size])
#     
#     zeros = tf.zeros([batch_size, state_size])
#     state1 = tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)
#     state2 = tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)
#     init_state = tuple((state1, state2))
    
    #init_state = tuple((tf.nn.rnn_cell.LSTMStateTuple(lstm0_c, lstm0_h),tf.nn.rnn_cell.LSTMStateTuple(lstm1_c, lstm1_h)))
    #init_state = tuple((tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros),tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)))
    
    #print(init_state)
  
    
    #init_state1 = tf.nn.rnn_cell.LSTMStateTuple(lstm0_c, lstm0_h)
    
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, input_x, initial_state=init_state, swap_memory=True)
    #rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, input_x, initial_state=init_state)
    #rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, input_x, dtype=tf.float32)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(input_y, [-1,num_classes])
    #print(y_reshaped)
    y_label = tf.argmax(y_reshaped, axis=1)

    logits = tf.matmul(rnn_outputs, W) + b
    predictions = tf.nn.softmax(logits) 
    #print(predictions)

    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = input_x,
        y = input_y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )



######################################
state_size = 128
num_classes = 4
batch_size = 32
num_steps = 100
num_layers = 2
learning_rate = 1e-3
build_with_dropout=False


rawdata_path = '/dscrhome/yd44/imputation/testdata' #'/work/yd44/imputation/sample_perl'
chr = 'chr1'
num_epochs = 2

t = time.time()
g = build_lstm_graph(
	state_size=state_size, num_classes=num_classes, 
	batch_size=batch_size, num_steps=num_steps, 
	build_with_dropout=build_with_dropout)
print("It took", time.time() - t, "seconds to build the graph.")
t = time.time()
data = prepare_data(rawdata_path, chr, num_classes,batch_size, num_steps, name=None)
#train_network(g, rawdata_path, chr = 'chr1', num_epochs=2, num_steps = 20, batch_size = 32, num_class = 4 verbose = True, save=False)
print("It took", time.time() - t, "seconds to train for 3 epochs.")

