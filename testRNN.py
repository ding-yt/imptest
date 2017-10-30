import numpy as np
import tensorflow as tf
import time
import os

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_lstm_graph(
    state_size = 128,
    num_classes = 4,
    batch_size = 32,
    num_steps = 100,
    num_layers = 2,
    learning_rate = 1e-3,
    build_with_dropout=False):

    reset_graph()

    input_x = tf.placeholder(tf.int32, [batch_size, num_steps, num_classes], name='input_placeholder')
    input_y = tf.placeholder(tf.int32, [batch_size, num_steps, num_classes], name='labels_placeholder')
    dropout = tf.constant(0.9)
	   
    
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
      
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
    
    #rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, input_x, initial_state=init_state, swap_memory=True)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, input_x, initial_state=init_state)
    #rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, input_x, initial_state=init_state)
    #rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, input_x, dtype=tf.float32)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(input_y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
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


###################

t = time.time()
g = build_lstm_graph()
# 	state_size=state_size, num_classes=num_classes, 
# 	batch_size=batch_size, num_steps=num_steps, 
# 	build_with_dropout=build_with_dropout)
print("It took", time.time() - t, "seconds to build the graph.")


