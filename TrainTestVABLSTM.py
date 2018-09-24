'''
PV-23 #Create B-LSTM STT Model 
STT model that implements the Bi-Directional LSTM RNN with CTC loss

Used With California Polythechnic University California, Pomona Voice Assitant Project
Author: Jason Chang
Project Manager: Gerry Fernando Patia
Date: 10 June, 2018
'''

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
from create_training_data import make_test_data

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

checkpoint_steps = 10

if not os.path.isdir('./VABLSTMcheckpoints'):
	os.makedirs('./VABLSTMcheckpoints')

if not os.path.isdir('./VABLSTMsummaries/train'):
	os.makedirs('./VABLSTMsummaries/train')

if not os.path.isdir('./VABLSTMsummaries/test'):
	os.makedirs('./VABLSTMsummaries/test')

# Number of input features
num_features = 26
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 150
num_hidden = 50
batch_size = 4
learning_rate = 0.01
momentum = 0.9

# Load the data from pickle files
with open('./data/train_data_batched_Aaron_small.pkl', 'rb') as f:
	batched_data = pickle.load(f)

# Load original text targets
with open('./data/original_targets_batched_Aaron_small.pkl', 'rb') as f:
	original_targets = pickle.load(f)

#Gets one the batches
num_valid_batches = 1
num_train_batches = len(batched_data) - num_valid_batches

#Valid batch is the last one and Train batches are until last batch
valid_batches = batched_data[-num_valid_batches:]
valid_orig_targets = original_targets[-num_valid_batches:]
train_batches = batched_data[:num_train_batches]
train_orig_targets = original_targets[:num_train_batches]

del batched_data

tf.reset_default_graph()
###Creating placeholders####

# Has size [batch_size, max_stepsize, num_features], but the batch_size and max_stepsize can vary along each step
inputs = tf.placeholder(tf.float32, [None, None, num_features], name = 'inputs')
tf.summary.audio('audio', inputs, sample_rate=16000, max_outputs=3)
# Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32, name = 'targets')
# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None], name = 'seq_len')

def test_decoding(sess, decoded, input_feed_dict, input_original):
	'''
	Runs the classifier on a feed dictionary and prints the decoded predictions.
	Args:
		sess: tf.Session 
		dedcoded: list of length top_paths, where decoded[j] is a SparseTensor containing the decoded outputs
		input_feed_dict: decoded value in sparse value
		input_original: original value in string
	'''

	d = sess.run(decoded, feed_dict=input_feed_dict)

	str_decoded = ''.join([chr(x) for x in np.asarray(d[0][1]) + FIRST_INDEX])
	# Replacing blank label to none
	str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
	# Replacing space label to space
	str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

	print('Original: %s' % input_original)
	print('Decoded: %s' % str_decoded)
	print(' ')

def bidirectional_dynamic_rnn(inputs, seq_len):
	'''
	Creates Bi-Directiona LSTM Recurrent Neural Network
	Weights are truncated_normal rather than random because truncated normal overcomes saturation of time functions like sigmoid 
	(where if the value is too big/small [randomm], the neuron stops learning)
	Args:
		inputs: placeholder with size: [None, None, num_features]
		seq_len: placeholder with size: [None]

	Returns:
		logits
	'''

	with tf.name_scope("bidirectional_dynamic_rnn"):
		#Initializing the Weights and Biases
		layer = {'weights':tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name = 'weights'),
				 'biases':tf.Variable(tf.constant(0., shape=[num_classes]), name = 'biases')}

		# Defining the LSTM cells
		fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
		bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)

		# Use dynamic RNN to account for different sequence length. Second output is state which is not needed
		outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, sequence_length=seq_len, dtype=tf.float32)

		shape = tf.shape(inputs)
		batch_s, max_timesteps = shape[0], shape[1]

		#Reshaping to apply the same weights over the timesteps
		outputs = tf.reshape(outputs, [-1, num_hidden])

		#Applying linear transform
		logits = tf.matmul(outputs, layer['weights']) + layer['biases']

		#Reshaping back to the original shape
		logits = tf.reshape(logits, [batch_s, -1, num_classes])

		#Time major: [max_time, batch_size, num_classes]. Required for CTC loss fucntion
		logits = tf.transpose(logits, (1, 0, 2))

		#Record
		tf.summary.histogram("weights", layer['weights'])
		tf.summary.histogram("biases", layer['biases'])
		tf.summary.histogram("logits", logits)

		return logits

def train_neural_network(inputs, targets, seq_len):
	'''
	Trains Bi-Directiona LSTM Recurrent Neural Network using ctc loss
	Uses MomentumOptimizer rather than AdamOptimizer because has faster learning rate
	Args:
		inputs: placeholder with size: [None, None, num_features]
		targets: sparse_placeholder
		seq_len: placeholder with size: [None]

	Returns:
		ckpt models, matplotlib plot
	'''
	logits = bidirectional_dynamic_rnn(inputs, seq_len)

	with tf.name_scope("cost"):
		loss = tf.nn.ctc_loss(targets, logits, seq_len)
		cost = tf.reduce_mean(loss)

		#Record
		tf.summary.scalar('cost', cost)

	with tf.name_scope("train"):
		#Slower optimization
		#optimizer = tf.train.AdamOptimizer().minimize(cost)
		optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True).minimize(cost)

	with tf.name_scope("label_error_rate"):
		#Option 1: (it's slower but you'll get better results)
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len)
		# Option 2: 
		#decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

		# Inaccuracy: label error rate
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

		# Record the label error rate
		tf.summary.scalar('label_error_rate', ler)

	'''
	#For matplot
	points_train_cost = []
	points_train_ler = []
	points_valid_cost = []
	points_valid_ler = []
	'''

	saver = tf.train.Saver()
	merged = tf.summary.merge_all()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		train_writer = tf.summary.FileWriter('./VABLSTMsummaries/train', sess.graph)
		test_writer = tf.summary.FileWriter('./VABLSTMsummaries/test', sess.graph)
		train_writer.add_graph(sess.graph)
		test_writer.add_graph(sess.graph)

		for curr_epoch in range(num_epochs):
			start = time.time()
			train_cost = train_ler = 0
			for batch in range(num_train_batches):

				#print('Batch {} / {}'.format(batch+1, num_train_batches))
				#Input is with format of [4, ?(max_time_step), 26]
				train_feed = {inputs: train_batches[batch][0],
							  targets: train_batches[batch][1],
							  seq_len: train_batches[batch][2]}

				batch_cost, _, summary = sess.run([cost, optimizer, merged], train_feed)			#sess.run([optimizer, cost], feed) does not work? works in different code
				train_cost += batch_cost
				train_ler += sess.run(ler, feed_dict = train_feed)
				train_writer.add_summary(summary, curr_epoch * num_train_batches + batch)

			train_cost /= num_train_batches
			train_ler /= num_train_batches
			#points_train_cost.append(train_cost)
			#points_train_ler.append(train_ler)

			valid_cost = valid_ler = 0
			'''
			##Testing with validation set
			for batch in range(num_valid_batches):
				val_feed = {inputs: valid_batches[batch][0],
							targets: valid_batches[batch][1],
							seq_len: valid_batches[batch][2]}

				val_cost, val_ler, summary = sess.run([cost, ler, merged], feed_dict = val_feed)
				valid_cost += val_cost
				valid_ler += val_ler
				test_writer.add_summary(summary, curr_epoch * num_train_batches + batch)
			'''

			##Testing with training_batches to see if model is working correctly
			for batch in range(num_train_batches):

				val_feed = {inputs: train_batches[batch][0],
							targets: train_batches[batch][1],
							seq_len: train_batches[batch][2]}

				val_cost, val_ler, summary = sess.run([cost, ler, merged], feed_dict=val_feed)

				valid_cost += val_cost
				valid_ler += val_ler
				test_writer.add_summary(summary, curr_epoch * num_train_batches + batch)

			valid_cost /= num_valid_batches
			valid_ler /= num_valid_batches

			#points_valid_cost.append(valid_cost)
			#points_valid_ler.append(valid_ler)

			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
			print(log.format(curr_epoch, num_epochs, train_cost, train_ler, val_cost, val_ler, time.time() - start))

			#Saves a model after every number of checkpoint_steps
			if curr_epoch % checkpoint_steps == 0:
				saver.save(sess, './VABLSTMcheckpoints/model_BVA_Aaron_small.ckpt')
				print('Train decoding: ')

				train_feed = {inputs: train_batches[0][0],
							  targets: train_batches[0][1],
							  seq_len: np.asarray(train_batches[0][2])}

				train_original = ' '.join(train_orig_targets[0])

				test_decoding(sess, decoded, train_feed, train_original)

				print('Validation decoding: ')

				'''
				#Decoding with Validation set
				val_feed = {inputs: valid_batches[0][0],
							targets: valid_batches[0][1],
							seq_len: valid_batches[0][2]}

				valid_original = ' '.join(valid_orig_targets[0])
				
				test_decoding(sess, decoded, val_feed, valid_original)
				'''

				#Decoding the training batch to check if its working
				randBatch = random.randint(0, len(train_batches)-1)
				val_feed = {inputs: train_batches[randBatch][0],
							targets: train_batches[randBatch][1],
							seq_len: train_batches[randBatch][2]}

				valid_original = ' '.join(train_orig_targets[randBatch])

				test_decoding(sess, decoded, val_feed, valid_original)
		'''
		#Show graph with matplot
		plt.figure(1)
		plt.subplot(121)
		plt.plot(range(len(points_train_cost)), points_train_cost, 'b.', linestyle = '-', linewidth=2)
		plt.plot(range(len(points_valid_cost)), points_valid_cost, 'g.', linestyle = '-', linewidth=2)
		plt.title('cost')
		plt.subplot(122)
	
		plt.plot(range(len(points_valid_ler)), points_train_ler, 'b.', linestyle = '-', linewidth=2)
		plt.plot(range(len(points_train_ler)), points_valid_ler, 'g.', linestyle = '-', linewidth=2)
		plt.title('ler')
		plt.show()
		'''

def use_neural_network(inputs, seq_len):
	logits = bidirectional_dynamic_rnn(inputs, seq_len)
	decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len)

	spec_path = './VABLSTMcheckpoints/model_BVA_Aaron_small.ckpt'
	data = make_test_data()
	
	saver = tf.train.Saver()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, spec_path)
		print("Model restored from file: %s" % spec_path)

		test_feed = {inputs: data[0],
					 seq_len: np.asarray(data[1])}

		test_original = 'his slim hands gripped the edges of the table'

		test_decoding(sess, decoded, test_feed, test_original)

#train_neural_network(inputs, targets, seq_len)
use_neural_network(inputs, seq_len)