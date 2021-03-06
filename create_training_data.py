'''
PV-20 #Create preprocessing sound clips 
Creates training data by extracting the mfcc of data
The extracted data is then formatted to fit the CTC loss function

batched data = [[batch_size, max_length, 26], batch_targets, batch_seq_len]
original_targets = [original text of transcript]

Used With California Polythechnic University California, Pomona Voice Assitant Project
Author: Jason Chang
Project Manager: Gerry Fernando Patia
Date: 8 July, 2018
'''
import os
import numpy as np
import pickle
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from RecordAudioData import recordAudioTest

#Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

voxforge_data_dir = './Voxforge'

#Some configs
num_features = 26

if not os.path.isdir('./data'):
	os.makedirs('./data')

def list_files_for_speaker(folder):
	'''
	Generates a list of wav files from the voxforge dataset.
	Args:
		###If want specific speaker
		speaker: substring contained in the speaker's folder name, e.g. 'Aaron'
		###
		folder: base folder containing the downloaded voxforge data

	Returns: list of paths to the wavfiles
	'''
	#If you want specific speaker, add speaker arg into function
	#speaker_folders = [d for d in os.listdir(folder) if speaker in d]
	speaker_folders = [d for d in os.listdir(folder)]
	wav_files = []

	for d in speaker_folders:
		for f in os.listdir(os.path.join(folder, d, 'wav')):
			wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))

	return wav_files


def sparse_tuple_from(sequences, dtype=np.int32):
	'''
	Create a sparse representention of x.
	Args:
		sequences: a list of lists of type dtype where each element is a sequence
	Returns:
		A tuple with (indices, values, shape)
	'''
	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

	return indices, values, shape


def extract_features_and_targets(wav_file, txt_file):
	'''
	Extract MFCC features from an audio file and target character annotations from
	a corresponding text transcription
	Args:
		wav_file: audio wav file
		txt_file: text file with transcription

	Returns:
		features, targets, sequence length, original text transcription
	'''

	fs, audio = wav.read(wav_file)

	features = mfcc(audio, samplerate=fs, numcep= num_features)

	#Tranform in 3D array
	features = np.asarray(features[np.newaxis, :])
	features = (features - np.mean(features))/np.std(features)
	features_seq_len = features.shape[1]

	#Readings targets
	with open(txt_file, 'r') as f:
		for line in f.readlines():
			if line[0] == ';':
				continue

			#Get only the words between [a-z] and replace period for none
			original = ' '.join(line.strip().lower().split(' ')).replace('.', '').replace("'", '').replace('-', '').replace(',','')
			targets = original.replace(' ', '  ')
			targets = targets.split(' ')
	   
	#Adding blank label
	targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

	#Transform char into index
	targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
	return features, targets, features_seq_len, original


def make_batched_data(wav_files, batch_size=4):
	'''
	Generate batches of data given a list of wav files from the downloaded Voxforge data.
	Args:
		wav_files: list of wav files
		batch_size: batch size

	Returns:
		batched data, original text transcriptions
	'''

	batched_data = []
	original_targets = []
	num_batches = int(np.floor(len(wav_files) / batch_size))        #Rounds down

	for n_batch in range(num_batches):

		batch_features = []
		batch_targets = []
		batch_seq_len = []
		batch_original = []

		for f in wav_files[n_batch * batch_size: (n_batch+1) * batch_size]:

			txt_file = f.replace('\\wav\\', '\\txt\\').replace('.wav', '.txt')
			features, targets, seq_len, original = extract_features_and_targets(f, txt_file)

			batch_features.append(features)
			batch_targets.append(targets)
			batch_seq_len.append(seq_len)
			batch_original.append(original)

		# Creating sparse representation to feed the placeholder
		batch_targets = sparse_tuple_from(batch_targets)
		max_length = max(batch_seq_len)

		padded_features = np.zeros(shape=(batch_size, max_length, batch_features[0].shape[2]), dtype=np.float)

		for i, feat in enumerate(batch_features):
			padded_features[i, :feat.shape[1], :] = feat

		batched_data.append((padded_features, batch_targets, batch_seq_len))
		original_targets.append(batch_original)

	return batched_data, original_targets

def make_test_data(batch_size=4):
	'''
	Generates test data from a recroding
	Args:
		wav_files: list of wav files
		batch_size: batch size

	Returns:
		test_data
	'''
	if not os.path.isdir('./data/test_audio'):
		os.makedirs('./data/test_audio')
	fileName = recordAudioTest()

	wav_file = os.path.join('./data/test_audio', fileName)
	txt_file = './data/test_audio/test_audio_text.txt'
	features, _, seq_len, _ = extract_features_and_targets(wav_file, txt_file)
	
	padded_features = np.zeros(shape=(batch_size, seq_len, num_features), dtype=np.float)
	padded_features[0, :features.shape[1], :] = features

	#Add padding for seq_len
	seq_len = [seq_len]
	for x in range(batch_size-1):
		seq_len.append(0)

	test_data = [padded_features, seq_len]

	return test_data
	
if __name__ == '__main__':
	
	wav_files = list_files_for_speaker(voxforge_data_dir)

	batched_data, original_targets = make_batched_data(wav_files, batch_size=4)
	#batched data = [[batch_size, max_length, 26], batch_targets, batch_seq_len]
	#original_targets = [original text of transcript]

	with open('./data/train_data_batched.pkl', 'wb') as f:
		pickle.dump(batched_data, f)

	with open('./data/original_targets_batched.pkl', 'wb') as f:
		pickle.dump(original_targets, f)