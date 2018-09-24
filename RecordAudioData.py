'''
RecordAudioData names, records and creates a 3 second audio clip 
for data collection for the CPP AI project.(exports as wav to root)
Built on top of and using existing example from PyAudio
Found here: https://people.csail.mit.edu/hubert/pyaudio
Used with California Polytechnic University California, Pomona Artificial Intelegence Club
Author: Data Collection Lead, Chris Leal
Date: 30 November, 2017
	
Note: While recording a word try your best to center it within the three second 
	window. Additionally this script requires pyAudio dependency to be installed
'''

import pyaudio
import wave
import time
import os

# CONSTANTS
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 16000
RECORD_SECONDS = 5

#_______________________

def recordAudio (fileName):
	p = pyaudio.PyAudio()
	stream = p.open(format = FORMAT, channels = CHANNELS, 
				rate = RATE, input = True, frames_per_buffer = CHUNK)
	print("***Started Recording***")
	
	frames = []
	for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	
	print("* done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	fileName = ifFileExist(fileName)
	wf = wave.open(fileName, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close

	print("**File: " + fileName + " has been recorded")

def recordAudioTest():
	print ("When recording remeber to speak clearly and louldy." + "\n"
		+ "Also try to keep the background noise to a minimum")
	recordDummyAudio()
	fileName = 'test0.wav'
	print ("Will record " + fileName)
	time.sleep(2)

	p = pyaudio.PyAudio()
	stream = p.open(format = FORMAT, channels = CHANNELS, 
				rate = RATE, input = True, frames_per_buffer = CHUNK)
	print("***Started Recording***")
	
	frames = []
	for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	
	print("* done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	#fileName = ifFileExist(os.path.join('./data/test_audio', fileName))
	wf = wave.open(os.path.join('./data/test_audio', fileName), 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close

	print("**File: " + fileName + " has been recorded")
	return fileName

	
def ifFileExist(fileName):
	counter = 0
	while True:
		if os.path.exists(fileName + str(counter) + ".wav"):
			counter += 1
		else:
			return fileName + str(counter) + ".wav"

def recordDummyAudio():
	#the mic would produce a wierd thumping for the first audio clip recorded
	#records and doesnot save a dummy clip to get rid of that thumping
	
	p = pyaudio.PyAudio()
	stream = p.open(format = FORMAT, channels = CHANNELS, 
				rate = RATE, input = True, frames_per_buffer = CHUNK)
	#print("***Started Recording***")
	
	frames = []
	for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	
	#print("* done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	
'''
print ("When recording remember to speak clearly and louldy." + "\n"
		+ "Also try to keep the background noise to a minimum")
recordDummyAudio()
fileName = "start"
while 1:
	fileName = input("Enter the word you will speak or enter ? to exit: ")
	if(fileName == "?"):
		print("*** exiting ***")
		break
	else:
		print ("Will record " + fileName)
		time.sleep(2)
		recordAudio(fileName)
'''