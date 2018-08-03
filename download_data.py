import os
import urllib
import tarfile
from bs4 import BeautifulSoup

#Set two speakers for training/testing sets
file_name = 'tgz'  # defines substring that is searched for in the tar file names
voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
target_folder = './Voxforge'
num_of_audio_folders = 30


def download_file(url, target_folder):
	"""
	Downloads and extracts a tar file given a URL and a target folder.
	"""
	stream = urllib.request.urlopen(url)
	tar = tarfile.open(fileobj=stream, mode="r|gz")

	for item in tar:
		tar.extract(item, target_folder)


if __name__ == '__main__':

	if not os.path.isdir(target_folder):
		os.makedirs(target_folder)

	html_page = urllib.request.urlopen(voxforge_url)
	soup = BeautifulSoup(html_page, "html5lib")
	#List all links
	links = soup.findAll('a')

	#Download files for the specified speaker, 12 for extra link without tgz
	speaker_refs = [l['href'] for l in links[0:num_of_audio_folders+12] if file_name in l['href']]
	print(speaker_refs)
	print(len(speaker_refs))

	
	for i, ref in enumerate(speaker_refs):
		print('Downloading {} / {} files'.format(i+1, len(speaker_refs)))
		download_file(voxforge_url + '/' + ref, target_folder)
	