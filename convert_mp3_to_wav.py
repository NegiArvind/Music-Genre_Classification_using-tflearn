import os
from config import rawDatasetPath

from pydub import AudioSegment
# sound = AudioSegment.from_mp3("/path/to/file.mp3")
# sound.export("/output/path/file.wav", format="wav")

tempPath='temp'
if not os.path.exists(tempPath):
	os.makedirs(tempPath)
genres=os.listdir(rawDatasetPath)
for genre in genres:
	genrePath=os.path.join(rawDatasetPath,genre)
	if not os.path.exists(os.path.join(tempPath,genre)):
		os.makedirs(os.path.join(tempPath,genre))
	musics=os.listdir(genrePath)
	for i,music in enumerate(musics):
		tempMusicPath=os.path.join(tempPath,genre)+"/"+music[:-3]+'.wav'
		#os.rename(os.path.join(genrePath,music),os.path.join(genrePath,music[:-3]+"mp3")) # converting mp3 file into wav
		sound=AudioSegment.from_mp3(os.path.join(genrePath,music))
		sound.export(os.path.join(tempMusicPath),format='wav')