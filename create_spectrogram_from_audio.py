import os
from config import rawDatasetPath
from config import pixelPerSecond
from config import spectrogramsPath
from config import sliceSize
from subprocess import Popen,PIPE,STDOUT
from PIL import Image
from config import slicesPath

# currentPath=os.getcwd()
# print(currentPath)
def createSpectrograms():
	
	genres=os.listdir(rawDatasetPath)
	for genre in genres:
		genrePath=os.path.join(rawDatasetPath,genre)
		print("Start Generating spectrogram for --{}-- genre".format(genre))
		musics=os.listdir(genrePath)
		for i,music in enumerate(musics):
			createSpectrogramFromAudio(os.path.join(genrePath,music),music,genre)
			print("Spectrogram Created {} / {}".format(i+1,len(musics)))

	# creating slices for every spectrogram
	print("Start creating slices..")
	createSlices()

def createSpectrogramFromAudio(musicPath,musicName,genre):

	if not os.path.exists(spectrogramsPath):
		os.makedirs(spectrogramsPath)
	spectrogramGenrePath=os.path.join(spectrogramsPath,genre)

	if not os.path.exists(spectrogramGenrePath):
		os.makedirs(spectrogramGenrePath)

	#This below command is used to create the spectrogram from the audio
	command='sox '+"'"+musicPath+"'"+' -n spectrogram -Y 200 -X {} -m -r -o "{}.png"'.format(pixelPerSecond,
		                      os.path.join(spectrogramGenrePath,musicName[:-4])) 
	#Popen class is used to run the command in the shell
	popen=Popen(command,shell=True,stdin=PIPE,stdout=PIPE,stderr=STDOUT,close_fds=True,cwd=currentPath)

	output,errors=popen.communicate()
	print(output)
	if errors:
		print(errors)

def createSlices():
	if not os.path.exists(slicesPath):
		os.makedirs(slicesPath)
	genres=os.listdir(spectrogramsPath)
	for genre in genres:
		genrePath=os.path.join(slicesPath,genre)
		spectrogramGenrePath=os.path.join(spectrogramsPath,genre)
		if not os.path.exists(genrePath):
			os.makedirs(genrePath)
		for imageName in os.listdir(spectrogramGenrePath):
			createSliceForSpectrogram(imageName,os.path.join(genrePath,imageName[:-4]),spectrogramGenrePath)

def createSliceForSpectrogram(spectrogramName,slicePath,spectrogramGenrePath):
	image=Image.open(os.path.join(spectrogramGenrePath,spectrogramName))
	print(image)
	width,height=image.size

	#compute approximately number of 128*128 samples
	nb_slices=int(width/sliceSize)
	for i in range(nb_slices):
		print("Creating slice: ",(i+1),"/",nb_slices," for ",spectrogramName)
		startPixel=i*sliceSize
		imageCrop=image.crop((startPixel,0,startPixel+sliceSize,sliceSize)) # croping the image into 128*128 pixel
		imageCrop.save(slicePath+"_{}.png".format(i+1)) 

#createSpectrograms()