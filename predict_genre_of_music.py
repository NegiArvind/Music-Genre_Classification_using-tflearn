import argparse
from model import createModel
from config import validationRatio,testRatio,sliceSize,nbEpoch,batchSize
from create_spectrogram_from_audio import createSpectrograms
from config import slicesPath
from datasetTools import getDataset
import os
import random
import string 
import numpy as np 
from config import sliceSize
from collections import Counter

currentPath=os.getcwd()

def createSpectrogramForOneMusic(musicPath,sliceSize):
  
  temp=musicPath.split('/')
  musicName=temp[len(temp)-1]
  musicName=musicName[:-4]
  command='sox '+"'"+musicPath+"'"+' -n spectrogram -Y 200 -X {} -m -r -o "{}.png"'.format(pixelPerSecond,
                                                                                           driveLocation+musicName)
  
  popen=Popen(command,shell=True,stdin=PIPE,stdout=PIPE,stderr=STDOUT,close_fds=True,cwd=currentPath)
  
  output,errors=popen.communicate()
  print(output)
  if errors:
    print(errors)
    
  image=Image.open(driveLocation+musicName+".png") #Opening the image
  width,height=image.size
  nbSamples=int(width/sliceSize)
  os.remove(driveLocation+musicName+".png") # Removing the image(no need of it)
  slicedImages=[]
  for i in range(nbSamples):
    print("Creating slice: ",(i+1),"/",nbSamples," for ",musicName+'.png')
    startPixel=i*sliceSize
    imageCrop=image.crop((startPixel,0,startPixel+sliceSize,sliceSize)) # croping the image into 128*128 pixel
    slicedImages.append(imageCrop)
    
  return slicedImages

def getImageDataFromSlice(musicPath,imageSize):
  slicedImages=createSpectrogramForOneMusic(musicPath,imageSize)
  imageData=[]
  for image in slicedImages:
    image=image.resize((imageSize,imageSize),resample=Image.ANTIALIAS)
    image=np.asarray(image,dtype=np.uint8).reshape(imageSize,imageSize,1)
    image=image/255
    imageData.append(image)
  return imageData

def predictGenre(musicPath,imageSize):
	imagesData=getImageDataFromSlice(musicPath,imageSize)
	#Load the weights(model)
	print("Loading weights...")
	model.load(driveLocation+'TrainedModel/genreClassifierModel.tflearn')
	print('weights loaded!')

	predictions=model.predict(imagesData) # returns the list of prediction
	best_class_indices=np.argmax(predictions,axis=1)
	best_class_probabilities=predictions[np.arange(len(best_class_indices)),best_class_indices]

	count=0
	most_common,num_most_common = Counter(best_class_indices).most_common(1)[0] # return value which repeate most and number of times it repeates

	sum=0
	if num_most_common>1:
		for i in range(len(best_class_indices)):
			if most_common==best_class_indices[i]:
				sum+=best_class_probabilities[i]
		meanProbability=sum/len(best_class_indices)
		print('Higher vote ---  %s:  %.3f' % (genres[most_common],meanProbability*100))

	else:
		print('maximum prob --- %s:  %.3f ' %(genres[np.argmax(best_class_probabilities)],best_class_probabilities[np.argmax[best_class_probabilities]]*100))
    
if __name__=='__main__':
  predictGenre(driveLocation+'musics/3QE-CHAOS.mp3',sliceSize)