from config import slicesPath
from random import shuffle
from config import datasetPath
import pickle
from config import sliceSize
from config import validationRatio
from config import testRatio
import os
from PIL import Image
import numpy as np


def getDataset(sliceSize,validationRatio,testRatio,mode):

	if not os.path.exists(datasetPath+"/train_X.pkl"):
		createDatasetFromSlices(sliceSize,validationRatio,testRatio)
	else:
		print("Using existing data")
	return loadDataset(mode)


def loadDataset(mode):
	if mode=="train":
		print("Loading training and validation dataset...")
		with open("{}/train_X.pkl".format(datasetPath),'rb') as f:
			train_X=pickle.load(f)
		with open("{}/train_y.pkl".format(datasetPath),'rb') as f:
			train_y=pickle.load(f)
		with open("{}/validation_X.pkl".format(datasetPath),'rb') as f:
			validation_X=pickle.load(f)
		with open("{}/validation_y.pkl".format(datasetPath),'rb') as f:
			validation_y=pickle.load(f)
		print("Training and validation dataset loaded")
		return train_X,train_y,validation_X,validation_y

	else:
		print("Loading testing dataset...")
		with open("{}/test_X.pkl".format(datasetPath),'rb') as f:
			test_X=pickle.load(f)
		with open("{}/test_y.pkl".format(datasetPath),'rb') as f:
			test_y=pickle.load(f)
		print("Testing dataset loaded")
		return test_X,test_y


def createDatasetFromSlices(sliceSize,validationRatio,testRatio):
  
  print("Start Creating Dataset...")
  genres=os.listdir(slicesPath)
#   data=[]
#   for genre in genres:
#     for imageName in os.listdir(os.path.join(slicesPath,genre)):
#       imgData=getImageData(os.path.join(slicesPath,genre)+"/"+imageName,sliceSize)
#       label=[1 if genre==g else 0 for g in genres]
#       print(label)
#       data.append((imgData,label))
      
#   shuffle(data)
#   X,y=zip(*data)
  
#   #split data
#   validationNb=int(len(X)*validationRatio)
#   testNb=int(len(X)*testRatio)
#   trainNb=len(X)-(validationNb+testNb)
  
#   #prepare for Tflearn at the same time
#   train_X=np.array(X[:trainNb]).reshape([-1,sliceSize,sliceSize,1])
#   train_y=np.array(y[:trainNb])
#   validation_X=np.array(X[trainNb:trainNb+validationNb]).reshape(-1,sliceSize,sliceSize,1)
#   validation_y=np.array(y[trainNb:trainNb+validationNb])
#   test_X=np.array(X[-testNb:]).reshape(-1,sliceSize,sliceSize,1)
#   test_y=np.array(y[-testNb:])
#   print("Dataset successfully created ")
  
  trainSet=[]
  testSet=[]
  validationSet=[]
  
  for genre in genres:
    
    print(genres)
    print(genre)
    images=os.listdir(os.path.join(slicesPath,genre))
    shuffle(images)
    nb_validation=int(len(images)*validationRatio)
    nb_test=int(len(images)*testRatio)
    print("images",len(images))
    print("validation",nb_validation)
    print("test",nb_test)
    
    i=0;
    for _ in range(len(images)-nb_validation-nb_test):
      print("train",str(i))
      imgData=getImageData(os.path.join(slicesPath,genre)+"/"+images[i],sliceSize)
      label=[1 if genre==g else 0 for g in genres]
      print(label)
      trainSet.append((imgData,label))
      i+=1
       
    for _ in range(nb_validation):
      print("validation",str(i))
      imgData=getImageData(os.path.join(slicesPath,genre)+"/"+images[i],sliceSize)
      label=[1 if genre==g else 0 for g in genres]
      print(label)
      validationSet.append((imgData,label))
      i+=1
      
    for _ in range(nb_test):
      print("test",str(i))
      imgData=getImageData(os.path.join(slicesPath,genre)+"/"+images[i],sliceSize)
      label=[1 if genre==g else 0 for g in genres]
      print(label)
      testSet.append((imgData,label))
      i+=1
      
  train_X,train_y=zip(*trainSet)
  validation_X,validation_y=zip(*validationSet)
  test_X,test_y=zip(*testSet)
  
  train_X=np.array(train_X).reshape(-1,sliceSize,sliceSize,1)
  train_y=np.array(train_y)
  validation_X=np.array(validation_X).reshape(-1,sliceSize,sliceSize,1)
  validation_y=np.array(validation_y)
  test_X=np.array(test_X).reshape(-1,sliceSize,sliceSize,1)
  test_y=np.array(test_y)
  print("Dataset successfully created ")
  
  # save the dataset 
  saveDataset(train_X,train_y,validation_X,validation_y,test_X,test_y)

def saveDataset(train_X,train_y,validation_X,validation_y,test_X,test_y):
	if not os.path.exists(datasetPath):
		os.makedirs(datasetPath)

	print("Start Saving the dataset...")
	with open("{}/train_X.pkl".format(datasetPath),'wb') as f:
		pickle.dump(train_X,f)
	with open("{}/train_y.pkl".format(datasetPath),'wb') as f:
		pickle.dump(train_y,f)
	with open("{}/validation_X.pkl".format(datasetPath),'wb') as f:
		pickle.dump(validation_X,f)
	with open("{}/validation_y.pkl".format(datasetPath),'wb') as f:
		pickle.dump(validation_y,f)
	with open("{}/test_X.pkl".format(datasetPath),'wb') as f:
		pickle.dump(test_X,f)
	with open("{}/test_y.pkl".format(datasetPath),'wb') as f:
		pickle.dump(test_y,f)

	print("Dataset successfully Saved")


def getImageData(imagePath,imageSize):
	image=Image.open(imagePath)
	image=image.resize((imageSize,imageSize),resample=Image.ANTIALIAS)
	image=np.asarray(image,dtype=np.uint8).reshape(imageSize,imageSize,1)
	image=image/255
	return image

