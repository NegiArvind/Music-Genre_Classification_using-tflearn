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


parser=argparse.ArgumentParser()
parser.add_argument("mode",help="Train or tests the CNN",nargs='+',choices=["train","test","slice"])
args=parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Slice size: {}".format(sliceSize))
print("--------------------------")

if "slice" in args.mode:
	print("inside slice")
	createSpectrograms() # it will create slices for every spectrograms
	sys.exit()

nbClasses=len(os.listdir(slicesPath))

#create model
model=createModel(nbClasses,sliceSize)

if "train" in args.mode:

	#create or load new dataset
	train_X,train_y,validation_X,validation_y=getDataset(sliceSize,validationRatio,testRatio,mode="train")

	# #Define run id for graphs
	# run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().
	# 	choice(string.ascii_uppercase) for _ in range(10))
	run_id="MusicGenres"

	#Train the model
	model.fit(train_X,train_y,n_epoch=nbEpoch,batch_size=batchSize,shuffle=True,
		validation_set=(validation_X,validation_y),snapshot_step=100,show_metric=True,run_id=run_id)

	print("Model Trained")

	#Save trained model
	print("Saving the weights")
	model.save('genreClassifierModel.tflearn')
	print("Weight saved!")

if "test" in args.mode:

	#load the test Dataset
	test_X,test_y=getDataset(sliceSize,validationRatio,testRatio,mode="test")

	#Load the weights(model)
	print("Loading weights...")
	model.load('genreClassifierModel.tflearn')
	print('weights loaded!')

	genres=os.listdir(slicesPath)

	predictions=model.predict(test_X) # returns the list of prediction
	best_class_indices=np.argmax(predictions,axis=1)
	best_class_probabilities=predictions[np.arange(len(best_class_indices)),best_class_indices]

	y=np.argmax(test_y,axis=1)

	for i in range(len(best_class_indices)):
		genreType=''
		if best_class_probabilities[i]>0.52:
			genreType=genres[best_class_indices[i]]
		else:
			genreType='Unknown'
		print('%4d  %s: \t %.3f \t %s' % (i,genreType, best_class_probabilities[i],genres[y[i]]))
	
	accuracy=np.mean(np.equal(best_class_indices,y))
	print("Accuracy : {}".format(accuracy))
	testAccuracy=model.evaluate(test_X,test_y)[0]
	print("Test Accuracy : {}".format(testAccuracy))


