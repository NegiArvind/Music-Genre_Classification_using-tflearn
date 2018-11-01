import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

def createModel(nbClasses,imageSize):
  
  print("Creating model...")
  input_layer=input_data(shape=[None,imageSize,imageSize,1],name='input')
  
  convNetwork=conv_2d(incoming=input_layer,nb_filter=32,filter_size=3,activation='elu',weights_init='Xavier')
  convNetwork=max_pool_2d(incoming=convNetwork,kernel_size=[2,2])
  
  convNetwork=conv_2d(incoming=convNetwork,nb_filter=64,filter_size=3,activation='elu',weights_init='Xavier')
  convNetwork=max_pool_2d(incoming=convNetwork,kernel_size=[2,2])
  
  convNetwork=conv_2d(incoming=convNetwork,nb_filter=128,filter_size=3,activation='elu',weights_init='Xavier')
  convNetwork=max_pool_2d(incoming=convNetwork,kernel_size=[2,2])
  
  convNetwork=conv_2d(incoming=convNetwork,nb_filter=256,filter_size=3,activation='elu',weights_init='Xavier')
  convNetwork=max_pool_2d(incoming=convNetwork,kernel_size=[2,2])
  
  convNetwork=conv_2d(incoming=convNetwork,nb_filter=512,filter_size=3,activation='elu',weights_init='Xavier')
  convNetwork=max_pool_2d(incoming=convNetwork,kernel_size=[2,2])
  
  convNetwork=fully_connected(incoming=convNetwork,n_units=1024,activation='elu') # 1024 nodes in hidden layer
  convNetwork=dropout(incoming=convNetwork,keep_prob=0.5)
  
  convNetwork=fully_connected(incoming=convNetwork,n_units=512,activation='elu') # 1024 nodes in hidden layer
  convNetwork=dropout(incoming=convNetwork,keep_prob=0.5)
  
  convNetwork=fully_connected(incoming=convNetwork,n_units=nbClasses,activation='softmax') # nbClasses nodes in output layer
  convNetwork=regression(incoming=convNetwork,metric='accuracy',optimizer='rmsprop',loss='categorical_crossentropy') # it will optimize the loss
  
  model=tflearn.DNN(convNetwork)
  print("Model successfully created! ")
  return model


