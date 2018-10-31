# Music-Genre_Classification_using-tflearn
it will classify the genre of a music.

### Tools Required
tflearn  
sox(for creating spectrogram image from music)   
tensorflow(backend)  
numpy  

### Structured of dataset
The structure of dataset is like as below  
data  
   |  
   |_________RawDataset__________  
   |                             |____________________genre1_______abc.mp3  
   |_________Spectrograms        |_____________genre2      |_______xyz.mp3  
   |                                                 |_______exa1.mp3  
   |_________Slices                                  |_______exa2.mp3  
   |  
   |_________Dataset  
                  |_____train_X.pkl  
                  |_____train_y.pkl  
                  |_____validation_X.pkl  
                  |_____validation_y.pkl  
                  |_____test_X.pkl  
                  |_____test_y.pkl  
   The structure for Spectrograms will be same as structure of RawDataset.The  only change will be instead of abc.mp3 there will be abc.png,xyz.png.  
   Same for the Slices also.  
                        

