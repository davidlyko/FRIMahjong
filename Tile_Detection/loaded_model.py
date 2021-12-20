import Final_Implementation
from tensorflow import keras
import numpy as np
import tensorflow as tf
import glob

modelLoad = keras.models.load_model("Tile_detection")

images = sorted(glob.glob('C:/Users/18452/FRI/FRI_PROJECT/FRIMahjong/crack_1/*.jpg'))
a = len(images)
s = (a, 1)
labels = np.zeros(s)
croppedImages = np.zeros(a, 100, 100, 3)

for index,i in enumerate(images): #Same for loop for every label and image registration
    crack1 = cv2.imread(i) #reads the image for a folder
    #crack1 = cv2.cvtColor(crack1,cv2.COLOR_BGR2GRAY)#makes image gray
    vResizedImage =cv2.resize(crack1,(100,100))#Resizes image, (700,700) should be used in line 17
    labels[index,:]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#Crack 1; the first label array will always be [1,0], any other label array after will be [0,1]. Ex: three folder, label 1: [1,0,0], label 2: [0,1,0], label three [0,0,1], etc
    croppedImages[index,:,:]=vResizedImage #Adds the resized images to the croppedImages array

xTrain = croppedImages[:]
yTrain = labels[:]
xTest = croppedImages[:]
yTest = labels[:]
print(xTrain.shape)
xTrain = xTrain.reshape(a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v+w, 100, 100, 3).astype('float32')
xTest = xTest.reshape(a, 100, 100, 3).astype('float32')
xTrain = xTrain / 255
xTest = xTest / 255

prediction = modelLoad.predict(xTest)
print(prediction.shape())
print(prediction[0,:])
print(yTest[0,:])
print(prediciton)
