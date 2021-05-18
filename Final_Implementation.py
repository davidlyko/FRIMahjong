import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import *
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.layers import Flatten
import keras
import random

#random.seed(42)

'''
crack1 = cv2.imread("crack1_1.jpg",cv2.COLOR_BGR2GRAY)
vCroppedImage=crack1[0:500,0:500]
vResizedImage =cv2.resize(crack1,(700,700))
#cv2.imshow("Crack 1",vResizedImage)
grayImage = cv2.cvtColor(vResizedImage,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image",grayImage)
'''

images = sorted(glob.glob('C:/Users/geral/crack/crack_1/*.jpg'))#Directory of the first folder
images1 = sorted(glob.glob('C:/Users/geral/crack/crack_2/*.jpg'))#Directory of the second folder
images2 = sorted(glob.glob('C:/Users/geral/crack/crack_3/*.jpg'))#Directory of the third folder
images3 = sorted(glob.glob('C:/Users/geral/crack/crack_4/*.jpg'))#Directory of the forth folder
images4 = sorted(glob.glob('C:/Users/geral/crack/crack_5/*.jpg'))#Directory of the fifth folder
images5 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_5/*.jpg'))
images6 = sorted(glob.glob('C:/Users/geral/dot/dot_9/*.jpg'))
images7 = sorted(glob.glob('C:/Users/geral/crack/crack_6/*.jpg'))
images8 = sorted(glob.glob('C:/Users/geral/crack/crack_8/*.jpg'))
images9 = sorted(glob.glob('C:/Users/geral/crack/crack_9/*.jpg'))
images10 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_1/*.jpg'))
images11 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_2/*.jpg'))
images12 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_3/*.jpg'))
images13 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_4/*.jpg'))
images14 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_6/*.jpg'))
images15 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_7/*.jpg'))
images16 = sorted(glob.glob('C:/Users/geral/bamboo/bamboo_8/*.jpg'))
images17 = sorted(glob.glob('C:/Users/geral/dot/dot_4/*.jpg'))
images18 = sorted(glob.glob('C:/Users/geral/dot/dot_5/*.jpg'))
images19 = sorted(glob.glob('C:/Users/geral/dot/dot_6/*.jpg'))
images20 = sorted(glob.glob('C:/Users/geral/dot/dot_7/*.jpg'))
images21 = sorted(glob.glob('C:/Users/geral/dot/dot_8/*.jpg'))
a = len(images)
b = len(images1)
c = len(images2)
d = len(images3)
e = len(images4)
f = len(images5)
g = len(images6)
h = len(images7)
j = len(images8)
k = len(images9)
l = len(images10)
m = len(images11)
n = len(images12)
o = len(images13)
p = len(images14)
q = len(images15)
r = len(images16)
s = len(images17)
t = len(images18)
u = len(images19)
v = len(images20)
w = len(images21)

labels = np.zeros((a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v+w,22)) #Used to label images in a folder. First index is the total number of pictures in the folder, and the second index is the number of folders
croppedImages=np.zeros((a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v+w,500,500))#Used to create an empty array, first index is the total number of images, second and third are x,y pixels (Have to be the same as line 23)
# for loop to store labels and images

for index,i in enumerate(images): #Same for loop for every label and image registration
    crack1 = cv2.imread(i) #reads the image for a folder
    crack1 = cv2.cvtColor(crack1,cv2.COLOR_BGR2GRAY)#makes image gray
    vResizedImage =cv2.resize(crack1,(500,500))#Resizes image, (700,700) should be used in line 17
    labels[index,:]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#Crack 1; the first label array will always be [1,0], any other label array after will be [0,1]. Ex: three folder, label 1: [1,0,0], label 2: [0,1,0], label three [0,0,1], etc
    croppedImages[index,:,:]=vResizedImage #Adds the resized images to the croppedImages array

# for loop to store 2nd set of labels and images
for index,i in enumerate(images1): # index inside enumerate will be for the 2nd set of images
    crack2 = cv2.imread(i) #Changed the name from crack1 to crack2. 
    crack2 = cv2.cvtColor(crack2,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    #vCroppedImage=crack2[0:500,0:500] #Same as previous for loop
    vResizedImage =cv2.resize(crack2,(500,500)) #Same as previous for loop
    labels[index+a,:]=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#Crack 2, ([0,0,1]) for 3 folders, etc   
    croppedImages[index+a,:,:]=vResizedImage # We use index+len(images) for this and the previous line because we want to store these images and labels after the 1st set
#for loop to store 3rd set of labels and images
for index,i in enumerate(images2): # 
    crack3 = cv2.imread(i) 
    crack3 = cv2.cvtColor(crack3,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    #vCroppedImage=crack3[0:500,0:500] #Same as previous for loop
    vResizedImage =cv2.resize(crack3,(500,500)) #Same as previous for loop
    labels[index+a+b,:]=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])#Crack 3, ([0,0,1]) for 3 folders, etc   
    croppedImages[index+a+b,:,:]=vResizedImage
#for loop to store 4th set of labels and images
for index,i in enumerate(images3): # 
    crack4 = cv2.imread(i) 
    crack4 = cv2.cvtColor(crack4,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    #vCroppedImage=crack4[0:500,0:500] #Same as previous for loop
    vResizedImage =cv2.resize(crack4,(500,500)) #Same as previous for loop
    labels[index+a+b+c,:]=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c,:,:]=vResizedImage
#for loop to store 5th set of labels and images
for index,i in enumerate(images4): #
    crack5 = cv2.imread(i) 
    crack5 = cv2.cvtColor(crack5,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    #vCroppedImage=crack5[0:500,0:500] #Same as previous for loop
    vResizedImage =cv2.resize(crack5,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d,:]=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d,:,:]=vResizedImage
    
for index,i in enumerate(images5): # 
    bamboo5 = cv2.imread(i) 
    bamboo5 = cv2.cvtColor(bamboo5,cv2.COLOR_BGR2GRAY) #Same as previous for loop    
    vResizedImage =cv2.resize(bamboo5,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e,:]=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e,:,:]=vResizedImage
    
for index,i in enumerate(images6): # 
    dot9 = cv2.imread(i) 
    dot9 = cv2.cvtColor(dot9,cv2.COLOR_BGR2GRAY) #Same as previous for loop    
    vResizedImage =cv2.resize(dot9,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f,:]=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f,:,:]=vResizedImage

for index,i in enumerate(images7): # 
    crack6 = cv2.imread(i) 
    crack6 = cv2.cvtColor(crack6,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(crack6,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g,:]=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g,:,:]=vResizedImage

for index,i in enumerate(images8): # 
    crack8 = cv2.imread(i) 
    crack8 = cv2.cvtColor(crack8,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(crack8,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h,:]=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h,:,:]=vResizedImage

for index,i in enumerate(images9): #
    crack9 = cv2.imread(i) 
    crack9 = cv2.cvtColor(crack9,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(crack9,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j,:]=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j,:,:]=vResizedImage

for index,i in enumerate(images10): #
    bamboo1 = cv2.imread(i) 
    bamboo1 = cv2.cvtColor(bamboo1,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo1,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k,:]=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k,:,:]=vResizedImage

for index,i in enumerate(images11): #
    bamboo2 = cv2.imread(i) 
    bamboo2 = cv2.cvtColor(bamboo2,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo2,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l,:,:]=vResizedImage

for index,i in enumerate(images12): #
    bamboo3 = cv2.imread(i) 
    bamboo3 = cv2.cvtColor(bamboo3,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo3,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m,:,:]=vResizedImage

for index,i in enumerate(images13): #
    bamboo4 = cv2.imread(i) 
    bamboo4 = cv2.cvtColor(bamboo4,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo4,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n,:,:]=vResizedImage

for index,i in enumerate(images14): #
    bamboo6 = cv2.imread(i) 
    bamboo6 = cv2.cvtColor(bamboo6,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo6,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o,:,:]=vResizedImage

for index,i in enumerate(images15): #
    bamboo7 = cv2.imread(i) 
    bamboo7 = cv2.cvtColor(bamboo7,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo7,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p,:,:]=vResizedImage

for index,i in enumerate(images16): #
    bamboo8 = cv2.imread(i) 
    bamboo8 = cv2.cvtColor(bamboo8,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(bamboo8,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q,:,:]=vResizedImage

for index,i in enumerate(images17): #
    dot4 = cv2.imread(i) 
    dot4 = cv2.cvtColor(dot4,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(dot4,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r,:,:]=vResizedImage

for index,i in enumerate(images18): #
    dot5 = cv2.imread(i) 
    dot5 = cv2.cvtColor(dot5,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(dot5,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s,:,:]=vResizedImage

for index,i in enumerate(images19): #
    dot6 = cv2.imread(i) 
    dot6 = cv2.cvtColor(dot6,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(dot6,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t,:,:]=vResizedImage

for index,i in enumerate(images20): #
    dot7 = cv2.imread(i) 
    dot7 = cv2.cvtColor(dot7,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(dot7,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u,:,:]=vResizedImage

for index,i in enumerate(images21): #
    dot8 = cv2.imread(i) 
    dot8 = cv2.cvtColor(dot8,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vResizedImage =cv2.resize(dot8,(500,500)) #Same as previous for loop
    labels[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v,:]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])   
    croppedImages[index+a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v,:,:]=vResizedImage

print(croppedImages[0,:,:])
print()
print(croppedImages[-1,:,:])
print()
print(labels[0,:]) # prints out 
print()
print(labels[-1,:]) # prints out 
print()

#Model building, link: https://colab.research.google.com/drive/1xpEjpdlCpJewlAtwszSlYsVwoZxPr8EH?usp=sharing#scrollTo=F2hN3QmAIeB8
#model = Sequential() # prints out [0,1] x-train=croppedImages, y-train=labels



#print(croppedImages.shape) # (total number of images, resolution,resolution
print(labels.shape)

xTrain = croppedImages[:]
yTrain = labels[:]
xTest = croppedImages[:]
yTest = labels[:]
print(xTrain.shape)
xTrain = xTrain.reshape(a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v+w, 500*500).astype('float32')
xTest = xTest.reshape(a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+q+r+s+t+u+v+w, 500*500).astype('float32')
xTrain = xTrain / 255
xTest = xTest / 255
print(xTrain.shape)
#yTrain = keras.utils.to_categorical(yTrain)
#yTest = keras.utils.to_categorical(yTest)

'''
plt.imshow(croppedImages[0,:,:], cmap='gray')
plt.title("Label : {}".format(labels[0,:]))
plt.xticks([])
plt.yticks([])
#plt.show()

#Displays 9 images from the crack 1 folder
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(croppedImages[i,:,:], cmap='gray', interpolation='none')
  plt.title("Label: {}".format(labels[i]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
#Displays 9 images from the crack 2 folder
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  #print (x_train[1,:,:].shape)
  plt.imshow(croppedImages[i+a,:,:], cmap='gray', interpolation='none')
  plt.title("Label: {}".format(labels[i+a]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
#Displays 9 images from the crack 3 folder
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  #print (x_train[1,:,:].shape)
  plt.imshow(croppedImages[i+a+b,:,:], cmap='gray', interpolation='none')
  plt.title("Label: {}".format(labels[i+a+b]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
#Displays 9 images from the crack 4 folder
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  #print (x_train[1,:,:].shape)
  plt.imshow(croppedImages[i+a+b+c,:,:], cmap='gray', interpolation='none')
  plt.title("Label: {}".format(labels[i+a+b+c]))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
#Displays 9 images from the crack 5 folder
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  #print (x_train[1,:,:].shape)
'''

model = Sequential()

model.add(Dense(512, activation='relu', input_dim=500*500))

model.add(Dense(256, activation='relu')) 

model.add(Dense(128,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(22, activation='softmax'))

print (model.summary())


model.compile(loss='categorical_crossentropy', optimizer=RMSprop() ,metrics=['accuracy'])

#callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(xTrain, yTrain, batch_size=4, epochs=20, verbose=1, validation_data=(xTest, yTest))
#model.predict_on_batch(32)
accuracy = model.evaluate(xTest, yTest, verbose=0)
print ('Accuracy is:', accuracy)
#
#callbacks = [callback]
#model.save("name")
#modelLoad = keras.models.load_model("name")

#prediction = modelLoad.predict(xTest)
#print(prediction.shape())
#print(prediction[0,:])
#print(yTest[0,:])
cv2.waitKey(0);
