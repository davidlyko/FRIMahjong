import cv2
import numpy as np
import glob
import keras

crack1 = cv2.imread("crack1_1.jpg",cv2.COLOR_BGR2GRAY)
vCroppedImage=crack1[0:500,0:500]
vResizedImage =cv2.resize(crack1,(700,700))
#cv2.imshow("Crack 1",vResizedImage)
grayImage = cv2.cvtColor(vResizedImage,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image",grayImage)


images = sorted(glob.glob('C:/Users/geral/crack/crack_1/*.jpg'))#Directory of the first folder
images1 = sorted(glob.glob('C:/Users/geral/crack/crack_2/*.jpg'))#Directory of the second folder
labels = np.zeros((len(images)+len(images1),2)) #Used to label images in a folder. First index is the total number of pictures in the folder, and the second index is the number of folders
croppedImages=np.zeros((len(images)+len(images1),700,700))#Used to create an empty array, first index is the total number of images, second and third are x,y pixels (Have to be the same as line 23)
# for loop to store labels and images
for index,i in enumerate(images): #Same for loop for every label and image registration
    crack1 = cv2.imread(i) #reads the image for a folder
    crack1 = cv2.cvtColor(crack1,cv2.COLOR_BGR2GRAY)#makes image gray
    vCroppedImage=crack1[0:500,0:500]#crop individual image, 
    vResizedImage =cv2.resize(crack1,(700,700))#Resizes image, (700,700) should be used in line 17
    labels[index,:]=np.array([1,0])#Crack 1; the first label array will always be [1,0], any other label array after will be [0,1]. Ex: three folder, label 1: [1,0,0], label 2: [0,1,0], label three [0,0,1], etc
    croppedImages[index,:,:]=vResizedImage #Adds the resized images to the croppedImages array
print(sorted(glob.glob('C:/Users/geral/crack/crack_1/*.jpg')))
#path = "C:/Users/geral/crack/crack_1"

# for loop to store 2nd set of labels and images
for index,i in enumerate(images1): # index inside enumerate will be for the 2nd set of images
    crack2 = cv2.imread(i) #Changed the name from crack1 to crack2. 
    crack2 = cv2.cvtColor(crack2,cv2.COLOR_BGR2GRAY) #Same as previous for loop
    vCroppedImage=crack2[0:500,0:500] #Same as previous for loop
    vResizedImage =cv2.resize(crack2,(700,700)) #Same as previous for loop
    labels[index+len(images),:]=np.array([0,1])#Crack 2, ([0,0,1]) for 3 folders, etc   
    croppedImages[index+len(images),:,:]=vResizedImage # We use index+len(images) for this and the previous line because we want to store these images and labels after the 1st set

print(croppedImages[0,:,:])
print()
print(croppedImages[-1,:,:])
print()
print(labels[0,:]) # prints out [1,0]
print()
print(labels[-1,:]) # prints out [0,1]

#Model building, link: https://colab.research.google.com/drive/1xpEjpdlCpJewlAtwszSlYsVwoZxPr8EH?usp=sharing#scrollTo=F2hN3QmAIeB8
#model = Sequential() # prints out [0,1]




cv2.waitKey(0);