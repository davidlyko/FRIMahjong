import cv2
import numpy

crack1 = cv2.imread("crack1_1.jpg")
vCroppedImage=crack1[0:500,0:500]
vResizedImage =cv2.resize(crack1,(700,700))
cv2.imshow("Crack 1",vResizedImage)

images = []
path = "C:/Users/geral/crack/crack_1"


cv2.waitKey(0);
