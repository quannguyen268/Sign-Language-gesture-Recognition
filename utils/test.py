import cv2
import numpy as np
import os
from imutils import contours,perspective
from scipy.spatial import distance as dist


#Test extract hand from original pose

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


image = cv2.imread('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/both_hand_images/035_002_001_frame_0.png')
image = cv2.resize(image, (1000,500), cv2.INTER_AREA)
hand = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(hand,(3,3),0)
(T,thresh) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[:2]
cnts = contours.sort_contours(cnts)[0]
output = cv2.merge([hand] * 3)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y - 5:y + h + 5, x - 5:x + w + 5]
    cv2.rectangle(image, (x - 2, y - 2),
                  (x + w + 4, y + h + 4), (0, 255, 0), 1)
    cv2.imshow("Output", image)
    cv2.waitKey()



