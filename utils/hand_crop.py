import cv2
import numpy as np
import os
from imutils import contours,perspective, grab_contours
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/both_hand_images/035_002_001_frame_116.png')
image = cv2.resize(image, (1000,500), cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray,(3,3),0)
(T,thresh) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[:2]
# cnts = contours.sort_contours(cnts)[0]
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))

ref = np.zeros((96,96),dtype='uint8')
coor = [[24,0], [24,48]]
refObj = None
for (c,p) in zip(cnts,coor):
    # compute bouding box for the contour then extract the digit
    (x, y, w, h) = cv2.boundingRect(c)
    px, py = p
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)
    ref[px:px+48, py:py+48] = roi

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)

    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])
    if refObj is None:
        # unpack the ordered bounding box, then compute the
        # midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and
        # bottom-right
        (tl, tr, br, bl) = box

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # compute the Euclidean distance between the midpoints,
        # then construct the reference object
        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        refObj = (box, (cX, cY), D / 0.955)

        continue
    D1 = dist.euclidean((cX, cY), (refObj[1][0],refObj[1][1])) / refObj[2]
    print(D1)

    cv2.imshow("Image", roi)
    cv2.imshow("Image", ref)
    cv2.waitKey(0)
