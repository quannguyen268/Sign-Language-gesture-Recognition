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

refObj = None
for c in cnts:
    # compute bouding box for the contour then extract the digit
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)

    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])
    print(cX, cY)
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
        print(refObj[1])

        continue


        # draw the contours on the image

    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
    refCoords = np.vstack([refObj[0], refObj[1]])
    objCoords = np.vstack([box, (cX, cY)])
    D1 = dist.euclidean((477, 309.75), (560, 352)) / refObj[2]
    print(refObj[1][0])

    for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):

        # draw circles corresponding to the current points and
        # connect them with a line
        cv2.circle(image, (int(xA), int(yA)), 5, color, -1)
        cv2.circle(image, (int(xB), int(yB)), 5, color, -1)
        cv2.line(image, (int(xA), int(yA)), (int(xB), int(yB)),
                     color, 2)
        # compute the Euclidean distance between the coordinates,
        # and then convert the distance in pixels to distance in
        # units
        D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
        print(D)
        (mX, mY) = midpoint((xA, yA), (xB, yB))
        cv2.putText(image, "{:.2f}in".format(D), (int(mX), int(mY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)


