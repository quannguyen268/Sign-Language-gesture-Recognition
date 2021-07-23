import cv2
import os


cap = cv2.VideoCapture('/home/quan/Videos/quy.mp4')  # capturing input video
count = 0


while True:
    ret, frame = cap.read()  # extract frame
    if ret is False:
        break




    framename = '/home/quan/PycharmProjects/MiAI_FaceRecog_2/Dataset/FaceData/raw/quy/' + str(count) + ".png"




    if not os.path.exists(framename):

        cv2.imwrite(framename, frame)
    count += 1